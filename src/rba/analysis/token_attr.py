"""Token attribution at the language-model entrance.

The score is (logit of the predicted grade) - (logit of the gold grade).
Integrated gradients and grad×input are taken with respect to the residual
that enters layer 0, after the vision tower has already written image tokens.
That is a token attribution, not a pixel attribution. Padding positions are
dropped. Image tokens are those whose id equals the model image-token id.
"""

from __future__ import annotations

import torch

from rba.models.paths import ResolvedModel
from rba.util import call_model, decision_index, logits_of, move_batch


def attribute_batch(
    model: torch.nn.Module,
    batch: dict,
    resolved: ResolvedModel,
    pred_ids: torch.Tensor,
    gold_ids: torch.Tensor,
    image_token_id: int | None,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    method: str,
    steps: int,
) -> list[dict]:
    layers = resolved.layers(model)
    layer0 = layers[0]
    moved = move_batch({k: v for k, v in batch.items() if k != "case_ids"}, device, dtype)
    captured: dict = {}

    def capture(_module, inputs):
        captured["h0"] = inputs[0].detach()
        return None

    handle = layer0.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            call_model(model, moved)
    finally:
        handle.remove()
    baseline = captured["h0"]
    alphas = [1.0] if method == "gradxinput" else [step / steps for step in range(1, steps + 1)]
    grads = []
    for alpha in alphas:
        holder: dict = {}

        def pre_hook(_module, inputs, alpha=alpha):
            scaled = (baseline * alpha).detach().requires_grad_(True)
            holder["hidden"] = scaled
            return (scaled,) + tuple(inputs[1:])

        handle = layer0.register_forward_pre_hook(pre_hook)
        try:
            with torch.enable_grad():
                output = call_model(model, moved)
                logits = logits_of(output)
                index = decision_index(moved["attention_mask"])
                batch_index = torch.arange(logits.size(0), device=logits.device)
                diff = logits[batch_index, index, pred_ids.to(logits.device)] - logits[
                    batch_index, index, gold_ids.to(logits.device)
                ]
                diff.sum().backward()
                grads.append(holder["hidden"].grad.detach())
        finally:
            handle.remove()
            model.zero_grad(set_to_none=True)
    mean_grad = torch.stack(grads, dim=0).mean(0)
    scores = (baseline * mean_grad).sum(dim=-1).float().cpu()
    input_ids = moved["input_ids"].cpu()
    mask = moved["attention_mask"].cpu()
    records = []
    for row, case_id in enumerate(batch["case_ids"]):
        valid = mask[row] != 0
        score = scores[row]
        image = (input_ids[row] == image_token_id) & valid if image_token_id is not None else torch.zeros_like(valid)
        text = valid & ~image
        abs_score = score.abs()
        abs_total = float(abs_score[valid].sum()) + 1e-12
        text_signed = float(score[text].sum())
        image_signed = float(score[image].sum())
        records.append(
            {
                "id": case_id,
                "method": method,
                "text_signed": text_signed,
                "image_signed": image_signed,
                "text_abs_share": float(abs_score[text].sum()) / abs_total,
                "image_abs_share": float(abs_score[image].sum()) / abs_total,
                "n_text": int(text.sum()),
                "n_image": int(image.sum()),
                "top_text": _top_tokens(score, text, input_ids[row], tokenizer, limit=6),
            }
        )
    return records


def _top_tokens(score, text_mask, input_ids, tokenizer, limit: int) -> list[dict]:
    positions = torch.nonzero(text_mask, as_tuple=False).flatten()
    if positions.numel() == 0:
        return []
    values = score[positions]
    order = torch.argsort(values.abs(), descending=True)[:limit]
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    rows = []
    for pick in order.tolist():
        position = int(positions[pick])
        token_id = int(input_ids[position])
        if convert is not None:
            try:
                token = convert(token_id)
            except Exception:
                token = f"#{token_id}"
        else:
            token = f"#{token_id}"
        raw_token = str(token)
        decode = getattr(tokenizer, "decode", None)
        context = None
        if callable(decode):
            try:
                token = decode([token_id], skip_special_tokens=False)
                context = decode(input_ids[max(0, position-2):position+3].tolist(), skip_special_tokens=False)
            except (TypeError, ValueError):
                pass
        if not token or "\ufffd" in str(token):
            token = raw_token
        token = str(token).replace("\n", "\\n")
        if len(token) > 24:
            token = token[:24] + "…"
        rows.append({"position": position, "id": token_id, "token": token, "display": token, "raw_token": raw_token, "context": context, "score": float(values[pick])})
    return rows
