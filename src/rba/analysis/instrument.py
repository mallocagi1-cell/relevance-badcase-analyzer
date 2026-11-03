"""One forward that records logit lens, direct logit attribution and attention share.

Math, stated so the report does not over-claim:

* Decision position = last non-pad token. The grade is the *next* token.
* Logit lens applies the model's own final RMSNorm, then the grade rows of
  lm_head, to each layer's residual output. The last layer's lens must match
  the model's grade distribution; that is a wiring check, not a finding.
* Direct logit attribution uses the residual direction
  d = W_U[pred] - W_U[gold]. A residual write c contributes c · d to the
  pre-norm logit difference. The decoder is a plain residual stream, so
  embed_in + sum of attention writes + sum of MLP writes equals the vector
  that enters the final norm, and the dots add up. They do NOT equal the
  post-norm logit difference: RMSNorm is not linear. Both numbers are kept.
* Head writes are the slices of the attention output projection
  (full attention: o_proj; GatedDeltaNet: out_proj). Their sum equals the
  attention write when the projection has no bias.
* Positive attribution means the component increases the predicted (wrong)
  grade relative to the gold grade.
"""

from __future__ import annotations

import torch

from rba.models.paths import ResolvedModel, attention_module, layer_kind, output_projection
from rba.util import call_model, decision_index, logits_of, move_batch


def _gather(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(tensor.size(0), device=tensor.device)
    return tensor[batch, index]


def instrument_batch(
    model: torch.nn.Module,
    batch: dict,
    resolved: ResolvedModel,
    grade_ids: list[int],
    pred_ids: torch.Tensor,
    gold_ids: torch.Tensor,
    image_token_id: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict]:
    model.eval()
    layers = resolved.layers(model)
    lm_head = resolved.lm_head(model)
    final_norm = resolved.final_norm(model)
    grade_index = torch.tensor(grade_ids, device=device, dtype=torch.long)
    pred_ids = pred_ids.to(device)
    gold_ids = gold_ids.to(device)

    moved = move_batch({k: v for k, v in batch.items() if k != "case_ids"}, device, dtype)
    hooks = []
    store: dict = {
        "layer_out": [None] * len(layers),
        "embed": None,
        "prenorm": None,
        "heads": [None] * len(layers),
        "attn": [None] * len(layers),
        "mlp": [None] * len(layers),
        "attn_row": [None] * len(layers),
        "kinds": [layer_kind(layer) for layer in layers],
    }

    def project(hidden: torch.Tensor) -> torch.Tensor:
        if final_norm is not None:
            hidden = final_norm(hidden)
        rows = lm_head.weight.index_select(0, grade_index).to(dtype=hidden.dtype)
        logits = torch.matmul(hidden, rows.T)
        return torch.softmax(logits.float(), dim=-1)

    index_holder: dict[str, torch.Tensor] = {}

    def layer_hook(layer_index):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            index = index_holder["index"]
            vector = _gather(hidden, index).detach()
            store["layer_out"][layer_index] = vector
        return hook

    def embed_hook(_module, inputs):
        hidden = inputs[0]
        index = index_holder["index"]
        store["embed"] = _gather(hidden, index).detach()
        return None

    def norm_hook(_module, inputs):
        hidden = inputs[0]
        index = index_holder["index"]
        store["prenorm"] = _gather(hidden, index).detach()
        return None

    def mlp_hook(layer_index):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            index = index_holder["index"]
            vector = _gather(hidden, index)
            direction = index_holder["direction"]
            store["mlp"][layer_index] = torch.sum(vector * direction, dim=-1).detach()
        return hook

    def attn_module_hook(layer_index):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            index = index_holder["index"]
            vector = _gather(hidden, index)
            direction = index_holder["direction"]
            store["attn"][layer_index] = torch.sum(vector * direction, dim=-1).detach()
            weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if torch.is_tensor(weights):
                batch_index = torch.arange(weights.size(0), device=weights.device)
                row = weights[batch_index, :, index, :].mean(dim=1)
                store["attn_row"][layer_index] = row.detach().float()
        return hook

    def proj_hook(layer_index, proj, head_dim, n_heads):
        def hook(module, inputs, _output):
            incoming = inputs[0]
            index = index_holder["index"]
            direction = index_holder["direction"]
            taken = _gather(incoming, index)
            weight = module.weight
            attrs = []
            for head in range(n_heads):
                sl = slice(head * head_dim, (head + 1) * head_dim)
                write = torch.nn.functional.linear(taken[:, sl], weight[:, sl])
                attrs.append(torch.sum(write * direction, dim=-1))
            store["heads"][layer_index] = torch.stack(attrs, dim=1).detach()
        return hook

    hooks.append(layers[0].register_forward_pre_hook(embed_hook))
    if final_norm is not None:
        hooks.append(final_norm.register_forward_pre_hook(norm_hook))
    for layer_index, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(layer_hook(layer_index)))
        found = attention_module(layer)
        if found is not None:
            attn_name, attn = found
            hooks.append(attn.register_forward_hook(attn_module_hook(layer_index)))
            proj_info = output_projection(attn)
            if proj_info is not None:
                proj, head_dim, n_heads = proj_info
                hooks.append(proj.register_forward_hook(proj_hook(layer_index, proj, head_dim, n_heads)))
            store["kinds"][layer_index] = attn_name
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            hooks.append(mlp.register_forward_hook(mlp_hook(layer_index)))

    try:
        with torch.no_grad():
            # Decision index and direction depend on the mask, which is known
            # before the forward. Register them before the call.
            mask = moved["attention_mask"]
            index = decision_index(mask)
            index_holder["index"] = index
            weight = lm_head.weight
            direction = weight.index_select(0, pred_ids) - weight.index_select(0, gold_ids)
            index_holder["direction"] = direction.to(dtype=weight.dtype)
            output = call_model(model, moved)
            logits = logits_of(output)
            taken = _gather(logits, index)
            grade_logits = taken.index_select(-1, grade_index)
            full_lse = torch.logsumexp(taken.float(), dim=-1)
            grade_logprob = grade_logits.float() - full_lse.unsqueeze(-1)
            grade_mass = grade_logprob.exp().sum(-1)
            full_top = taken.argmax(-1)
            cond = torch.softmax(grade_logits.float(), dim=-1)
    finally:
        for hook in hooks:
            hook.remove()

    if store["prenorm"] is None:
        # No final norm: the last layer output is the residual that is read out.
        store["prenorm"] = store["layer_out"][-1]

    records = []
    ids = batch["case_ids"]
    input_ids = moved["input_ids"]
    mask = moved["attention_mask"]
    for row in range(len(ids)):
        lens = []
        if store["embed"] is not None:
            probs = project(store["embed"][row : row + 1]).squeeze(0)
            lens.append({"layer": "embed", "probs": probs.cpu().tolist()})
        for layer_index, hidden in enumerate(store["layer_out"]):
            probs = project(hidden[row : row + 1]).squeeze(0)
            lens.append({"layer": layer_index, "probs": probs.cpu().tolist()})
        heads = None if store["heads"][0] is None and all(item is None for item in store["heads"]) else [
            None if store["heads"][layer_index] is None else store["heads"][layer_index][row].float().cpu().tolist()
            for layer_index in range(len(layers))
        ]
        attn = [
            None if store["attn"][layer_index] is None else float(store["attn"][layer_index][row].float())
            for layer_index in range(len(layers))
        ]
        mlp = [
            None if store["mlp"][layer_index] is None else float(store["mlp"][layer_index][row].float())
            for layer_index in range(len(layers))
        ]
        embed_attr = None
        if store["embed"] is not None:
            embed_attr = float(torch.sum(store["embed"][row] * direction[row]).float())
        component_sum = (embed_attr or 0.0) + sum(value or 0.0 for value in attn) + sum(value or 0.0 for value in mlp)
        prenorm_dot = float(torch.sum(store["prenorm"][row].float() * direction[row].float()))
        image_fraction, attn_share, attn_layers = _image_share(
            input_ids[row],
            mask[row],
            index[row].item(),
            [None if store["attn_row"][i] is None else store["attn_row"][i][row] for i in range(len(layers))],
            image_token_id,
        )
        records.append(
            {
                "id": ids[row],
                "lens": lens,
                "cond_probs": cond[row].cpu().tolist(),
                "grade_mass": float(grade_mass[row]),
                "full_top_is_grade": bool(int(full_top[row]) in set(grade_ids)),
                "full_top_id": int(full_top[row]),
                "logit_diff": float((grade_logits[row, pred_ids[row] and 0] * 0) + (taken[row, pred_ids[row]] - taken[row, gold_ids[row]]).float()),
                "embed_attr": embed_attr,
                "attn_attr": attn,
                "head_attr": heads,
                "mlp_attr": mlp,
                "kinds": list(store["kinds"]),
                "component_sum": component_sum,
                "prenorm_dot": prenorm_dot,
                "image_token_fraction": image_fraction,
                "attn_image_share": attn_share,
                "attn_layers_used": attn_layers,
                "hidden": [vector[row].float().cpu() for vector in store["layer_out"]],
            }
        )
    # logit_diff above is ugly but correct: taken[:, pred] - taken[:, gold].
    # Rewrite it cleanly to avoid the accidental expression.
    for row, record in enumerate(records):
        record["logit_diff"] = float((taken[row, pred_ids[row]] - taken[row, gold_ids[row]]).float())
        record["hidden"] = torch.stack(record["hidden"], dim=0)
    return records


def collect_hidden_batch(model, batch, resolved: ResolvedModel, device, dtype) -> list[torch.Tensor]:
    """Decision-position residual after each layer, with no attribution work."""
    layers = resolved.layers(model)
    moved = move_batch({k: v for k, v in batch.items() if k != "case_ids"}, device, dtype)
    saved: list = [None] * len(layers)
    index_holder: dict = {}

    def hook(layer_index):
        def _hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            saved[layer_index] = _gather(hidden, index_holder["index"]).detach().float().cpu()
        return _hook

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
    try:
        with torch.no_grad():
            index_holder["index"] = decision_index(moved["attention_mask"])
            call_model(model, moved)
    finally:
        for handle in handles:
            handle.remove()
    stacked = torch.stack(saved, dim=1)  # [batch, layers, hidden]
    return [stacked[row] for row in range(stacked.size(0))]


def _image_share(input_ids, mask, index, rows, image_token_id):
    valid = mask != 0
    length = int(valid.sum())
    if image_token_id is None:
        return 0.0, None, 0
    image = (input_ids == image_token_id) & valid
    fraction = float(image.sum()) / max(length, 1)
    used = []
    for row in rows:
        if row is None:
            continue
        # row aligns with the sequence. Renormalize over valid keys in case
        # a backend dropped the padding mask.
        weights = row.clone()
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        denom = float(weights.sum())
        if denom <= 0:
            continue
        used.append(float(weights[image].sum() / denom))
    if not used:
        return fraction, None, 0
    return fraction, sum(used) / len(used), len(used)
