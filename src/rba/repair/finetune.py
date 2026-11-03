"""Targeted repair training for when patches are not enough.

Loss is on the single grade token: cross-entropy over the grade tokens only, at the
decision position (the last token). Badcases get an extra hinge that pushes the grade the
model currently confuses them with below the gold grade (hard negative from the confusion
group). Control cases are replayed with CE plus a KL term toward the original model's grade
distribution, which is what limits regression. Training defaults to LoRA restricted to the
layers/modules the analysis localized; the adapters are merged into the weights at the end.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from rba import __version__
from rba.console import info, reset, warn
from rba.errors import RBAError
from rba.models.paths import attention_module, output_projection
from rba.repair.common import check_output_dir, save_checkpoint, score_items
from rba.repair.evaluate import INVALID_GRADE, compare, metric_block
from rba.util import call_model, file_fingerprint, logits_of, move_batch, to_builtin

VERIFIED = {"消融有效", "已验证"}


@dataclass
class FinetuneOptions:
    out: str = "runs/finetune"
    mode: str = "lora"              # lora | full
    targets: str = "localized"      # localized | all
    layers: list[int] | None = None
    modules: list[str] | None = None  # suffixes like o_proj, out_proj, down_proj, lm_head
    steps: int = 100
    lr: float | None = None
    train_batch_size: int = 4
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    hard_negative_weight: float = 1.0
    margin: float = 1.0
    replay_ratio: float = 0.5
    kl_weight: float = 1.0
    control_holdout: float = 0.5
    holdout: float = 0.3
    extra_train: str | None = None
    balance_groups: bool = True
    grad_checkpointing: bool = False
    max_grad_norm: float = 1.0
    seed: int = 0
    overwrite: bool = False
    report: dict | None = field(default=None, repr=False)


def _hash_fraction(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"ft:{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _require_peft():
    try:
        import peft  # noqa: F401
    except ImportError as exc:
        raise RBAError(
            "LoRA 训练需要 peft，当前环境没有安装。",
            "pip install 'rba[finetune]'（或 pip install peft）。不想装就用 --mode full（只训练定位到的模块）。",
        ) from exc
    return peft


# --------------------------------------------------------------------------- localization


def localized_targets(report: dict | None) -> tuple[list[int], bool, list[str]]:
    """Layers (and whether lm_head) named by the analysis' patches. Verified first."""
    if not report:
        return [], False, []
    picked: list[tuple[int, dict]] = []
    for group in report.get("groups") or []:
        for item in group.get("hypotheses") or []:
            patch = item.get("patch") or {}
            if not patch.get("method"):
                continue
            rank = 0 if item.get("status") in VERIFIED else (1 if item.get("recommend") else 2)
            picked.append((rank, patch))
    if not picked:
        return [], False, []
    best = min(rank for rank, _ in picked)
    chosen = [patch for rank, patch in picked if rank == best]
    layers: set[int] = set()
    lm_head = False
    sources = []
    for patch in chosen:
        method = patch["method"]
        if method in {"head_scale", "mlp_zero"} and patch.get("layer") is not None:
            layers.add(int(patch["layer"]))
        elif method == "steering":
            layers.update(int(v) for v in (patch.get("layers") or [patch.get("layer")]) if v is not None)
        elif method == "lm_head_blend":
            lm_head = True
        sources.append(f"{method}@{patch.get('layer', patch.get('layers', '-'))}")
    if best == 2:
        warn("报告里没有「消融有效」或推荐的补丁，按全部假设涉及的层来定位。这只是弱证据。")
    return sorted(layers), lm_head, sources


def target_modules(model, resolved, layers: list[int], lm_head: bool, suffixes: list[str] | None) -> list[str]:
    names = {id(module): name for name, module in model.named_modules()}
    stack = resolved.layers(model)
    targets: list[str] = []
    for index in layers:
        if index < 0 or index >= len(stack):
            raise RBAError(f"层号 {index} 超出范围 0..{len(stack) - 1}。", "--layers 写报告里列出的层。")
        layer = stack[index]
        if suffixes:
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear) and name.split(".")[-1] in suffixes:
                    targets.append(names[id(module)])
            continue
        found = attention_module(layer)
        proj = output_projection(found[1]) if found is not None else None
        if proj is not None:
            targets.append(names[id(proj[0])])
        down = getattr(getattr(layer, "mlp", None), "down_proj", None)
        if isinstance(down, nn.Linear):
            targets.append(names[id(down)])
    if lm_head and (not suffixes or "lm_head" in suffixes):
        targets.append(names[id(resolved.lm_head(model))])
    if not targets:
        raise RBAError("按定位结果没找到可训练的线性层。", "用 --modules 指定，例如 --modules o_proj,out_proj,down_proj；或 --targets all。")
    return sorted(set(targets))


# --------------------------------------------------------------------------- LoRA without PeftModel wrappers


def add_lora(model, targets: list[str], opts: FinetuneOptions) -> list[nn.Parameter]:
    peft = _require_peft()
    short = [name for name in targets]
    config = peft.LoraConfig(r=opts.lora_r, lora_alpha=opts.lora_alpha, lora_dropout=opts.lora_dropout,
                             target_modules=short, bias="none")
    peft.inject_adapter_in_model(config, model)
    params = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()
            param.requires_grad_(True)
            params.append(param)
        else:
            param.requires_grad_(False)
    if not params:
        raise RBAError("LoRA 没有挂上任何模块。", f"检查目标模块名：{targets[:4]}")
    return params


def merge_lora(model) -> int:
    from peft.tuners.lora import LoraLayer

    merged = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, LoraLayer):
            base = module.get_base_layer()
            dtype = base.weight.dtype
            module.merge(safe_merge=True)
            base.weight.data = base.weight.data.to(dtype)
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, base)
            merged += 1
    return merged


# --------------------------------------------------------------------------- training


def _grade_logits_train(model, loaded, batch, grade_id_list) -> torch.Tensor:
    moved = move_batch({k: v for k, v in batch.items() if k != "case_ids"}, loaded.device, loaded.dtype)
    payload = dict(moved)
    if loaded.kind == "hf":
        payload["logits_to_keep"] = 1
    logits = logits_of(call_model(model, payload))
    from rba.util import decision_index

    if logits.size(1) == 1:
        taken = logits[:, -1]
    else:
        index = decision_index(moved["attention_mask"])
        taken = logits[torch.arange(logits.size(0), device=logits.device), index]
    columns = torch.tensor(grade_id_list, device=taken.device, dtype=torch.long)
    return taken.float().index_select(-1, columns)


def finetune(cfg, opts: FinetuneOptions) -> dict:
    from rba.models.loader import load_models
    from rba.models.paths import resolve_paths
    from rba.pipeline import _batch, _content_key, _effective_batch, _encode_all, _pad_id, _validate_and_load
    from rba.schema import load_cases

    reset()
    if opts.mode not in {"lora", "full"}:
        raise RBAError(f"--mode 只能是 lora 或 full，收到 {opts.mode}。", "默认 lora。")
    if opts.mode == "lora":
        _require_peft()
    out = Path(opts.out).resolve()
    check_output_dir(out, opts.overwrite)
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    cfg.analyses.delta_w = False
    if not cfg.holdout_fraction:
        cfg.holdout_fraction = opts.holdout
        if opts.holdout > 0:
            warn(f"分析时没有留出验收集。这次按 id 哈希留出 {opts.holdout:.0%} 坏例不参与训练，只用来评估；"
                 "注意它们参与过分析阶段选层，不是完全独立的。")
    bad_cases, good_cases, split = _validate_and_load(cfg)
    extra_cases = load_cases(opts.extra_train, cfg.grades, what="额外训练数据") if opts.extra_train else []
    loaded = load_models(cfg)
    model = loaded.ft
    resolved = resolve_paths(model, cfg.module_paths, loaded.preset)
    grades = cfg.grades
    grade_ids = loaded.grade_ids
    grade_id_list = [grade_ids[g] for g in grades]
    excluded = []
    bad_enc, good_enc = _encode_all(cfg, loaded, bad_cases, good_cases, excluded)
    extra_enc, _ = _encode_all(cfg, loaded, extra_cases, [], excluded) if extra_cases else ([], [])
    holdout_ids = set((split or {}).get("holdout_ids") or [])
    bad_train = [item for item in bad_enc if item.id not in holdout_ids]
    bad_eval = [item for item in bad_enc if item.id in holdout_ids]
    eval_content = {_content_key(item.case) for item in bad_eval}
    good_enc = [item for item in good_enc if _content_key(item.case) not in eval_content]
    extra_enc = [item for item in extra_enc if _content_key(item.case) not in eval_content]
    good_eval = [item for item in good_enc if _hash_fraction(opts.seed, item.id) < opts.control_holdout]
    eval_ids = {item.id for item in good_eval}
    good_train = [item for item in good_enc if item.id not in eval_ids]
    if not bad_train and not extra_enc:
        raise RBAError("没有可训练的坏例（全被留作验收或被剔除）。", "调小 --holdout，或用 --extra-train 补训练数据。")
    if not good_train:
        warn("没有对照集用于回放，训练无法约束回退。强烈建议传 --goodcases。")
    if not bad_eval:
        warn("没有留出的坏例，修复率只能在训练坏例上看（会偏乐观）。")
    eval_batch = _effective_batch(cfg, loaded.device, bad_enc + good_enc)

    # ---- targets
    if opts.targets == "all":
        layers = list(range(len(resolved.layers(model))))
        lm_head = False
        sources = ["--targets all"]
    elif opts.layers:
        layers, lm_head, sources = sorted(set(opts.layers)), False, ["--layers"]
    else:
        layers, lm_head, sources = localized_targets(opts.report)
        if not layers and not lm_head:
            n = len(resolved.layers(model))
            layers = list(range(max(0, n - 2), n))
            sources = ["默认：最后两层"]
            warn("没有从报告里拿到定位结果（没传 --run 或报告里没有补丁），默认训练最后两层的输出投影和 down_proj。")
    if opts.modules and "lm_head" in opts.modules:
        lm_head = True
    targets = target_modules(model, resolved, layers, lm_head, opts.modules)
    info(f"训练目标（{opts.mode}）：{len(targets)} 个模块，层 {layers}" + ("，含 lm_head" if lm_head else "") + "。")

    # ---- before scores (also used for hard-negative mining and KL replay)
    def scored(items, desc):
        return score_items(loaded, items, grade_id_list, grades, cfg.decode_protocol, eval_batch, desc)

    before_train = scored(bad_train + extra_enc, "微调·训练集改前")
    before_bad_eval = scored(bad_eval, "微调·验收集改前")
    before_good_train = scored(good_train, "微调·回放集改前")
    before_good_eval = scored(good_eval, "微调·对照改前")
    confusions = {(row["label"], row["pred"]) for row in before_train if row["pred"] != row["label"] and row["pred"] != INVALID_GRADE}

    examples = []  # (item, label_index, negative_index or None, kind, ref_logits or None, weight)
    groups: dict[str, list[int]] = {}
    for item, row in zip(bad_train + extra_enc, before_train):
        neg = grades.index(row["pred"]) if row["pred"] not in {row["label"], INVALID_GRADE} else None
        key = f"{row['label']}->{row['pred']}" if neg is not None else "correct"
        groups.setdefault(key, []).append(len(examples))
        examples.append((item, grades.index(item.case.label), neg, "bad", None, 1.0))
    replay = []
    n_hard_controls = 0
    for item, row in zip(good_train, before_good_train):
        logits = torch.tensor(row["grade_logits"])
        order = logits.argsort(descending=True).tolist()
        runner_up = grades[order[1]] if grades[order[0]] == row["label"] else grades[order[0]]
        hard = (row["label"], runner_up) in confusions
        n_hard_controls += int(hard)
        replay.append((item, grades.index(item.case.label), grades.index(runner_up) if hard else None, "control", logits, 2.0 if hard else 1.0))
    info(f"训练坏例 {len(bad_train)} 条" + (f" + 额外 {len(extra_enc)} 条" if extra_enc else "") + f"，混淆组 {len([k for k in groups if k != 'correct'])} 个；"
         f"回放对照 {len(replay)} 条（其中 {n_hard_controls} 条是同一混淆方向上的难对照，加倍权重）。")

    # ---- trainable params
    if opts.mode == "lora":
        params = add_lora(model, targets, opts)
        lr = opts.lr or 1e-4
    else:
        wanted = set(targets) if opts.targets != "all" else None
        params = []
        for name, param in model.named_parameters():
            module_name = name.rsplit(".", 1)[0]
            train = wanted is None or module_name in wanted
            param.requires_grad_(train)
            if train:
                params.append(param)
        lr = opts.lr or 1e-5
        if loaded.dtype in {torch.bfloat16, torch.float16}:
            warn("全量模式直接在低精度权重上更新，小学习率下可能更新不动；优先用 LoRA。")
    n_trainable = sum(p.numel() for p in params)
    info(f"可训练参数 {n_trainable:,}，学习率 {lr:g}，{opts.steps} 步，每步 {opts.train_batch_size} 条。")
    if opts.grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.train()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    pad_id = _pad_id(loaded.tokenizer)
    single = any(item.model_inputs is not None for item, *_ in examples + replay)
    rng = random.Random(opts.seed)
    group_keys = list(groups)

    def draw_bad():
        if opts.balance_groups and len(group_keys) > 1:
            return examples[rng.choice(groups[rng.choice(group_keys)])]
        return rng.choice(examples)

    history = []
    started = time.time()
    n_replay_per_step = round(opts.train_batch_size * opts.replay_ratio) if replay else 0
    n_bad_per_step = max(1, opts.train_batch_size - n_replay_per_step)
    for step in range(opts.steps):
        chosen = [draw_bad() for _ in range(n_bad_per_step)] + [rng.choice(replay) for _ in range(n_replay_per_step)]
        chunks = [[c] for c in chosen] if single else [chosen]
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        parts = {"ce": 0.0, "hinge": 0.0, "kl": 0.0}
        for chunk in chunks:
            batch = _batch(loaded, [c[0] for c in chunk], pad_id)
            z = _grade_logits_train(model, loaded, batch, grade_id_list)
            labels = torch.tensor([c[1] for c in chunk], device=z.device)
            weights = torch.tensor([c[5] for c in chunk], device=z.device)
            ce = nn.functional.cross_entropy(z, labels, reduction="none")
            loss = (ce * weights).sum()
            parts["ce"] += float(ce.detach().sum())
            for i, c in enumerate(chunk):
                if c[2] is not None and opts.hard_negative_weight > 0:
                    hinge = torch.relu(opts.margin - (z[i, c[1]] - z[i, c[2]]))
                    loss = loss + opts.hard_negative_weight * c[5] * hinge
                    parts["hinge"] += float(hinge.detach())
                if c[3] == "control" and opts.kl_weight > 0:
                    ref = torch.log_softmax(c[4].to(z.device), -1)
                    kl = nn.functional.kl_div(torch.log_softmax(z[i], -1), ref, log_target=True, reduction="sum")
                    loss = loss + opts.kl_weight * kl
                    parts["kl"] += float(kl.detach())
            loss = loss / len(chosen)
            loss.backward()
            total += float(loss.detach())
        if opts.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(params, opts.max_grad_norm)
        optimizer.step()
        history.append({"step": step + 1, "loss": total, **{k: v / len(chosen) for k, v in parts.items()}})
        if (step + 1) % max(1, opts.steps // 10) == 0 or step == 0:
            info(f"第 {step + 1}/{opts.steps} 步：loss {total:.4f}（CE {parts['ce'] / len(chosen):.3f}，难负例 {parts['hinge'] / len(chosen):.3f}，回放 KL {parts['kl'] / len(chosen):.3f}）")
        if not math.isfinite(total):
            raise RBAError(f"第 {step + 1} 步 loss 变成 {total}。", "调小 --lr，或换 --mode lora。")
    if opts.grad_checkpointing and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    merged = merge_lora(model) if opts.mode == "lora" else 0
    if merged:
        info(f"已把 {merged} 个 LoRA 适配器合并回权重。")

    after_train = scored(bad_train + extra_enc, "微调·训练集改后")
    after_bad_eval = scored(bad_eval, "微调·验收集改后")
    after_good_eval = scored(good_eval, "微调·对照改后")

    def block(b, a):
        if not b:
            return None
        cmp = compare(b, a)
        cmp["before"] = metric_block(b, grades)
        cmp["after"] = metric_block(a, grades)
        cmp["n_invalid_after"] = sum(1 for r in a if r["pred"] == INVALID_GRADE)
        return cmp

    result = {
        "rba_version": __version__,
        "mode": opts.mode,
        "targets": targets,
        "layers": layers,
        "localized_from": sources,
        "n_trainable": n_trainable,
        "lr": lr,
        "options": {k: v for k, v in opts.__dict__.items() if k != "report"},
        "n_train_bad": len(bad_train),
        "n_extra": len(extra_enc),
        "n_replay": len(replay),
        "n_hard_controls": n_hard_controls,
        "confusion_pairs": sorted(f"{a}->{b}" for a, b in confusions),
        "history": history,
        "seconds": time.time() - started,
        "train": block(before_train, after_train),
        "bad_holdout": block(before_bad_eval, after_bad_eval),
        "good_holdout": block(before_good_eval, after_good_eval),
        "excluded": [{"id": item.id, "reason": item.reason} for item in excluded],
        "source_model": str(loaded.ft_path),
        "source_fingerprint": file_fingerprint(loaded.ft_path) if loaded.kind == "hf" else None,
    }
    model_dir = out / "model"
    if loaded.kind == "hf":
        save_checkpoint(model, loaded.ft_path, model_dir)
        result["model_dir"] = str(model_dir)
        result["model_fingerprint"] = file_fingerprint(model_dir)
    else:
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir / "model_state.pt")
        result["model_dir"] = str(model_dir)
        warn("内置小模型不是 HF 类，只存了 state_dict（model_state.pt），仅供测试。")
    (out / "finetune.json").write_text(json.dumps(to_builtin(result), ensure_ascii=False, indent=2), encoding="utf-8")
    for key, label in (("bad_holdout", "留出坏例"), ("good_holdout", "留出对照"), ("train", "训练坏例（会偏乐观）")):
        cmp = result[key]
        if cmp:
            fix = "—" if cmp["fix_rate"] is None else f"{cmp['fix_rate']:.1%}"
            reg = "—" if cmp["regression_rate"] is None else f"{cmp['regression_rate']:.1%}"
            acc_b = cmp["before"]["accuracy"]
            acc_a = cmp["after"]["accuracy"]
            info(f"{label} {cmp['n']} 条：准确率 {acc_b:.1%} → {acc_a:.1%}，修复率 {fix}，回退率 {reg}。")
    info(f"结果：{out / 'finetune.json'}；模型：{model_dir}")
    return to_builtin(result)
