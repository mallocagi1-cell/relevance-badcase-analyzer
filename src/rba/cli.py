"""Chinese CLI. `rba analyze --base --ft --badcases` needs no config file."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

from rba import __version__
from rba.config import DEFAULT_TEMPLATE, Config, apply_overrides, config_from_used
from rba.console import error, info, lines
from rba.errors import RBAError
from rba.models.loader import detect_directory
from rba.pipeline import analyze, evaluate_patch

EPILOG = """
示例：
  rba demo
  python -m rba analyze --config example.yaml --dry-run
  rba analyze --base /path/base --ft /path/ft --badcases bad.jsonl --goodcases good.jsonl
  rba init --base /path/base --ft /path/ft --badcases bad.jsonl
  rba eval --run runs/example --patch runs/example/patches/某组_h1.json
  rba export --run runs/example --patch runs/example/patches/某组_h1.json --out exported/
  rba finetune --run runs/example --out runs/example/finetune --steps 200

退出码：0 成功；2 配置或数据错误（已给出建议）；3 分析完成但大模型调用失败；1 程序内部错误（traceback 在输出目录的 error.log）。
""".strip()

BUILTIN_MODELS = {"tiny", "qwen3_5_tiny"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rba",
        description="分析微调相关性模型的坏例，定位最小修改，并在验收集和对照集上评估。",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"rba {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="按模型目录生成一份可改的配置")
    init.add_argument("--base", help="基座模型目录，或 tiny")
    init.add_argument("--ft", help="微调模型目录，或 tiny")
    init.add_argument("--badcases", help="坏例 jsonl/csv")
    init.add_argument("--goodcases", help="对照集 jsonl/csv")
    init.add_argument("--out", default="rba.yaml", help="写出的配置路径，默认 rba.yaml")

    analyze_p = sub.add_parser("analyze", help="跑分析并写 HTML/Markdown 报告")
    _add_common(analyze_p)
    analyze_p.add_argument("--dry-run", action="store_true", help="只写提示词，不调用大模型")
    analyze_p.add_argument("--fresh", action="store_true", help="忽略并覆盖该输出目录里的缓存")
    analyze_p.add_argument("--max-cases", type=int, help="只取前 N 条坏例，便于试跑")
    analyze_p.add_argument("--holdout", type=float, help="按 id 哈希留出这一比例的坏例给 rba eval 做独立验收，例如 0.5")

    eval_p = sub.add_parser("eval", help="在验收集（或发现集）和对照集上评估一个补丁")
    _add_common(eval_p)
    eval_p.add_argument("--patch", help="分析报告写出的补丁 JSON")
    eval_p.add_argument("--run", help="某次分析的输出目录。会完整恢复那次的配置（模板、等级、解码协议、模块路径等）")

    export_p = sub.add_parser("export", help="把验收过的补丁精确烘焙进权重，存成不依赖 rba 的标准 checkpoint")
    _add_common(export_p)
    export_p.add_argument("--run", help="某次分析的输出目录，恢复那次的配置；核对用的样本也来自那次")
    export_p.add_argument("--patch", nargs="+", required=True, help="一个或多个补丁 JSON，按给出的顺序烘焙")
    export_p.add_argument("--out", required=True, help="导出目录（会写 config.json、权重、分词器文件和 rba_export_manifest.json）")
    export_p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], help="加载、烘焙、保存用的精度。默认沿用配置")
    export_p.add_argument("--force", action="store_true", help="允许导出未经复核（untested）或被复核否定的补丁")
    export_p.add_argument("--allow-all-positions", action="store_true", help="允许把只在决策位置复核过的补丁烘焙成作用于所有位置")
    export_p.add_argument("--untie-embeddings", action="store_true", help="lm_head 与 embedding 共享权重时，拆开后再改 lm_head 行")
    export_p.add_argument("--tolerance", type=float, help="等价性核对的 logit 容差。默认 float32 1e-3、bfloat16 0.25、float16 0.05")
    export_p.add_argument("--overwrite", action="store_true", help="--out 已有内容时清空后再写")
    export_p.add_argument("--check-only", action="store_true", help="只检查补丁能不能烘焙，不写模型")

    ft_p = sub.add_parser("finetune", help="补丁不够时做针对性修复训练（默认 LoRA，只训定位到的层）")
    _add_common(ft_p)
    ft_p.add_argument("--run", help="某次分析的输出目录。用它的配置和报告里定位到的层/模块")
    ft_p.add_argument("--out", required=True, help="输出目录：model/（合并后的 checkpoint）和 finetune.json")
    ft_p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], help="训练精度。2B 单卡用 bfloat16")
    ft_p.add_argument("--mode", choices=["lora", "full"], default="lora", help="lora（默认，需要 peft）或 full（直接训练目标模块的全部权重）")
    ft_p.add_argument("--targets", choices=["localized", "all"], default="localized", help="localized 只训定位到的层；all 训所有层")
    ft_p.add_argument("--layers", help="手动指定层号，逗号分隔，例如 3,5。会盖掉报告的定位")
    ft_p.add_argument("--modules", help="模块名后缀，逗号分隔，例如 o_proj,out_proj,down_proj,lm_head。默认是各层的注意力输出投影和 down_proj")
    ft_p.add_argument("--extra-train", help="额外训练数据（jsonl/csv，格式同坏例）。只进训练，不进评估")
    ft_p.add_argument("--steps", type=int, default=100, help="优化步数，默认 100")
    ft_p.add_argument("--lr", type=float, help="学习率。默认 LoRA 1e-4，full 1e-5")
    ft_p.add_argument("--train-batch-size", type=int, default=4, help="每步样本数（坏例 + 回放），默认 4")
    ft_p.add_argument("--lora-r", type=int, default=8, help="LoRA 秩，默认 8")
    ft_p.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha，默认 16")
    ft_p.add_argument("--hard-negative-weight", type=float, default=1.0, help="难负例间隔损失的权重，0 表示关掉，默认 1")
    ft_p.add_argument("--margin", type=float, default=1.0, help="金标 logit 至少要比混淆等级高多少，默认 1")
    ft_p.add_argument("--replay-ratio", type=float, default=0.5, help="每步里对照回放占的比例，默认 0.5")
    ft_p.add_argument("--kl-weight", type=float, default=1.0, help="回放样本上贴近原模型等级分布的 KL 权重，默认 1")
    ft_p.add_argument("--control-holdout", type=float, default=0.5, help="对照集里留作评估、不参与回放的比例，默认 0.5")
    ft_p.add_argument("--train-holdout", type=float, default=0.3, help="分析时没留验收集时，这次留出多少坏例只做评估，默认 0.3")
    ft_p.add_argument("--no-balance-groups", action="store_true", help="不按混淆组均衡采样（默认每组被抽到的机会相同）")
    ft_p.add_argument("--grad-checkpointing", action="store_true", help="梯度检查点，省显存、慢一些")
    ft_p.add_argument("--overwrite", action="store_true", help="--out 已有内容时清空后再写")

    report_p = sub.add_parser("report", help="从已有的分析目录重新生成可视化报告 report.html（不重算）")
    report_p.add_argument("run_dir", help="rba analyze / rba demo 的输出目录")
    report_p.add_argument("--language", choices=["zh", "en"], default=None, help="报告语言，默认沿用该次分析的 language（zh）")

    demo = sub.add_parser("demo", help="用内置小模型和样例数据干跑一遍")
    demo.add_argument("--arch", choices=["tiny", "qwen3_5_tiny"], default="tiny", help="tiny 是同结构的替身；qwen3_5_tiny 是真实类的随机小模型")
    demo.add_argument("--output-dir", default="runs/demo")

    args = parser.parse_args(argv)
    output_hint: Path | None = None
    try:
        if args.cmd == "init":
            _cmd_init(args)
        elif args.cmd == "analyze":
            cfg = _cfg_from_analyze_args(args)
            output_hint = Path(cfg.output_dir) if cfg.output_dir else None
            result = analyze(cfg)
            if result.get("llm_failures"):
                error(
                    f"{len(result['llm_failures'])} 组的大模型调用失败。报告已写出：{result['html']}",
                    "修好 llm.base_url / llm.model / 密钥后重跑同一条命令，分析阶段会从缓存接着用。",
                )
                sys.exit(3)
        elif args.cmd == "eval":
            cfg, patch, run_meta = _cfg_from_eval_args(args)
            output_hint = Path(cfg.output_dir) if cfg.output_dir else None
            result = evaluate_patch(cfg, patch, run_meta)
            out = Path(cfg.output_dir or "runs/eval")
            out.mkdir(parents=True, exist_ok=True)
            path = out / "eval.json"
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            info(f"验收结果：{path}")
            from rba.report.visual import write_eval_report

            info(f"可视化：{write_eval_report(result, out, cfg.language)}")
            run_dir = Path(args.run) if getattr(args, "run", None) else out
            if (run_dir / "report.json").exists() and (out == run_dir or out.parent == run_dir):
                from rba.report.visual import build_visual_report

                info(f"已把验收结果并入：{build_visual_report(run_dir)}")
        elif args.cmd == "export":
            from rba.repair.bake import export_patches

            cfg, _meta = _cfg_from_run_args(args, "export")
            patches = [(Path(p).name, _load_patch(p)) for p in args.patch]
            export_patches(cfg, patches, Path(args.out), force=args.force, allow_all_positions=args.allow_all_positions,
                           untie=args.untie_embeddings, overwrite=args.overwrite, tolerance=args.tolerance, dry_run=args.check_only)
        elif args.cmd == "finetune":
            from rba.repair.finetune import FinetuneOptions, finetune

            cfg, _meta = _cfg_from_run_args(args, "finetune")
            output_hint = Path(args.out)
            finetune(cfg, _finetune_options(args))
        elif args.cmd == "report":
            from rba.report.visual import build_visual_report

            info(f"可视化报告：{build_visual_report(Path(args.run_dir), args.language)}")
        elif args.cmd == "demo":
            output_hint = Path(args.output_dir)
            _cmd_demo(args)
        else:
            parser.error(f"未知命令 {args.cmd}")
    except RBAError as exc:
        error(exc.message, exc.hint)
        sys.exit(2)
    except KeyboardInterrupt:
        error("已中断。", "已完成的阶段留在输出目录里。用同一个命令再跑会从缓存继续；想重算就加 --fresh。")
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - last resort: never a bare traceback for the user
        log_path = _write_error_log(output_hint, exc)
        error(
            f"程序内部错误：{type(exc).__name__}: {str(exc)[:300]}",
            f"完整 traceback 已写到 {log_path}。请带着这个文件和命令行反馈。已完成的阶段留在缓存里。",
        )
        sys.exit(1)


def _write_error_log(output_hint: Path | None, exc: BaseException) -> Path:
    folder = output_hint if output_hint is not None else Path.cwd() / "runs"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "error.log"
    except OSError:
        path = Path.cwd() / "rba-error.log"
    body = "\n".join(lines()) + "\n\n" + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        path.write_text(body, encoding="utf-8")
    except OSError:
        path = Path.cwd() / "rba-error.log"
        path.write_text(body, encoding="utf-8")
    return path


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="YAML 配置。命令行里的同名字段会盖掉文件")
    parser.add_argument("--base", help="基座模型目录，或 tiny / qwen3_5_tiny")
    parser.add_argument("--ft", help="微调模型目录，或 tiny / qwen3_5_tiny")
    parser.add_argument("--badcases", help="坏例 jsonl 或 csv")
    parser.add_argument("--goodcases", help="对照集。不传则只报告坏例，看不出回退")
    parser.add_argument("--output-dir", help="报告目录。不传则按数据指纹放进 runs/")
    parser.add_argument("--device", help="auto、cpu 或 cuda")
    parser.add_argument("--batch-size", type=int, help="0 表示自动，显存不够会减半；带处理器视觉输入时固定为 1")
    parser.add_argument("--seed", type=int, help="内置小模型的随机种子")
    parser.add_argument("--decode-protocol", choices=["constrained", "free"], help="constrained 只在等级 token 里取 argmax；free 用全词表 argmax，等级外算非法输出")
    parser.add_argument("--language", choices=["zh", "en"], help="分析结论语言：zh（默认）让大模型用中文回答；en 让大模型用英文回答。报告本身固定为中文")
    parser.add_argument("--allow-missing-images", action="store_true", help="图片读不到时剔除该样本并写进报告，而不是停下")


def _cfg_from_analyze_args(args) -> Config:
    if not args.config and not (args.base and args.ft and args.badcases):
        raise RBAError(
            "缺少输入。",
            "零配置：rba analyze --base 基座 --ft 微调 --badcases 坏例.jsonl\n"
            "或：rba analyze --config example.yaml --dry-run\n"
            "先看效果：rba demo",
        )
    cfg = Config.load(args.config) if args.config else Config()
    if not args.config:
        cfg.goodcases = None
    return _apply_cli(cfg, args)


def _apply_cli(cfg: Config, args) -> Config:
    """Only fields the user typed override the config; everything else stays."""
    overrides = {
        "base_model": getattr(args, "base", None),
        "ft_model": getattr(args, "ft", None),
        "badcases": getattr(args, "badcases", None),
        "goodcases": getattr(args, "goodcases", None),
        "output_dir": getattr(args, "output_dir", None),
        "device": getattr(args, "device", None),
        "batch_size": getattr(args, "batch_size", None),
        "seed": getattr(args, "seed", None),
        "decode_protocol": getattr(args, "decode_protocol", None),
        "holdout_fraction": getattr(args, "holdout", None),
        "language": getattr(args, "language", None),
    }
    cfg = apply_overrides(cfg, {key: value for key, value in overrides.items() if value is not None})
    if getattr(args, "allow_missing_images", False):
        cfg.missing_image = "exclude"
    if getattr(args, "dry_run", False):
        cfg.dry_run = True
    if getattr(args, "fresh", False):
        cfg.fresh = True
        cfg.resume = False
    if getattr(args, "max_cases", None):
        cfg.max_cases = args.max_cases
    _resolve_cli_paths(cfg, args)
    cfg.validate_static()
    return cfg


def _resolve_cli_paths(cfg: Config, args) -> None:
    if getattr(args, "badcases", None):
        cfg.badcases = str(Path(args.badcases).resolve())
    if getattr(args, "goodcases", None):
        cfg.goodcases = str(Path(args.goodcases).resolve())
    if getattr(args, "base", None) and args.base not in BUILTIN_MODELS:
        cfg.base_model = str(Path(args.base).resolve())
    if getattr(args, "ft", None) and args.ft not in BUILTIN_MODELS:
        cfg.ft_model = str(Path(args.ft).resolve())


def _cfg_from_eval_args(args):
    if not args.patch:
        raise RBAError(
            "验收需要一个补丁。",
            "用分析报告 patches/ 里的 JSON：rba eval --run runs/某次 --patch runs/某次/patches/组_h1.json",
        )
    cfg, run_meta = _cfg_from_run_args(args, "eval")
    patch = _load_patch(args.patch)
    return cfg, patch, run_meta


def _load_patch(value: str) -> dict:
    path = Path(value)
    if not path.exists():
        raise RBAError(f"找不到补丁 {path}。", "路径写报告里的那一行，不要手改字段名。")
    try:
        patch = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RBAError(f"补丁不是合法 JSON：{exc}", "重新从报告生成，或检查有没有被编辑坏。") from exc
    if not isinstance(patch, dict) or "method" not in patch:
        raise RBAError("补丁里没有 method。", "用 head_scale、mlp_zero、steering、lm_head_blend 或 ablate_tokens。")
    return patch


def _cfg_from_run_args(args, what: str):
    run_meta: dict = {}
    if args.run:
        used_path = Path(args.run) / "config.used.json"
        if not used_path.exists():
            raise RBAError(
                f"{args.run} 里没有 config.used.json。",
                "把 --run 指到某次 analyze 的输出目录，或改用 --config / --ft / --badcases。",
            )
        try:
            used = json.loads(used_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RBAError(f"{used_path} 不是合法 JSON：{exc}", "重新跑一次 rba analyze 生成它。") from exc
        run_meta = used.get("runtime") or {}
        cfg = config_from_used(used)
        if args.config:
            raise RBAError("--run 和 --config 不能同时用。", "--run 已经完整恢复那次分析的配置；要换配置就不要传 --run。")
        if getattr(args, "output_dir", None) is None and what == "eval":
            args.output_dir = str(Path(args.run) / "eval")
    elif args.config:
        cfg = Config.load(args.config)
    else:
        if not (args.ft and args.badcases):
            raise RBAError(
                "验收还缺模型和坏例。",
                f"推荐：rba {what} --run 分析目录 ...。或者：rba {what} --ft 微调目录 --badcases 坏例.jsonl --goodcases 对照.jsonl ...",
            )
        cfg = Config()
        cfg.goodcases = None
        if not args.base:
            args.base = args.ft
    cfg = _apply_cli(cfg, args)
    if getattr(args, "dtype", None):
        cfg.dtype = args.dtype
    cfg.dry_run = True
    cfg.fresh = False
    return cfg, run_meta


def _int_list(value: str | None, flag: str) -> list[int] | None:
    if not value:
        return None
    try:
        return [int(v) for v in value.split(",") if v.strip()]
    except ValueError as exc:
        raise RBAError(f"{flag} 要写逗号分隔的整数，收到 {value!r}。", f"例如 {flag} 3,5") from exc


def _finetune_options(args):
    from rba.repair.finetune import FinetuneOptions

    report = None
    if args.run:
        path = Path(args.run) / "report.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
        else:
            raise RBAError(f"{args.run} 里没有 report.json，拿不到定位结果。", "用 --layers 手动指定层，或指向完整的分析目录。")
    for flag, value in (("--replay-ratio", args.replay_ratio), ("--control-holdout", args.control_holdout), ("--train-holdout", args.train_holdout)):
        if not 0.0 <= value < 1.0:
            raise RBAError(f"{flag} 要在 0 到 1 之间（不含 1），收到 {value}。", f"例如 {flag} 0.5")
    if args.steps <= 0:
        raise RBAError("--steps 要大于 0。", "例如 --steps 100")
    return FinetuneOptions(
        out=args.out, mode=args.mode, targets=args.targets, layers=_int_list(args.layers, "--layers"),
        modules=[m.strip() for m in args.modules.split(",") if m.strip()] if args.modules else None,
        steps=args.steps, lr=args.lr, train_batch_size=args.train_batch_size, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        hard_negative_weight=args.hard_negative_weight, margin=args.margin, replay_ratio=args.replay_ratio,
        kl_weight=args.kl_weight, control_holdout=args.control_holdout, holdout=args.train_holdout, extra_train=args.extra_train,
        balance_groups=not args.no_balance_groups, grad_checkpointing=args.grad_checkpointing,
        seed=args.seed or 0, overwrite=args.overwrite, report=report,
    )


def _cmd_demo(args) -> None:
    root = _examples_root()
    cfg = Config.load(root / "example.yaml")
    cfg.dry_run = True
    cfg.fresh = True
    cfg.resume = False
    cfg.output_dir = args.output_dir
    cfg.device = "cpu"
    cfg.dtype = "float32"
    if args.arch == "qwen3_5_tiny":
        cfg.base_model = "qwen3_5_tiny"
        cfg.ft_model = "qwen3_5_tiny"
        cfg.model_type = "qwen3_5_tiny"
        cfg.force_eager_attn = True
    analyze(cfg)


def _examples_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [Path.cwd(), here.parents[2], here.parents[1]]:
        if (candidate / "examples" / "badcases.jsonl").exists() and (candidate / "example.yaml").exists():
            return candidate
    raise RBAError(
        "找不到示例数据和 example.yaml。",
        "在仓库根目录运行 rba demo。示例文件在 examples/badcases.jsonl。",
    )


def _cmd_init(args) -> None:
    base = args.base or "tiny"
    ft = args.ft or "tiny"
    bad = args.badcases or "examples/badcases.jsonl"
    good = args.goodcases or "examples/goodcases.jsonl"
    detected = None
    if ft not in BUILTIN_MODELS:
        path = Path(ft)
        if not path.exists():
            raise RBAError(
                f"找不到微调模型 {ft}。",
                "init 只读 config.json，不会加载权重。路径应是模型目录；试跑用 rba init --base tiny --ft tiny --badcases examples/badcases.jsonl。",
            )
        detected = detect_directory(path)
        info(
            f"识别到 {detected['architecture'] or detected['model_type']}，"
            f"{detected['num_hidden_layers']} 层，hidden {detected['hidden_size']}，"
            f"图像 token id {detected['image_token_id']}。"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if device == "cuda" else "float32"
    builtin = ft in BUILTIN_MODELS
    layer_note = ""
    if detected and detected.get("layer_types"):
        kinds = detected["layer_types"]
        n_full = sum(1 for kind in kinds if kind == "full_attention")
        n_linear = sum(1 for kind in kinds if kind == "linear_attention")
        layer_note = f"# 层类型：full_attention {n_full}，linear_attention {n_linear}。线性层没有注意力矩阵。\n"
    text = f"""# rba 配置。命令行参数会盖掉这里的同名字段。
# 下一步：rba analyze --config {args.out}

base_model: {base}
ft_model: {ft}
model_type: {"tiny" if ft == "tiny" else "qwen3_5_tiny" if ft == "qwen3_5_tiny" else "auto"}
badcases: {bad}
goodcases: {good}

grades: ["-1", "0", "1", "2", "3"]
# 等级 token 协议。single_token：每个等级本身就是一个 token。
# 若 "-1" 被切成 "-" 和 "1"，并且训练时确实是按首 token 分类，才写 first_token（报告会标明）。
grade_mode: single_token
# 若要手写 token id，键用字符串，并且必须等于分词器对该等级的首 token：
# grade_token_ids: {{"-1": 12, "0": 15, "1": 16, "2": 17, "3": 18}}

# 解码协议。constrained：只在等级 token 里取 argmax；free：全词表 argmax，等级外算非法输出并单独统计。
decode_protocol: constrained

# 图片。真实模型用它自己的处理器读原图；{{image}} 在模板里会换成 image_placeholder。
image_token: "<|image_pad|>"
image_placeholder: "<|vision_start|><|image_pad|><|vision_end|>"
image_max_pixels: 262144   # 处理器允许的最大像素数，越小图像 token 越少、越省显存
missing_image: error       # error 停下；exclude 剔除该样本并写进报告
image_seq_len: 4           # 只对内置小模型有意义
input_mode: template       # template 按 prompt_template 拼；chat 用处理器的对话模板（含纯文本样本）

device: {device}
dtype: {dtype}
batch_size: 0              # 0 自动；带处理器视觉输入时固定为 1
max_length: 2048
seed: 0
plant_error: {"true" if builtin else "false"}

# 验收集。按 id 哈希留出这一比例的坏例不参与选补丁，只给 rba eval 用。0 表示不留（eval 只能做发现集复评）。
holdout_fraction: {"0.0" if builtin else "0.5"}

# 模块路径。留空（注释掉）表示加载后按模块树自动发现；只有自动发现失败时才手写。
{layer_note}# module_paths:
#   layers: model.language_model.layers
#   lm_head: lm_head
#   visual: model.visual
#   final_norm: model.language_model.norm

token_attr_method: gradxinput
ig_steps: 4
verify_top_k: 3
max_detail_cases: 4
force_eager_attn: false

llm:
  base_url: https://api.openai.com/v1
  model: gpt-4.1
  api_key_env: OPENAI_API_KEY
  max_prompt_tokens: 24000
  temperature: 0.2
  enabled: true

analyses:
  logit_lens: true
  dla: true
  token_attr: true
  delta_w: true
  probes: true
  cluster: true
  verify: true
  llm: true

prompt_template: |
{_indent(DEFAULT_TEMPLATE)}
"""
    out = Path(args.out)
    if out.exists():
        raise RBAError(
            f"{out} 已经存在，没有覆盖。",
            "换一个 --out，或先把旧文件挪走。init 不会悄悄改掉你改过的配置。",
        )
    out.write_text(text, encoding="utf-8")
    Config.load(out)
    info(f"已写入 {out.resolve()}。设备选了 {device} / {dtype}。")
    info(f"试跑：rba analyze --config {out} --dry-run")
    if builtin:
        info("当前基座和微调都是内置小模型。换成真实目录后再跑同一条命令。")


def _indent(text: str) -> str:
    return "\n".join("  " + line if line else "" for line in text.splitlines())
