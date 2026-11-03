"""Chinese CLI. `rba analyze --base --ft --badcases` needs no config file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from rba import __version__
from rba.config import DEFAULT_TEMPLATE, Config, apply_overrides
from rba.console import error, info
from rba.errors import RBAError
from rba.models.loader import detect_directory
from rba.models.paths import PRESET_PATHS
from rba.pipeline import analyze, evaluate_patch

EPILOG = """
示例：
  rba demo
  python -m rba analyze --config example.yaml --dry-run
  rba analyze --base /path/base --ft /path/ft --badcases bad.jsonl --goodcases good.jsonl
  rba init --base /path/base --ft /path/ft --badcases bad.jsonl
  rba eval --run runs/example --patch runs/example/patches/某组_h1.json
""".strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rba",
        description="分析微调相关性模型的坏例，定位最小修改，并在对照集上验收。",
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

    eval_p = sub.add_parser("eval", help="在坏例和对照集上验收一个补丁")
    _add_common(eval_p)
    eval_p.add_argument("--patch", help="分析报告写出的补丁 JSON")
    eval_p.add_argument("--run", help="某次分析的输出目录，用来补齐模型和数据路径")

    demo = sub.add_parser("demo", help="用内置小模型和样例数据干跑一遍")
    demo.add_argument("--arch", choices=["tiny", "qwen3_5_tiny"], default="tiny", help="tiny 是同结构的替身；qwen3_5_tiny 是真实类的随机小模型")
    demo.add_argument("--output-dir", default="runs/demo")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "init":
            _cmd_init(args)
        elif args.cmd == "analyze":
            cfg = _cfg_from_analyze_args(args)
            analyze(cfg)
        elif args.cmd == "eval":
            cfg, patch = _cfg_from_eval_args(args)
            result = evaluate_patch(cfg, patch)
            out = Path(cfg.output_dir or "runs/eval")
            out.mkdir(parents=True, exist_ok=True)
            path = out / "eval.json"
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            info(f"验收结果：{path}")
            info(result["conclusion"])
        elif args.cmd == "demo":
            _cmd_demo(args)
        else:
            parser.error(f"未知命令 {args.cmd}")
    except RBAError as exc:
        error(exc.message, exc.hint)
        sys.exit(2)
    except KeyboardInterrupt:
        error("已中断。", "已完成的阶段留在输出目录里。用同一个命令再跑会从缓存继续；想重算就加 --fresh。")
        sys.exit(130)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="YAML 配置。命令行里的同名字段会盖掉文件")
    parser.add_argument("--base", help="基座模型目录，或 tiny / qwen3_5_tiny")
    parser.add_argument("--ft", help="微调模型目录，或 tiny / qwen3_5_tiny")
    parser.add_argument("--badcases", help="坏例 jsonl 或 csv")
    parser.add_argument("--goodcases", help="对照集。不传则只报告坏例，看不出回退")
    parser.add_argument("--output-dir", help="报告目录。不传则按数据指纹放进 runs/")
    parser.add_argument("--device", help="auto、cpu 或 cuda")
    parser.add_argument("--batch-size", type=int, help="0 表示自动，显存不够会减半")
    parser.add_argument("--seed", type=int, help="内置小模型的随机种子")


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
    overrides = {
        "base_model": args.base,
        "ft_model": args.ft,
        "badcases": args.badcases,
        "goodcases": args.goodcases,
        "output_dir": args.output_dir,
        "device": args.device,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    cfg = apply_overrides(cfg, {key: value for key, value in overrides.items() if value is not None})
    if args.dry_run:
        cfg.dry_run = True
    if args.fresh:
        cfg.fresh = True
        cfg.resume = False
    if getattr(args, "max_cases", None):
        cfg.max_cases = args.max_cases
    _resolve_cli_paths(cfg, args)
    return cfg


def _resolve_cli_paths(cfg: Config, args) -> None:
    if args.badcases:
        cfg.badcases = str(Path(args.badcases).resolve())
    if args.goodcases:
        cfg.goodcases = str(Path(args.goodcases).resolve())
    if args.base and args.base not in {"tiny", "qwen3_5_tiny"}:
        cfg.base_model = str(Path(args.base).resolve())
    if args.ft and args.ft not in {"tiny", "qwen3_5_tiny"}:
        cfg.ft_model = str(Path(args.ft).resolve())


def _cfg_from_eval_args(args):
    if not args.patch and not args.run:
        raise RBAError(
            "验收需要一个补丁。",
            "用分析报告 patches/ 里的 JSON：rba eval --run runs/某次 --patch runs/某次/patches/组_h1.json",
        )
    base_cfg = None
    if args.run:
        used = Path(args.run) / "config.used.json"
        if not used.exists():
            raise RBAError(
                f"{args.run} 里没有 config.used.json。",
                "把 --run 指到某次 analyze 的输出目录，或改用 --config / --ft / --badcases。",
            )
        base_cfg = Config.from_dict(json.loads(used.read_text(encoding="utf-8")))
        if args.output_dir is None:
            args.output_dir = str(Path(args.run) / "eval")
    if args.config:
        base_cfg = Config.load(args.config)
    if base_cfg is None:
        if not (args.ft and args.badcases):
            raise RBAError(
                "验收还缺模型和坏例。",
                "rba eval --ft 微调目录 --badcases 坏例.jsonl --goodcases 对照.jsonl --patch 补丁.json",
            )
        base_cfg = Config()
        base_cfg.goodcases = None
    cfg = _cfg_from_analyze_args(
        argparse.Namespace(
            config=None,
            base=args.base or base_cfg.base_model,
            ft=args.ft or base_cfg.ft_model,
            badcases=args.badcases or base_cfg.badcases,
            goodcases=args.goodcases if args.goodcases is not None else base_cfg.goodcases,
            output_dir=args.output_dir,
            device=args.device or base_cfg.device,
            batch_size=args.batch_size,
            seed=args.seed if args.seed is not None else base_cfg.seed,
            dry_run=True,
            fresh=False,
            max_cases=None,
        )
    )
    # Keep template and grades from the run that produced the patch.
    cfg.prompt_template = base_cfg.prompt_template
    cfg.grades = list(base_cfg.grades)
    cfg.image_token = base_cfg.image_token
    cfg.image_seq_len = base_cfg.image_seq_len
    cfg.model_type = base_cfg.model_type
    cfg.plant_error = base_cfg.plant_error
    cfg.tiny = base_cfg.tiny
    cfg.dry_run = True
    patch_path = args.patch
    if patch_path is None:
        raise RBAError("没有 --patch。", "指向 patches 目录里的某一个 JSON。")
    path = Path(patch_path)
    if not path.exists():
        raise RBAError(f"找不到补丁 {path}。", "路径写报告里的那一行，不要手改字段名。")
    try:
        patch = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RBAError(f"补丁不是合法 JSON：{exc}", "重新从报告生成，或检查有没有被编辑坏。") from exc
    if not isinstance(patch, dict) or "method" not in patch:
        raise RBAError("补丁里没有 method。", "用 head_scale、mlp_zero、steering、lm_head_blend 或 ablate_tokens。")
    return cfg, patch


def _cmd_demo(args) -> None:
    root = _examples_root()
    cfg = Config.load(root / "example.yaml") if (root / "example.yaml").exists() else Config.load(Path("example.yaml"))
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
    preset_name = "qwen3_5"
    if ft not in {"tiny", "qwen3_5_tiny"}:
        path = Path(ft)
        if not path.exists():
            raise RBAError(
                f"找不到微调模型 {ft}。",
                "init 只读 config.json，不会加载权重。路径应是模型目录；试跑用 rba init --base tiny --ft tiny --badcases examples/badcases.jsonl。",
            )
        detected = detect_directory(path)
        preset_name = detected["preset"]
        info(
            f"识别到 {detected['architecture'] or detected['model_type']}，"
            f"{detected['num_hidden_layers']} 层，hidden {detected['hidden_size']}，"
            f"图像 token id {detected['image_token_id']}。"
        )
        if detected["architecture"] and "VL" in str(detected["architecture"]) and "Qwen3_5VL" == detected["architecture"]:
            info("类名若加载失败，程序会改试 Qwen3_5ForConditionalGeneration。transformers 5.22 里没有 Qwen3_5VL 这个类。")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if device == "cuda" else "float32"
    paths = PRESET_PATHS[preset_name]
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
# 若某个等级被切成多个子词，在这里手写单个 token id，例如
# grade_token_ids: {{"-1": 123, "0": 15, "1": 16, "2": 17, "3": 18}}

image_token: "<|image_pad|>"
image_seq_len: 4
input_mode: template

device: {device}
dtype: {dtype}
batch_size: 0
max_length: 2048
seed: 0
plant_error: {"true" if ft in {"tiny", "qwen3_5_tiny"} else "false"}

# 留空表示加载后按模块树自动发现。下面是根据 config.json 猜的，加载时还会再核对。
{layer_note}module_paths:
  layers: {paths["layers"]}
  lm_head: {paths["lm_head"]}
  visual: {paths["visual"]}
  final_norm: {paths["final_norm"]}

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
    # Make sure the file we just wrote still parses.
    Config.load(out)
    info(f"已写入 {out.resolve()}。设备选了 {device} / {dtype}。")
    info(f"试跑：rba analyze --config {out} --dry-run")
    if ft in {"tiny", "qwen3_5_tiny"}:
        info("当前基座和微调都是内置小模型。换成真实目录后再跑同一条命令。")


def _indent(text: str) -> str:
    return "\n".join("  " + line if line else "" for line in text.splitlines())
