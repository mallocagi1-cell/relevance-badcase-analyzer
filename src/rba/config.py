"""YAML and CLI configuration. Unknown keys fail early with a hint."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from rba.errors import RBAError

DEFAULT_GRADES = ["-1", "0", "1", "2", "3"]

DEFAULT_TEMPLATE = (
    "判断文档与查询的相关性。只输出一个等级，不要输出其它文字。\n"
    "等级含义：-1 冲突或答非所问，0 不相关，1 弱相关，2 相关，3 高度相关。\n"
    "查询：{query}\n"
    "文档：{doc}\n"
    "{image}相关性等级："
)


def _reject_unknown(data: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise RBAError(
            f"{where}出现了无法识别的字段：{', '.join(unknown)}。",
            f"可用字段：{', '.join(sorted(allowed))}。多半是拼写错误。",
        )


@dataclass
class LLMConfig:
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1"
    api_key_env: str = "OPENAI_API_KEY"
    max_prompt_tokens: int = 24000
    temperature: float = 0.2
    timeout_s: float = 180.0
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict | None) -> "LLMConfig":
        data = data or {}
        _reject_unknown(data, {f.name for f in fields(cls)}, "llm 配置里")
        return cls(**data)


@dataclass
class AnalysesConfig:
    logit_lens: bool = True
    dla: bool = True
    token_attr: bool = True
    delta_w: bool = True
    probes: bool = True
    cluster: bool = True
    verify: bool = True
    llm: bool = True

    @classmethod
    def from_dict(cls, data: dict | None) -> "AnalysesConfig":
        data = data or {}
        _reject_unknown(data, {f.name for f in fields(cls)}, "analyses 配置里")
        return cls(**data)


@dataclass
class TinySpec:
    hidden_size: int = 64
    num_layers: int = 4
    num_heads: int = 4
    intermediate_size: int = 128
    vocab_size: int = 128
    image_seq_len: int = 4
    image_size: int = 8
    layer_types: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "TinySpec":
        data = dict(data or {})
        _reject_unknown(data, {f.name for f in fields(cls)}, "tiny 配置里")
        return cls(**data)


@dataclass
class Config:
    base_model: str = "tiny"
    ft_model: str = "tiny"
    model_type: str = "auto"
    model_class: str | None = None
    badcases: str = "examples/badcases.jsonl"
    goodcases: str | None = "examples/goodcases.jsonl"
    grades: list[str] = field(default_factory=lambda: list(DEFAULT_GRADES))
    grade_token_ids: dict[str, int] | None = None
    prompt_template: str = DEFAULT_TEMPLATE
    image_token: str = "<|image_pad|>"
    image_token_id: int | None = None
    image_seq_len: int = 4
    image_placeholder: str = "<|vision_start|><|image_pad|><|vision_end|>"
    image_max_pixels: int | None = 262144
    missing_image: str = "error"
    input_mode: str = "template"
    grade_mode: str = "single_token"
    decode_protocol: str = "constrained"
    holdout_fraction: float = 0.0
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 0
    max_length: int = 2048
    output_dir: str | None = None
    seed: int = 0
    plant_error: bool = True
    max_cases: int | None = None
    max_detail_cases: int = 4
    cluster_k: int = 3
    probe_attributes: list[str] | None = None
    probe_max_cases: int = 64
    token_attr_method: str = "gradxinput"
    ig_steps: int = 4
    verify_top_k: int = 3
    delta_w_top_k: int = 12
    svd_rank: int = 4
    svd_max_dim: int = 4096
    module_paths: dict[str, str | None] = field(default_factory=dict)
    llm: LLMConfig = field(default_factory=LLMConfig)
    analyses: AnalysesConfig = field(default_factory=AnalysesConfig)
    tiny: TinySpec = field(default_factory=TinySpec)
    resume: bool = True
    fresh: bool = False
    dry_run: bool = False
    chars_per_token: float = 1.6
    language: str = "zh"
    max_llm_groups: int = 8
    prompt_max_components: int = 16
    verify_controls: bool = True
    force_eager_attn: bool = False
    config_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict, config_path: str | None = None) -> "Config":
        data = dict(data)
        allowed = {f.name for f in fields(cls)} - {"config_path"}
        _reject_unknown(data, allowed, "配置文件")
        llm = LLMConfig.from_dict(data.pop("llm", None))
        analyses = AnalysesConfig.from_dict(data.pop("analyses", None))
        tiny = TinySpec.from_dict(data.pop("tiny", None))
        module_paths = data.pop("module_paths", None) or {}
        if not isinstance(module_paths, dict):
            raise RBAError(
                "module_paths 必须是若干路径组成的映射。",
                "例如 module_paths: {layers: model.language_model.layers, lm_head: lm_head}。留空则自动发现。",
            )
        cfg = cls(**data)
        if cfg.grade_token_ids is not None:
            if not isinstance(cfg.grade_token_ids, dict):
                raise RBAError(
                    "grade_token_ids 必须是「等级: token id」的映射。",
                    '例如 grade_token_ids: {"-1": 12, "0": 15, "1": 16, "2": 17, "3": 18}。',
                )
            normalized: dict[str, int] = {}
            for key, value in cfg.grade_token_ids.items():
                text = str(key)
                if text in normalized:
                    raise RBAError(
                        f"grade_token_ids 里 {key!r} 和 {text!r} 规范化后重复。",
                        "YAML 里把键写成字符串，例如 \"-1\": 12，每个等级只出现一次。",
                    )
                normalized[text] = value
            cfg.grade_token_ids = normalized
        cfg.llm = llm
        cfg.analyses = analyses
        cfg.tiny = tiny
        cfg.module_paths = {str(k): (None if v in (None, "", "auto") else str(v)) for k, v in module_paths.items()}
        cfg.config_path = config_path
        cfg.validate_static()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise RBAError(
                f"找不到配置文件 {path}。",
                "先运行 rba init 生成 rba.yaml，或直接 rba analyze --base ... --ft ... --badcases ...。",
            )
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise RBAError(
                f"配置文件不是合法 YAML：{exc}",
                "检查缩进。冒号后面要有空格，多行模板用 | 接在 prompt_template: 下面。",
            ) from exc
        if not isinstance(data, dict):
            raise RBAError("配置文件的顶层必须是键值对。", "参考仓库根目录的 example.yaml。")
        return cls.from_dict(data, config_path=str(path))

    def validate_static(self) -> None:
        if self.language not in {"zh", "en"}:
            raise RBAError(f"language={self.language} 无法识别。", "使用 zh（默认，中文结论）或 en（大模型用英文回答）。")
        if len(self.grades) < 2:
            raise RBAError("至少需要两个等级。", "默认是 -1,0,1,2,3。在 grades 里列出微调时使用的等级字符串。")
        if len(set(self.grades)) != len(self.grades):
            raise RBAError("grades 里有重复等级。", "每个等级字符串只能出现一次。")
        self.grades = [str(g) for g in self.grades]
        if "{query}" not in self.prompt_template or "{doc}" not in self.prompt_template:
            raise RBAError(
                "prompt_template 必须包含 {query} 和 {doc}。",
                "模板要和微调时的输入一致，并且以“下一个 token 就是等级”的位置结束，不要把标准答案写进模板。",
            )
        if self.input_mode not in {"template", "chat", "auto"}:
            raise RBAError(
                f"input_mode={self.input_mode} 无法识别。",
                "使用 template（按 prompt_template 拼文本）、chat（处理器的对话模板）或 auto。",
            )
        if self.token_attr_method not in {"gradxinput", "integrated_gradients"}:
            raise RBAError(
                f"token_attr_method={self.token_attr_method} 无法识别。",
                "可选 gradxinput 或 integrated_gradients。",
            )
        if self.ig_steps < 1:
            raise RBAError("ig_steps 至少为 1。", "积分梯度常用 4 到 16 步。显存紧就用 gradxinput。")
        if self.max_prompt_tokens_value() < 500:
            raise RBAError(
                "llm.max_prompt_tokens 太小，提示词放不进任务说明。",
                "设到几千以上。默认 24000。",
            )
        if self.model_type not in {"auto", "tiny", "qwen3_5_tiny", "hf"}:
            raise RBAError(
                f"model_type={self.model_type} 无法识别。",
                "可选 auto、tiny、qwen3_5_tiny、hf。",
            )
        if self.missing_image not in {"error", "exclude"}:
            raise RBAError(
                f"missing_image={self.missing_image} 无法识别。",
                "error：有图片读不到就停下；exclude：剔除这条样本并写进报告。不会悄悄退化成纯文本。",
            )
        if self.grade_mode not in {"single_token", "first_token"}:
            raise RBAError(
                f"grade_mode={self.grade_mode} 无法识别。",
                "single_token：每个等级就是一个 token；first_token：只在训练协议明确用首 token 分类时声明。",
            )
        if self.decode_protocol not in {"constrained", "free"}:
            raise RBAError(
                f"decode_protocol={self.decode_protocol} 无法识别。",
                "constrained：只在等级 token 里取 argmax；free：全词表 argmax，落在等级外算非法输出。",
            )
        if not (0.0 <= float(self.holdout_fraction) < 1.0):
            raise RBAError("holdout_fraction 必须在 [0, 1) 之间。", "0 表示不留验收集；0.5 表示按 id 哈希留一半给 rba eval。")
        if self.image_max_pixels is not None and int(self.image_max_pixels) < 1024:
            raise RBAError("image_max_pixels 太小。", "至少 1024；留空表示用处理器默认值。")
        if "{image}" in self.prompt_template and not self.image_placeholder:
            raise RBAError("image_placeholder 不能为空。", "真实模型用处理器展开它，默认 <|vision_start|><|image_pad|><|vision_end|>。")

    def max_prompt_tokens_value(self) -> int:
        return int(self.llm.max_prompt_tokens)

    def resolve_path(self, value: str | None, from_cli: bool = False) -> str | None:
        if value is None:
            return None
        if value in {"tiny", "qwen3_5_tiny"}:
            return value
        path = Path(value)
        if path.is_absolute():
            return str(path)
        if from_cli and path.exists():
            return str(path.resolve())
        if self.config_path:
            beside = Path(self.config_path).resolve().parent / path
            if beside.exists():
                return str(beside.resolve())
        if path.exists():
            return str(path.resolve())
        if self.config_path:
            return str((Path(self.config_path).resolve().parent / path).resolve())
        return str(path.resolve())

    def fingerprint_payload(self) -> dict:
        return {
            "base": self.base_model,
            "ft": self.ft_model,
            "model_type": self.model_type,
            "model_class": self.model_class,
            "grades": self.grades,
            "grade_token_ids": self.grade_token_ids,
            "prompt_template": self.prompt_template,
            "image_token": self.image_token,
            "image_token_id": self.image_token_id,
            "image_seq_len": self.image_seq_len,
            "image_placeholder": self.image_placeholder,
            "image_max_pixels": self.image_max_pixels,
            "missing_image": self.missing_image,
            "input_mode": self.input_mode,
            "grade_mode": self.grade_mode,
            "decode_protocol": self.decode_protocol,
            "holdout_fraction": self.holdout_fraction,
            "dtype": self.dtype,
            "max_length": self.max_length,
            "seed": self.seed,
            "plant_error": self.plant_error,
            "max_cases": self.max_cases,
            "max_detail_cases": self.max_detail_cases,
            "cluster_k": self.cluster_k,
            "probe_attributes": self.probe_attributes,
            "probe_max_cases": self.probe_max_cases,
            "token_attr_method": self.token_attr_method,
            "ig_steps": self.ig_steps,
            "verify_top_k": self.verify_top_k,
            "delta_w_top_k": self.delta_w_top_k,
            "svd_rank": self.svd_rank,
            "module_paths": self.module_paths,
            "analyses": self.analyses.__dict__,
            "tiny": self.tiny.__dict__,
            "force_eager_attn": self.force_eager_attn,
            "verify_controls": self.verify_controls,
            "max_llm_groups": self.max_llm_groups,
            "svd_max_dim": self.svd_max_dim,
            "prompt_max_components": self.prompt_max_components,
        }


CONFIG_SCHEMA_VERSION = 2


def config_to_dict(cfg: Config) -> dict:
    """Every field, so a run can be replayed exactly by `rba eval --run`."""
    from dataclasses import asdict

    payload = asdict(cfg)
    payload.pop("config_path", None)
    payload["schema_version"] = CONFIG_SCHEMA_VERSION
    return payload


def config_from_used(payload: dict) -> Config:
    """Inverse of config_to_dict. Unknown or future schema versions fail loudly."""
    data = dict(payload)
    version = data.pop("schema_version", 1)
    if version != CONFIG_SCHEMA_VERSION:
        raise RBAError(
            f"config.used.json 的 schema_version={version}，当前程序是 {CONFIG_SCHEMA_VERSION}。",
            "用当前版本重新跑一次 rba analyze，再做 eval；旧目录的补丁不保证可比。",
        )
    data.pop("runtime", None)
    allowed = {f.name for f in fields(Config)} - {"config_path"}
    unknown = sorted(set(data) - allowed)
    for key in unknown:
        data.pop(key)
    return Config.from_dict(data)


def apply_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(cfg, key):
            raise RBAError(f"内部错误：未知覆盖项 {key}。", "这是程序问题，请保留命令行并反馈。")
        setattr(cfg, key, value)
    cfg.validate_static()
    return cfg
