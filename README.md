# rba：相关性模型坏例分析

微调过的 Qwen3.5-VL 拿「查询 + 文档（文字或图片）」解码**一个 token**，这个 token 是相关性等级（默认 `-1 0 1 2 3`）。本工具在出错的样本上抽出压缩过的权重和激活证据，排出可复核的根因，再交给一个 OpenAI 兼容接口做最小修改建议。做法对着 [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)：先定位组件，只改该改的那一点，并用对照集看有没有伤及原来判对的样本。

## 三分钟看完流程

在仓库根目录：

```bash
pip install -e .
rba demo
python -m rba analyze --config example.yaml --dry-run
```

`rba demo` 和上面的 dry-run 用的都是 CPU 上的内置小模型，不会下载权重，也不会调用大模型。报告在 `runs/demo/report.html` 和 `runs/example/report.html`。打开先看「先看这里」：每个混淆组一组假设，后面是可以直接复制的 `rba eval` 命令。

小模型被人为挪过 `lm_head` 的等级 `0` 和第 0 层第 0 个头，用来确认工具链。报告顶部有说明。那些数字不是线上模型的结论。

## 零配置分析真实模型

```bash
rba analyze \
  --base /path/to/base \
  --ft /path/to/Qwen3.5-VL-2B-ft \
  --badcases bad.jsonl \
  --goodcases good.jsonl
```

不写配置文件也能跑。程序会读 `config.json` 认架构，检查每个等级是不是单个 token，再加载微调模型。基座不进 GPU，ΔW 按张量从磁盘读。

想先生成一份带注释的配置：

```bash
rba init --base /path/to/base --ft /path/to/ft --badcases bad.jsonl --goodcases good.jsonl
rba analyze --config rba.yaml --dry-run
```

`--dry-run` 把提示词写到输出目录的 `prompts/`，不访问网络。去掉它才会用 `llm.api_key_env`（默认 `OPENAI_API_KEY`）去请求 `llm.base_url`。密钥没设置时，程序在加载模型之前就停，并告诉你怎么改。

同一条命令再跑会接着用缓存。换了数据或配置会拒绝混进旧目录，这时加 `--fresh`，或换一个 `--output-dir`。

验收某条假设：

```bash
rba eval --run runs/example --patch runs/example/patches/3_to_0_h1.json
```

坏例修复率高于对照回退率，才值得留下这个补丁。补丁只在本次进程里生效，不会把 2B 权重另存一份。

## 数据

`jsonl` 每行一个对象，或带表头的 `csv`。必填 `query`、`label`，以及 `doc` / `image` 至少一个。`prediction` 可空，程序会重跑。`attributes` 是给探针用的对象；CSV 用 `attr_` 前缀。

```json
{"id": "b001", "query": "红烧肉怎么做", "doc": "五花肉焯水后炖四十分钟。", "image": "images/sample.ppm", "label": 3, "prediction": 0, "attributes": {"domain": "food"}}
```

完整例子在 `examples/badcases.jsonl`、`examples/goodcases.jsonl`、`examples/badcases.csv`。图片路径相对样本文件所在目录。读不到的图片会跳过并按纯文本继续，不会让整次运行失败。

`label` 必须落在 `grades` 里。默认五个字符串：`-1`、`0`、`1`、`2`、`3`。预测是这五个 logit 的 argmax，不是全词表 argmax。若某个等级被切成多个子词（`-1` 经常是 `-` 和 `1`），程序会在加载权重前报错。这时在配置里写 `grade_token_ids`，把每个等级映射到一个 token id。

## 模板

`prompt_template` 必须和微调时的输入一致，并且停在「下一个 token 就是等级」的位置，不要把答案写进去。占位符只有 `{query}`、`{doc}`、`{image}`。`{image}` 会换成 `image_token` 重复 `image_seq_len` 次；没有图时是空字符串。

真实的 Qwen 视觉塔要吃处理器打包好的 patch。`input_mode: chat` 会走 `AutoProcessor`。对不上训练模板时用默认的 `template`。内置小模型自己把图片散到图像 token 上，用来测文本/图像归因。

## 报告里有什么

`report.html` 和 `report.md` 是同一份报告。开头是按组排好的假设：

| 状态 | 含义 |
| --- | --- |
| 已验证 | 在该组抽样上，这个干预让 logit(预测)−logit(金标) 下降，且多数样本同向 |
| 仅相关 | 归因或 ΔW 指向它，但还没做消融，或消融不稳定 |
| 被复核否定 | 消融反而让误差变大，留在列表里，不拿去当根因 |
| 证据不足 | 残差分解对不上，或等级词上几乎没有概率 |

抽样复核不是对照集。下一步命令会在全部坏例和对照集上重跑。

后面才是混淆矩阵、logit lens、直接 logit 归因、token 归因、ΔW、探针，以及给大模型的提示词。`report.json` 是同一份结构。

## 方法（给要核对公式的人）

决策位置是最后一个非填充 token。

Logit lens：每一层残差过模型自己的最终 RMSNorm，再读五个等级行。最后一层应当和真实等级分布一致，这是接线检查。

直接 logit 归因的方向是 `d = W_U[预测] − W_U[金标]`。残差写入 `c` 的贡献是 `c·d`。语言层入口（视觉特征已经写进去之后）加上每层注意力写入和 MLP 写入，等于进入最终 RMSNorm 的向量。这个和不等于 norm 之后的 logit 差，因为 RMSNorm 不是线性的。两边都记。头的切分在输出投影上：全注意力是 `o_proj`，GatedDeltaNet 是 `out_proj`。没有 bias 时，头的和等于该模块的写入。

ΔW 只说明微调动过哪里。只有输出维就是残差维的矩阵（`o_proj`、`out_proj`、`down_proj`）才把奇异方向投到等级词上。范数大不是根因。`*.norm.weight` 标成全局旋钮。

Token 归因是语言层入口上的 grad×input 或积分梯度，目标同样是错误 logit，按图像 token id 和文本切开。这不是像素梯度。注意力份额只在模块真的返回权重时计算；线性注意力层没有注意力矩阵。要测份额就设 `force_eager_attn: true`（更慢、更占显存）。

探针是用户属性上的岭回归，样本够时留出一半。它不进入根因排序。

提示词的结构和那篇博客的实验记录一致：先写任务和验收方式，再给压缩后的数，要求最小修改，并写明什么观察会推翻它。模型回复和测量排序分开，不当成已经证实的根因。

## 显存

2B、bf16，单卡大约 5GB 再加激活。默认只把微调模型放到 GPU；基座用 safetensors 逐张量读。batch 默认很小，遇到 OOM 会减半重试。归因和复核只跑每组最多 `max_detail_cases` 条（默认 4），并且只覆盖样本最多的若干个混淆组。长序列把 `max_length` 放在决策 token 还在的前提下尽量缩小。CPU 上 dtype 会被改成 float32。

## Qwen3.5-VL 的模块名

在 transformers 5.22 里核对过：

| 用途 | 路径 |
| --- | --- |
| 类 | `Qwen3_5ForConditionalGeneration` |
| 语言层 | `model.language_model.layers` |
| 全注意力 | `self_attn`（`q_proj` `k_proj` `v_proj` `o_proj`） |
| 线性注意力 | `linear_attn`（GatedDeltaNet，写入在 `out_proj`） |
| MLP | `mlp.gate_proj` / `up_proj` / `down_proj` |
| 最终 norm | `model.language_model.norm` |
| 视觉塔 | `model.visual` |
| 输出头 | `lm_head` |

这个版本**没有** `Qwen3_5VLForConditionalGeneration`。加载时会先试 `config.json` 里的 `architectures`，再试上表这个类名，然后才是 `AutoModelForImageTextToText`。`module_paths` 可以手写；写错会报中文错误并给出自动发现的候选。留空则在模块树上找带 MLP 和注意力的 `ModuleList`。Qwen2-VL、Qwen2.5-VL、Qwen3-VL 在同一版本里也是 `model.language_model` 这一套。

层类型是交替的 full attention 和 linear attention。线性层没有注意力矩阵，归因走 `out_proj` 的 value head。

带图片的前向还需要处理器给出的 `mm_token_type_ids`。模板模式不会替你编这个；对得上训练的做法是把 `input_mode` 设成 `chat`，或保证模板里的图像占位个数和视觉 merger 一致。

## 内置模型

| `model_type` | 是什么 |
| --- | --- |
| `tiny` | 同名模块的小替身，CPU，支持把 PPM/PNG 散到图像 token |
| `qwen3_5_tiny` | 随机初始化的真实 `Qwen3_5ForConditionalGeneration`（2 层，含一层 GatedDeltaNet） |

`rba demo --arch qwen3_5_tiny` 走真实类。测试 `tests/test_real_qwen.py` 检查分解恒等式和头的可加性。

## 测试

```bash
pip install -e ".[dev]"
pytest
```

测试不下载权重。`test_real_qwen.py` 在当前环境的 transformers 里实例化一个很小的 Qwen3.5。

## 配置里常用的旋钮

`device`、`dtype`、`batch_size`（0 为自动）、`max_length`、`max_cases`、`max_detail_cases`、`verify_top_k`、`token_attr_method`（`gradxinput` 或 `integrated_gradients`）、`ig_steps`、`llm.max_prompt_tokens`、`analyses` 下面的开关、`module_paths`、`force_eager_attn`。不认识的字段会直接报错，避免静默拼错。
