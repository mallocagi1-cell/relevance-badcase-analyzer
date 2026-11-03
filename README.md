# rba：相关性模型坏例分析

微调过的 Qwen3.5-VL 拿「查询 + 文档（文字或图片）」解码**一个 token**，这个 token 是相关性等级（默认 `-1 0 1 2 3`）。本工具在出错的样本上抽出压缩过的权重和激活证据，做带对照的干预复核，把带编号的证据包交给一个 OpenAI 兼容接口提修改建议，再在验收集和对照集上评估补丁。做法对着 [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)：先定位组件，只改该改的那一点，并看有没有伤及原来判对的样本。

它交付的是「输入正确、评分一致、数值可核验、结论不过度声称」的证据工具。它不会自动证明根因；报告里每个状态的含义见[状态](#报告里的状态)。

## 三分钟看完流程

在仓库根目录：

```bash
pip install -e .
rba demo
python -m rba analyze --config example.yaml --dry-run
```

两条命令都用 CPU 上的内置小模型，不下载权重，不调用大模型。报告在 `runs/demo/report.html` 和 `runs/example/report.html`，`report.md` 是同一份内容。打开先看「先看这里」。

内置小模型被人为挪过 `lm_head` 的等级 `0` 和第 0 层第 0 个头。因为它是随机初始化的，几乎不把概率放在等级词上，所以每组第一条都是「先处理这个：下一个 token 的概率大多不在等级词上」。这是检查在正常工作，不是植入的错误；植入的头在它后面，复核表里能看到目标干预让错误 logit 大幅下降，而随机对照和 identity 对照接近 0。报告顶部有同样的说明。

## 零配置分析真实模型

```bash
rba analyze \
  --base /path/to/base \
  --ft /path/to/Qwen3.5-VL-2B-ft \
  --badcases bad.jsonl \
  --goodcases good.jsonl
```

程序先检查两个目录都有 `config.json` 和权重，读 `config.json` 认架构，加载微调模型的分词器和处理器，核对每个等级是不是单个 token，再加载权重。基座不进 GPU，ΔW 按张量从磁盘读。

想先生成一份带注释的配置：

```bash
rba init --base /path/to/base --ft /path/to/ft --badcases bad.jsonl --goodcases good.jsonl
rba analyze --config rba.yaml --dry-run
```

`--dry-run` 把提示词和证据包写到输出目录的 `prompts/`，不访问网络。去掉它才会用 `llm.api_key_env`（默认 `OPENAI_API_KEY`）去请求 `llm.base_url`。密钥没设置时，程序照常完成分析，只是等同于 `--dry-run`，并在报告里写一条注意。接口调用失败也不会丢报告：失败的组记在 `report.json` 的 `llm_failures`，进程退出码 3，修好后重跑同一条命令会从缓存接着用。

同一条命令再跑会接着用缓存。换了数据或配置会拒绝混进旧目录，这时加 `--fresh`，或换一个 `--output-dir`。

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--decode-protocol constrained\|free` | `constrained` 只在等级 token 里取 argmax；`free` 用全词表 argmax，落在等级外算非法输出，单独统计 |
| `--holdout 0.5` | 按 id 哈希留出一半坏例不参与选补丁，只给 `rba eval` 用 |
| `--allow-missing-images` | 图片读不到时剔除该样本并写进报告，而不是停下 |
| `--max-cases N` | 只取前 N 条坏例试跑 |
| `--fresh` | 忽略并覆盖输出目录里的缓存 |

退出码：0 成功；2 配置或数据错误（已给出建议）；3 分析完成但大模型调用失败；1 程序内部错误，完整 traceback 在输出目录的 `error.log`。

## 验收补丁

报告里每条推荐假设带一条命令：

```bash
rba eval --run runs/example --patch runs/example/patches/3_to_0_h2.json
```

`--run` 完整恢复那次分析的配置（模板、等级 token、解码协议、模块路径、dtype、设备语义），不会悄悄换成别的实验。`eval` 报告两个条件率：坏例修复率和对照回退率，各带 95% Wilson 区间，再加净正确数变化和新增非法输出数。两个条件率来自不同人群，不能直接相减当作线上净收益；净收益取决于线上错误占比。

只有设了 `holdout_fraction`（或 `--holdout`）时，坏例侧才是没有参与选补丁的**验收集**；否则结果写成「发现集复评」，不代表泛化。配置里默认是 0（示例数据只有 8 条），`rba init` 给真实模型写的是 0.5。

补丁只在本次进程里生效，不会把 2B 权重另存一份。当 `lm_head` 和输入 embedding 共享权重时，`lm_head_blend` 这类行编辑会同时改掉该等级 token 的输入表示：报告会标出来，`eval` 本轮拒绝验收这种补丁。

## 数据

`jsonl` 每行一个对象，或带表头的 `csv`。必填 `query`、`label`，以及 `doc` / `image` 至少一个。`prediction` 可空，程序会重跑。`attributes` 是给探针和分组回退统计用的对象；CSV 用 `attr_` 前缀。

```json
{"id": "b001", "query": "红烧肉怎么做", "doc": "五花肉焯水后炖四十分钟。", "image": "images/sample.ppm", "label": 3, "prediction": 0, "attributes": {"domain": "food"}}
```

完整例子在 `examples/badcases.jsonl`、`examples/goodcases.jsonl`、`examples/badcases.csv`。图片路径相对样本文件所在目录。读不到的图片不会被悄悄当成纯文本：默认停下并指出是哪条；加 `--allow-missing-images`（或配置 `missing_image: exclude`）会剔除这条样本并写进报告的「剔除的样本」。

`label` 必须落在 `grades` 里。默认五个字符串：`-1`、`0`、`1`、`2`、`3`。

## 等级 token 协议

`grade_mode: single_token`（默认）要求每个等级本身就是分词器里的一个 token。若某个等级被切成多个子词（`-1` 经常是 `-` 和 `1`），程序在加载权重前报错，并把实际切分写出来。

只有训练协议确实是「按首 token 分类」时，才声明 `grade_mode: first_token`：程序要求各等级的首 token 两两不同，报告和证据包都会标明这是首 token 分类器，不是完整序列的似然。`grade_token_ids` 可以手写，但键必须是字符串、值必须等于分词器对该等级的首 token id；它不是绕过多 token 错误的捷径。

## 解码协议

线上怎么解码，这里就怎么评。`decode_protocol: constrained` 只在等级 token 的 logit 里取 argmax；`free` 用全词表 argmax，等级外的输出记为非法，单独列在报告的「非法输出」里，不进入按等级索引的归因。两种协议都会报告全词表 top 是等级词的比例和等级词概率质量。`eval` 还会报告补丁有没有新增非法输出。

## 模板和图片

`prompt_template` 必须和微调时的输入一致，并且停在「下一个 token 就是等级」的位置，不要把答案写进去。占位符只有 `{query}`、`{doc}`、`{image}`。

真实模型的输入全部走它自己的处理器，有图和无图样本用同一套协议：

* `input_mode: template`：`{image}` 换成 `image_placeholder`（默认 `<|vision_start|><|image_pad|><|vision_end|>`），处理器按原图分辨率决定图像 token 数并打包视觉张量。模板里有图片样本却没有 `{image}` 会报错。
* `input_mode: chat`：所有样本（包括纯文本）走处理器的对话模板。
* `input_mode: auto`：处理器带 chat_template 就用 chat，否则 template。

图片不会被预先缩放。`image_max_pixels`（默认 262144）传给处理器，用来控制图像 token 数和显存。`max_length` 截断只截开头的文本 token，碰到图像块会报错而不是切一半。带处理器视觉张量的样本这一轮逐条前向（batch 1），程序会说明。

内置小模型（`tiny`）自己把 8×8 的图片散到 `image_token` × `image_seq_len` 个位置上，用来测文本/图像归因；`qwen3_5_tiny` 是真实类的随机小模型，没有处理器，图片不进入模型，只保留占位 token。

## 报告里有什么

`report.html` 和 `report.md` 内容相同（HTML 由同一份 Markdown 渲染）。开头是按组排好的假设，每条有状态、证据、证据编号、会推翻它的观察和下一步命令。然后是概览（含协议、非法输出和剔除样本）、图、证据（归因表、读出分解可靠性、token 归因、带对照的干预复核表）、ΔW、探针、给大模型的提示词和回复，最后是方法说明和状态表。`report.json` 是同一份结构，还包含逐例的复核差值。

### 报告里的状态

| 状态 | 含义 |
| --- | --- |
| 消融有效 | 在该组抽样上，决策位置的目标干预让 logit(原预测)−logit(金标) 下降，且多数样本同向。只说明干预有效，不是根因证明 |
| 仅相关 | 归因或 ΔW 指向它，但没做干预，或只做了诊断性干预（lm_head 行混合代数上必然缩小两行差距） |
| 被复核否定 | 干预反而让误差变大。留在列表里，不拿去当根因 |
| 消融无效果 | 干预几乎不改变错误 logit |
| 复核不稳定 | 效果在样本之间方向不一致 |
| 证据不足 | 读出分解对不上、identity 对照非零、随机对照同样有效，或选择样本与复核样本没有分开 |
| 先处理这个 | 等级词上几乎没有概率质量，或全词表 argmax 不是等级词。先核对模板和解码协议 |

没有「置信度」数字：没有校准过的数字不会出现在报告里，只有上面的证据强度。

## 方法（给要核对公式的人）

决策位置是最后一个非填充 token。

Logit lens：每一层残差过模型自己的最终 RMSNorm，再读等级行，同时保留 logit 和条件概率。最后一层应当和真实等级分布一致，这是接线检查。

读出归因：方向 `d = W_U[原预测] − W_U[金标]`。报告里的组件归因是最终 RMSNorm 之后、按真实 norm 增益和最终残差尺度折算的贡献，嵌入入口加各层注意力写入和 MLP 写入应当重建真实 logit 差，重建误差和「可靠」标记一起写出；旧的 norm 前点积只作诊断。头的切分在输出投影上：全注意力是 `o_proj`，GatedDeltaNet 是 `out_proj`。

干预复核在决策位置做（头的输出切片乘系数、MLP 在该位置清零、该位置加 steering 向量），每个目标带同类随机对照和 identity 对照；组里样本够多时复核用没参与选择的样本。`lm_head` 行混合只作诊断。

ΔW 只说明微调动过哪里。奇异向量成对保留并给出解释能量，符号任意；只有输出维就是残差维的矩阵才投到等级词上，没有样本激活系数时不说某个方向「促进」某个等级。共享参数只计一次。

Token 归因是语言层入口上的 grad×input 或积分梯度，按图像 token id 和文本切开；积分梯度写出零基线、端点和完整性误差。这不是像素梯度，也不是图像 token 数量占比。

探针是属性上的岭回归，交错抽取坏例和对照，报告留出一半和多数类基线，不进入根因排序。

## 显存

2B、bf16，单卡大约 5GB 再加激活。默认只把微调模型放到 GPU；基座用 safetensors 逐张量读，SVD 有维度上限。batch 默认很小，遇到 OOM 会减半重试；减到 1 还不够时会停下并按顺序给出可调项（`max_length`、`image_max_pixels`、`token_attr_method: gradxinput`、关掉 `token_attr` 或 `force_eager_attn`）。归因和复核只跑每组最多 `max_detail_cases` 条（默认 4）。CPU 上 dtype 会被改成 float32。

GatedDeltaNet 层在没有 `flash-linear-attention` / `causal-conv1d` 时走 transformers 的 PyTorch 参考实现：结果正确，速度慢。这两个包依赖 CUDA 工具链，不作为默认依赖，见 `pyproject.toml` 的注释。

## Qwen3.5-VL 的模块名

在 transformers 5.22 里核对过：

| 用途 | 路径 |
| --- | --- |
| 类 | `Qwen3_5ForConditionalGeneration` |
| 处理器 | `Qwen3VLProcessor`（`AutoProcessor` 自动选） |
| 语言层 | `model.language_model.layers` |
| 全注意力 | `self_attn`（`q_proj` `k_proj` `v_proj` `o_proj`） |
| 线性注意力 | `linear_attn`（GatedDeltaNet，写入在 `out_proj`） |
| MLP | `mlp.gate_proj` / `up_proj` / `down_proj` |
| 最终 norm | `model.language_model.norm` |
| 视觉塔 | `model.visual` |
| 输出头 | `lm_head` |

这个版本**没有** `Qwen3_5VLForConditionalGeneration`。加载时会先试 `config.json` 里的 `architectures`，再试上表这个类名，然后才是 `AutoModelForImageTextToText`。`module_paths` 留空（`rba init` 默认注释掉）则在模块树上自动发现；手写错了会报中文错误并给出自动发现的候选。

## 内置模型

| `model_type` | 是什么 |
| --- | --- |
| `tiny` | 同名模块的小替身，CPU，支持把 PPM/PNG 散到图像 token |
| `qwen3_5_tiny` | 随机初始化的真实 `Qwen3_5ForConditionalGeneration`（2 层，含一层 GatedDeltaNet），没有处理器 |

## 测试

```bash
pip install -e ".[dev]"
pytest
```

测试不下载权重。`tests/test_processor_path.py` 用真实的 `Qwen3VLProcessor` 加一个合成分词器驱动随机 Qwen3.5，核对命令行编码和直接调用处理器得到的决策 logits 一致、像素确实进入视觉塔。`tests/test_real_qwen.py` 检查分解恒等式和头的可加性。

## 配置里常用的旋钮

`device`、`dtype`、`batch_size`（0 为自动）、`max_length`、`max_cases`、`max_detail_cases`、`verify_top_k`、`token_attr_method`（`gradxinput` 或 `integrated_gradients`）、`ig_steps`、`grade_mode`、`decode_protocol`、`holdout_fraction`、`image_placeholder`、`image_max_pixels`、`missing_image`、`llm.max_prompt_tokens`、`analyses` 下面的开关、`module_paths`、`force_eager_attn`。不认识的字段会直接报错，避免静默拼错。
