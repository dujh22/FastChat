# FastChat
| [**演示**](https://chat.lmsys.org/) | [**Discord**](https://discord.gg/HSWAKCrnFx) | [**X**](https://x.com/lmsysorg) |

FastChat 是一个用于训练、部署和评估基于大型语言模型的聊天机器人的开放平台。
- FastChat 为 Chatbot Arena (https://chat.lmsys.org/) 提供支持，为70多个大语言模型提供了超过1000万次的聊天请求服务。
- Chatbot Arena 已收集了超过50万次人工投票，这些投票来自大语言模型的一对一对战，用于编制在线[LLM Elo 排行榜](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)。

FastChat 的核心功能包括：
- 最先进模型（如 Vicuna、MT-Bench）的训练和评估代码。
- 具有网页界面和兼容 OpenAI 的 RESTful API 的分布式多模型服务系统。

## 新闻
- [2024/03] 🔥 我们发布了 Chatbot Arena 技术[报告](https://arxiv.org/abs/2403.04132)。
- [2023/09] 我们发布了 **LMSYS-Chat-1M**，一个大规模真实世界 LLM 对话数据集。阅读[报告](https://arxiv.org/abs/2309.11998)。
- [2023/08] 我们发布了基于 Llama 2 的 **Vicuna v1.5**，具有 4K 和 16K 上下文长度。下载[权重](#vicuna-weights)。
- [2023/07] 我们发布了 **Chatbot Arena Conversations**，一个包含 33k 对话及人类偏好的数据集。在[这里](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)下载。

<details>
<summary>更多</summary>

- [2023/08] 我们发布了基于 Llama 2 的 **LongChat v1.5**，具有 32K 上下文长度。下载[权重](#longchat)。
- [2023/06] 我们推出了 **MT-bench**，一个具有挑战性的多轮问题集，用于评估聊天机器人。查看博客[文章](https://lmsys.org/blog/2023-06-22-leaderboard/)。
- [2023/06] 我们推出了 **LongChat**，我们的长上下文聊天机器人和评估工具。查看博客[文章](https://lmsys.org/blog/2023-06-29-longchat/)。
- [2023/05] 我们推出了 **Chatbot Arena**，用于 LLM 之间的对战。查看博客[文章](https://lmsys.org/blog/2023-05-03-arena)。
- [2023/03] 我们发布了 **Vicuna：一个开源聊天机器人，以 90% ChatGPT 的质量令 GPT-4 印象深刻**。查看博客[文章](https://vicuna.lmsys.org)。

</details>

<a href="https://chat.lmsys.org"><img src="assets/demo_narrow.gif" width="70%"></a>

## 目录
- [安装](#install)
- [模型权重](#model-weights)
- [命令行界面推理](#inference-with-command-line-interface)
- [Web GUI 服务](#serving-with-web-gui)
- [API](#api)
- [评估](#evaluation)
- [微调](#fine-tuning)
- [引用](#citation)

## 安装

### 方法 1：使用 pip

```bash
pip3 install "fschat[model_worker,webui]"
```

### 方法 2：从源码安装

1. 克隆此仓库并进入 FastChat 文件夹
```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
```

如果你在 Mac 上运行：
```bash
brew install rust cmake
```

2. 安装包
```bash
pip3 install --upgrade pip  # 启用 PEP 660 支持
pip3 install -e ".[model_worker,webui]"
```

## 模型权重
### Vicuna 权重
[Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) 基于 Llama 2，应在 Llama 的[模型许可](https://github.com/facebookresearch/llama/blob/main/LICENSE)下使用。

你可以使用以下命令开始聊天。它会自动从 Hugging Face 仓库下载权重。
下载的权重存储在用户主文件夹的 `.cache` 文件夹中（例如，`~/.cache/huggingface/hub/<model_name>`）。

在下面的"命令行界面推理"部分查看更多命令选项和如何处理内存不足的情况。

**注意：16K 版本需要 `transformers>=4.31`。**

| 大小 | 聊天命令 | Hugging Face 仓库 |
| ---  | --- | --- |
| 7B   | `python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5`  | [lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)   |
| 7B-16k   | `python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5-16k`  | [lmsys/vicuna-7b-v1.5-16k](https://huggingface.co/lmsys/vicuna-7b-v1.5-16k)   |
| 13B  | `python3 -m fastchat.serve.cli --model-path lmsys/vicuna-13b-v1.5` | [lmsys/vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5) |
| 13B-16k  | `python3 -m fastchat.serve.cli --model-path lmsys/vicuna-13b-v1.5-16k` | [lmsys/vicuna-13b-v1.5-16k](https://huggingface.co/lmsys/vicuna-13b-v1.5-16k) |
| 33B  | `python3 -m fastchat.serve.cli --model-path lmsys/vicuna-33b-v1.3` | [lmsys/vicuna-33b-v1.3](https://huggingface.co/lmsys/vicuna-33b-v1.3) |

**旧权重**：查看 [docs/vicuna_weights_version.md](docs/vicuna_weights_version.md) 了解所有版本的权重及其差异。

### 其他模型
除了 Vicuna，我们还发布了两个额外的模型：[LongChat](https://lmsys.org/blog/2023-06-29-longchat/) 和 FastChat-T5。
你可以使用以下命令与它们聊天。它们会自动从 Hugging Face 仓库下载权重。

| 模型 | 聊天命令 | Hugging Face 仓库 |
| ---  | --- | --- |
| LongChat-7B   | `python3 -m fastchat.serve.cli --model-path lmsys/longchat-7b-32k-v1.5`  | [lmsys/longchat-7b-32k](https://huggingface.co/lmsys/longchat-7b-32k-v1.5)   |
| FastChat-T5-3B   | `python3 -m fastchat.serve.cli --model-path lmsys/fastchat-t5-3b-v1.0`  | [lmsys/fastchat-t5-3b-v1.0](https://huggingface.co/lmsys/fastchat-t5-3b-v1.0) |

## 命令行界面推理

<a href="https://chat.lmsys.org"><img src="assets/screenshot_cli.png" width="70%"></a>

（实验性功能：你可以指定 `--style rich` 来启用富文本输出和更好的非 ASCII 内容的文本流质量。这可能在某些终端上无法正常工作。）

#### 支持的模型
FastChat 支持广泛的模型，包括
LLama 2、Vicuna、Alpaca、Baize、ChatGLM、Dolly、Falcon、FastChat-T5、GPT4ALL、Guanaco、MTP、OpenAssistant、OpenChat、RedPajama、StableLM、WizardLM、xDAN-AI 等。

查看[这里](docs/model_support.md)了解支持的模型完整列表和添加新模型的说明。

#### 单 GPU
以下命令需要约 14GB GPU 内存用于 Vicuna-7B，28GB GPU 内存用于 Vicuna-13B。
如果你没有足够的内存，请参见下面的["内存不足"部分](#not-enough-memory)。
`--model-path` 可以是本地文件夹或 Hugging Face 仓库名称。
```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5
```

#### 多 GPU
你可以使用模型并行来聚合同一机器上多个 GPU 的 GPU 内存。
```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --num-gpus 2
```

提示：
有时 huggingface/transformers 中的"auto"设备映射策略并不能完美平衡多个 GPU 之间的内存分配。
你可以使用 `--max-gpu-memory` 来指定每个 GPU 用于存储模型权重的最大内存。
这允许它为激活分配更多内存，因此你可以使用更长的上下文长度或更大的批量大小。例如：

```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --num-gpus 2 --max-gpu-memory 8GiB
```

#### 仅 CPU
这仅在 CPU 上运行，不需要 GPU。它需要约 30GB CPU 内存用于 Vicuna-7B，约 60GB CPU 内存用于 Vicuna-13B。
```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --device cpu
```

使用 Intel AI 加速器 AVX512_BF16/AMX 来加速 CPU 推理。
```
CPU_ISA=amx python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --device cpu
```

#### Metal 后端（搭载 Apple Silicon 或 AMD GPU 的 Mac 电脑）
使用 `--device mps` 在 Mac 电脑上启用 GPU 加速（需要 torch >= 2.0）。
使用 `--load-8bit` 开启 8 位压缩。
```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --device mps --load-8bit
```
Vicuna-7B 可以在 32GB M1 Macbook 上以 1-2 字/秒的速度运行。

#### Intel XPU（Intel 数据中心和 Arc A 系列 GPU）
安装 [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html)。设置 OneAPI 环境变量：
```
source /opt/intel/oneapi/setvars.sh
```

使用 `--device xpu` 启用 XPU/GPU 加速。
```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --device xpu
```
Vicuna-7B 可以在 Intel Arc A770 16GB 上运行。

#### Ascend NPU
安装 [Ascend PyTorch Adapter](https://github.com/Ascend/pytorch)。设置 CANN 环境变量：
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

使用 `--device npu` 启用 NPU 加速。
```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --device npu
```
Vicuna-7B/13B 可以在 Ascend NPU 上运行。

#### 内存不足
如果你没有足够的内存，可以通过在上述命令中添加 `--load-8bit` 来启用 8 位压缩。
这可以将内存使用量减少约一半，模型质量略有下降。
它与 CPU、GPU 和 Metal 后端兼容。

使用 8 位压缩的 Vicuna-13B 可以在具有 16GB VRAM 的单个 GPU 上运行，如 Nvidia RTX 3090、RTX 4080、T4、V100（16GB）或 AMD RX 6800 XT。

```
python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5 --load-8bit
```

除此之外，你可以在上述命令中添加 `--cpu-offloading` 将不适合 GPU 的权重卸载到 CPU 内存中。
这需要启用 8 位压缩并安装 bitsandbytes 包，该包仅在 linux 操作系统上可用。

#### 更多平台和量化
- 对于 AMD GPU 用户，请在安装 FastChat 之前安装 ROCm 和 [PyTorch 的 ROCm 版本](https://pytorch.org/get-started/locally/)。另请参见此[帖子](https://github.com/lm-sys/FastChat/issues/104#issuecomment-1613791563)。
- FastChat 支持 ExLlama V2。参见 [docs/exllama_v2.md](/docs/exllama_v2.md)。
- FastChat 支持使用 [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) 进行 GPTQ 4 位推理。参见 [docs/gptq.md](/docs/gptq.md)。
- FastChat 支持使用 [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) 进行 AWQ 4 位推理。参见 [docs/awq.md](/docs/awq.md)。
- [MLC LLM](https://mlc.ai/mlc-llm/)，由 [TVM Unity](https://github.com/apache/tvm/tree/unity) 编译器支持，通过 Vulkan、Metal、CUDA 和 WebGPU 在手机、消费级 GPU 和网络浏览器上原生部署 Vicuna。

#### 使用来自 modelscope 的模型
对于中国用户，你可以通过指定以下环境变量来使用来自 www.modelscope.cn 的模型。
```bash
export FASTCHAT_USE_MODELSCOPE=True
```

## Web GUI 服务

<a href="https://chat.lmsys.org"><img src="assets/screenshot_gui.png" width="70%"></a>

要使用 web UI 进行服务，你需要三个主要组件：与用户交互的 web 服务器、托管一个或多个模型的模型工作器，以及协调 web 服务器和模型工作器的控制器。你可以在[这里](docs/server_arch.md)了解更多关于架构的信息。

以下是在终端中要遵循的命令：

#### 启动控制器
```bash
python3 -m fastchat.serve.controller
```

此控制器管理分布式工作器。

#### 启动模型工作器
```bash
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5
```
等待进程完成加载模型，直到看到"Uvicorn running on ..."。模型工作器将自己注册到控制器。

要确保你的模型工作器正确连接到控制器，使用以下命令发送测试消息：
```bash
python3 -m fastchat.serve.test_message --model-name vicuna-7b-v1.5
```
你将看到一个简短的输出。

#### 启动 Gradio web 服务器
```bash
python3 -m fastchat.serve.gradio_web_server
```

这是用户将与之交互的用户界面。

通过遵循这些步骤，你将能够使用 web UI 服务你的模型。你现在可以打开浏览器并与模型聊天。
如果模型没有显示，请尝试重启 gradio web 服务器。

#### （可选）：高级功能、可扩展性、第三方 UI
- 你可以向单个控制器注册多个模型工作器，这可用于以更高吞吐量服务单个模型或同时服务多个模型。在这样做时，请为不同的模型工作器分配不同的 GPU 和端口。
```
# 工作器 0
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5 --controller http://localhost:21001 --port 31000 --worker http://localhost:31000
# 工作器 1
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path lmsys/fastchat-t5-3b-v1.0 --controller http://localhost:21001 --port 31001 --worker http://localhost:31001
```
- 你还可以启动多标签页 gradio 服务器，其中包括 Chatbot Arena 标签页。
```bash
python3 -m fastchat.serve.gradio_web_server_multi
```
- 基于 huggingface/transformers 的默认模型工作器具有很好的兼容性，但可能会很慢。如果你想要高吞吐量的批处理服务，可以尝试 [vLLM 集成](docs/vllm_integration.md)。
- 如果你想在自己的 UI 或第三方 UI 上托管它，请参见[第三方 UI](docs/third_party_ui.md)。

## API
### 兼容 OpenAI 的 RESTful API 和 SDK
FastChat 为其支持的模型提供兼容 OpenAI 的 API，因此你可以将 FastChat 用作 OpenAI API 的本地替代品。
FastChat 服务器与 [openai-python](https://github.com/openai/openai-python) 库和 cURL 命令都兼容。
REST API 可以在 Google Colab 免费版上执行，如我们仓库中的 [FastChat_API_GoogleColab.ipynb](https://github.com/lm-sys/FastChat/blob/main/playground/FastChat_API_GoogleColab.ipynb) 笔记本所示。
参见 [docs/openai_api.md](docs/openai_api.md)。

### Hugging Face 生成 API
参见 [fastchat/serve/huggingface_api.py](fastchat/serve/huggingface_api.py)。

### LangChain 集成
参见 [docs/langchain_integration](docs/langchain_integration.md)。

## 评估
我们使用 MT-bench，一组具有挑战性的多轮开放式问题来评估模型。
为了自动化评估过程，我们提示像 GPT-4 这样的强大 LLM 充当评判，评估模型响应的质量。
参见 [fastchat/llm_judge](fastchat/llm_judge) 中运行 MT-bench 的说明。

MT-bench 是评估你的模型的新推荐方式。如果你仍在寻找 vicuna 博客文章中使用的旧的 80 个问题，请访问 [vicuna-blog-eval](https://github.com/lm-sys/vicuna-blog-eval)。

## 微调
### 数据

Vicuna 是通过使用从 ShareGPT.com 收集的约 125K 用户共享对话，使用公共 API 对 Llama 基础模型进行微调创建的。为确保数据质量，我们将 HTML 转换回 markdown，并过滤掉一些不适当或低质量的样本。此外，我们将冗长的对话分成适合模型最大上下文长度的较小段落。有关清理 ShareGPT 数据的详细说明，请查看[这里](docs/commands/data_cleaning.md)。

我们不会发布 ShareGPT 数据集。如果你想尝试微调代码，你可以使用 [dummy_conversation.json](data/dummy_conversation.json) 中的一些虚拟对话运行它。你可以遵循相同的格式并插入你自己的数据。

### 代码和超参数
我们的代码基于 [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)，并增加了对多轮对话的支持。
我们使用与 Stanford Alpaca 类似的超参数。

| 超参数 | 全局批量大小 | 学习率 | 轮数 | 最大长度 | 权重衰减 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vicuna-13B | 128 | 2e-5 | 3 | 2048 | 0 |

### 使用本地 GPU 微调 Vicuna-7B

- 安装依赖
```bash
pip3 install -e ".[train]"
```

- 你可以使用以下命令使用 4 x A100（40GB）训练 Vicuna-7B。用实际的 Llama 权重路径更新 `--model_name_or_path`，用实际的数据路径更新 `--data_path`。
```bash
torchrun --nproc_per_node=4 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path data/dummy_conversation.json \
    --bf16 True \
    --output_dir output_vicuna \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True
```

提示：
- 如果你使用的是不支持 FlashAttention 的 V100，你可以使用 [xFormers](https://github.com/facebookresearch/xformers) 中实现的[内存高效注意力](https://arxiv.org/abs/2112.05682)。安装 xformers 并将上面的 `fastchat/train/train_mem.py` 替换为 [fastchat/train/train_xformers.py](fastchat/train/train_xformers.py)。
- 如果你由于"FSDP Warning: When using FSDP, it is efficient and recommended..."而遇到内存不足，请参见[这里](https://github.com/huggingface/transformers/issues/24724#issuecomment-1645189539)的解决方案。
- 如果你在模型保存期间遇到内存不足，请参见[这里](https://github.com/pytorch/pytorch/issues/98823)的解决方案。
- 要开启记录到流行的实验跟踪工具（如 Tensorboard、MLFlow 或 Weights & Biases），请使用 `report_to` 参数，例如传递 `--report_to wandb` 以开启记录到 Weights & Biases。

### 其他模型、平台和 LoRA 支持
更多关于训练其他模型（例如，FastChat-T5）和使用 LoRA 的说明在 [docs/training.md](docs/training.md) 中。

### 使用 SkyPilot 在任何云上微调
[SkyPilot](https://github.com/skypilot-org/skypilot) 是由加州大学伯克利分校开发的框架，用于在任何云（AWS、GCP、Azure、Lambda 等）上轻松且经济高效地运行 ML 工作负载。
在[这里](https://github.com/skypilot-org/skypilot/tree/master/llm/vicuna)查看 SkyPilot 文档，了解如何使用托管竞价实例来训练 Vicuna 并节省云成本。

## 引用
本仓库中的代码（训练、服务和评估）主要是为以下论文开发或从中派生的。
如果你觉得该仓库有帮助，请引用它。

```
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

我们也计划向这个仓库添加更多我们的研究。
