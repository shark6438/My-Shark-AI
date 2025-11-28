# 🦈 Shark-AI: 基于 Qwen2.5 的私有化情感陪伴机器人

> **"不仅仅是聊天，它拥有记忆，也拥有灵魂。"**
>
> 一个融合了 **LoRA 微调 (Fine-tuning)** 与 **RAG (检索增强生成)** 技术的本地化 LLM 全栈项目。专为 **RTX 3060 (6GB)** 等消费级显卡优化。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Torch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)
![RAG](https://img.shields.io/badge/RAG-LangChain-orange)

## 📖 项目简介

本项目旨在构建一个具备**特定人格**且**拥有私有知识**的 AI 助手。通过微调技术，我们将通用的 Qwen2.5 模型改造为温柔体贴的 "Shark"；通过 RAG 技术，我们赋予了它读取私有文档（如 `secret.txt`）的能力，使其能回答关于用户的隐私问题。

**核心亮点：**
- **🧠 双核驱动**：结合了微调带来的“性格”和 RAG 带来的“知识”。
- **⚡️ 低显存优化**：使用 `bitsandbytes` 4-bit 量化，在 6GB 显存下流畅运行 1.5B 模型。
- **🖥️ 完整 UI**：基于 Streamlit 构建的 Web 聊天界面，支持流式输出和多轮对话。

## 🛠️ 技术栈

- **基座模型**: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **微调技术**: LoRA (Low-Rank Adaptation) / PEFT
- **量化推理**: bitsandbytes (4-bit NF4)
- **RAG 框架**: LangChain + ChromaDB (向量数据库)
- **Embedding**: sentence-transformers/all-MiniLM-L6-v2
- **前端界面**: Streamlit

## 📂 文件结构

```text
Shark-AI/
├── shark_lora_output/      # 存放微调后的 LoRA 权重 (训练产物)
├── secret.txt              # RAG 知识库源文件 (你的私有数据)
├── shark_identity.json     # 微调用的自我认知数据集
├── app.py                  # 启动主程序 (即原本的 step13 代码)
├── train_lora.py           # 微调训练脚本
├── requirements.txt        # 依赖列表
└── README.md               # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备

建议使用 Conda 创建虚拟环境：

codeBash



```
conda create -n shark_ai python=3.10
conda activate shark_ai
```

### 2. 安装依赖

**注意：Windows 用户请务必先手动安装 bitsandbytes！**

codeBash



```
# Windows 用户请执行以下命令 (非官方轮子，支持 CUDA):
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.2.post2-py3-none-win_amd64.whl

# 然后安装其他依赖
pip install -r requirements.txt
```

### 3. 数据准备

1. 
2. 确保 shark_lora_output 文件夹存在（需要先运行训练脚本）。
3. 在 secret.txt 中写入你想让 AI 记住的秘密。

### 4. 启动应用

codeBash



```
streamlit run app.py
```

浏览器将自动打开，即可与你的专属 Shark 对话！

## 🎓 致谢与学习

本项目是 NLP 学习过程中的实战产物。特别感谢 HuggingFace 和 ModelScope 社区提供的开源模型与工具。

------




*Created with ❤️ by Shark*
