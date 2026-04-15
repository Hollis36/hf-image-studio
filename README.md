---
title: HF Image Studio
emoji: 🎨
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: "13-in-1 AI Image Toolkit powered by HF Inference API"
---

# HF Image Studio

A lightweight image generation, editing, and understanding studio powered by Hugging Face Inference API.

HF Image Studio 是一个面向实验、演示和快速原型的 Gradio 图像工具箱。它把 Hugging Face Inference API 封装成统一界面，覆盖图像生成、编辑、理解和批量处理场景，不要求本地 GPU。

## Project Overview

这个仓库适合：

- 想快速搭建 Hugging Face 图像应用 demo 的开发者
- 需要一个多功能图像实验台的研究者
- 想部署到 Hugging Face Spaces 的个人项目或作品集
- 想研究 Gradio + `huggingface_hub` + Smolagents 组合方式的用户

当前版本以 API 驱动为主，重点是快速集成和低部署门槛，而不是本地大模型推理。

## 功能一览

### 生成
| # | 功能 | 说明 | 模型 |
|---|------|------|------|
| 1 | 文生图 | 文字描述 → 图像 | FLUX.1-schnell / SDXL / SD3.5 |
| 2 | Prompt 助手 | LLM 优化提示词 + 生成 | Qwen / Llama / Mistral |
| 3 | 图生视频 | 文字描述 → 短视频 | AnimateDiff Lightning |

### 编辑
| # | 功能 | 说明 | 模型 |
|---|------|------|------|
| 4 | 风格转换 | 图片 + 风格描述 → 风格化图像 | FLUX.1 / SDXL Refiner |
| 5 | 图像编辑 | 涂抹区域 + 描述 → 局部重绘 | SDXL Inpainting |
| 6 | 超分辨率 | 低分辨率 → 高清增强 | SwinSR x2/x4 |
| 7 | 背景移除 | 一键移除背景 | RMBG-2.0 |

### 理解
| # | 功能 | 说明 | 模型 |
|---|------|------|------|
| 8 | 图像描述 | 图片 → 文字描述 | GIT-Large / BLIP |
| 9 | 视觉问答 | 图片 + 问题 → 回答 | ViLT VQA |
| 10 | OCR 识别 | 图片 → 提取文字 | TrOCR (印刷/手写) |
| 11 | 图像分割 | 语义分割 + 可视化 | SegFormer / DETR |

### 智能
| # | 功能 | 说明 | 技术 |
|---|------|------|------|
| 12 | AI Agent | 自然语言指令编排任务链 | Smolagents |
| 13 | 批量处理 | 多图批量执行同一操作 | 背景移除/超分/描述 |

## What's Inside

- `app.py`: Gradio 应用主入口
- `requirements.txt`: 运行依赖
- `README.md`: 仓库说明和 HF Space 元数据
- `DEPLOY.md`: 本地运行和 Hugging Face Spaces 部署说明
- `ROADMAP.md`: 功能扩展和研究方向记录

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入你自己的 HF_TOKEN
python app.py
```

应用默认读取环境变量 `HF_TOKEN`。推荐做法：

- 本地开发：在 shell 中导出 `HF_TOKEN`，或使用你自己的 `.env` 管理方式
- Hugging Face Spaces：把 `HF_TOKEN` 配置到 Repository secrets
- 不要把真实 token 写进源码、README、截图或提交记录

浏览器打开 `http://localhost:7860`

## How To Use

1. 配置好 `HF_TOKEN`
2. 安装依赖并启动 `app.py`
3. 按功能标签页体验生成、编辑、理解和智能编排
4. 如果准备公开部署，先阅读 [DEPLOY.md](./DEPLOY.md) 并完成 secrets 检查

## Privacy And Release Notes

公开前建议确认：

- 没有提交真实 token、`.env`、cookies 或浏览器导出的认证文件
- 没有在 README、日志、截图里暴露私人账号信息
- 没有依赖私有模型、私有 Space 或受限访问资源
- 如果你增加了自定义模型或提示词模板，确认它们适合公开发布

## 部署

详见 [DEPLOY.md](./DEPLOY.md)

## 研究方向

详见 [ROADMAP.md](./ROADMAP.md)

## License

MIT
