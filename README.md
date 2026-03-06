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

全功能图像工具箱 — 13 个 AI 功能，全部基于 HF Inference API，无需本地 GPU。

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

### 标注工具（新增）
| # | 功能 | 说明 | 技术 |
|---|------|------|------|
| 14 | AI 预标注引擎 | 零样本检测 + 分割 + 描述 + VQA 质检 | OWLv2 / SegFormer / GIT / ViLT |

## HF Auto Annotator（独立应用）

`annotator_app.py` 是一个独立的 **AI 预标注引擎**，定位为 Label Studio / CVAT 的上游预标注工具。

### 核心功能

| 功能 | 说明 | 模型 |
|------|------|------|
| 零样本目标检测 | 输入文本类别，自动检测 bounding box | OWLv2-base-patch16 |
| 语义分割 | 像素级分割 mask | SegFormer-b5 / DETR |
| 图像描述 | 自动生成标注辅助描述 | GIT-Large / BLIP |
| VQA 标注质检 | 自动验证标注正确性，标记低置信度 | ViLT VQA |

### 导出格式

- **COCO JSON**：标准格式，可直接导入 Label Studio / FiftyOne / CVAT
- **YOLO TXT**：坐标归一化，可直接用于 YOLO 训练
- **CSV**：简洁表格，便于数据分析

### 运行方式

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxx"
pip install -r requirements.txt
# 完整标注引擎
python annotator_app.py
# 或使用主应用中的快速入口
python app.py
```

### 竞品对比

| 竞品 | 弱点 | HF Auto Annotator 优势 |
|------|------|------------------------|
| Label Studio | AI 预标注需自建 ML Backend，配置复杂 | 零配置，HF API 一键调用 |
| CVAT | 模型市场有限，部署重 | 可调用 HF Hub 上千模型 |
| Roboflow | $249/月起步 | 免费开源，HF Spaces 零成本部署 |

## 快速开始

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxx"
pip install -r requirements.txt
python app.py
```

浏览器打开 http://localhost:7860

## 部署

详见 [DEPLOY.md](./DEPLOY.md)

## 研究方向

详见 [ROADMAP.md](./ROADMAP.md)

## License

MIT

