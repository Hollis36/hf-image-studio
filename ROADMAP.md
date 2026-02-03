# HF Image Studio — 研究方向与实现状态

基于当前项目的技术栈和 HF 平台能力，规划的扩展方向及实现状态。

---

## 实现状态总览

| # | 功能 | 状态 | 所在标签 |
|---|------|------|----------|
| 1 | 文生图 (Text-to-Image) | 已实现 | 生成 > 文生图 |
| 2 | Prompt 优化助手 | 已实现 | 生成 > Prompt 助手 |
| 3 | 图生视频 (Text-to-Video) | 已实现 | 生成 > 图生视频 |
| 4 | 风格转换 (Style Transfer) | 已实现 | 编辑 > 风格转换 |
| 5 | 图像编辑 (Inpainting) | 已实现 | 编辑 > 图像编辑 |
| 6 | 超分辨率 (Super Resolution) | 已实现 | 编辑 > 超分辨率 |
| 7 | 背景移除 (Background Removal) | 已实现 | 编辑 > 背景移除 |
| 8 | 图像描述 (Image Captioning) | 已实现 | 理解 > 图像描述 |
| 9 | 视觉问答 (Visual QA) | 已实现 | 理解 > 视觉问答 |
| 10 | OCR 文字识别 | 已实现 | 理解 > OCR 识别 |
| 11 | 图像分割 (Segmentation) | 已实现 | 理解 > 图像分割 |
| 12 | AI Agent (Smolagents) | 已实现 | 智能 > AI Agent |
| 13 | 批量处理 (Batch Processing) | 已实现 | 智能 > 批量处理 |

---

## 未来可继续拓展的方向

### 进阶功能

| 方向 | 说明 | 前置条件 |
|------|------|----------|
| 文生 3D (Text-to-3D) | 文字 → 3D 模型，需要 Gradio Model3D 组件 | ZeroGPU + shap-e 模型 |
| 图生视频进阶 | 图片 → 视频（SVD / Wan 2.1），效果更好 | ZeroGPU 或付费 GPU Space |
| 多轮 Agent 对话 | 支持上传图片 + 多轮对话式编辑 | Smolagents 多模态支持 |
| 批量 Pipeline 编排 | 自定义多步处理流水线（去背景 → 超分 → 统一尺寸） | 异步任务队列 |
| ControlNet 控制生成 | 通过边缘图/深度图/姿态图精确控制生成 | ZeroGPU + Diffusers |

### 社区贡献方向

| 方向 | 说明 |
|------|------|
| 中文数据集建设 | 收集中文图像描述数据集上传 HF Hub |
| 模型评测 | 构建 Text-to-Image 自动评测（FID / CLIP Score） |
| 贡献上游项目 | 为 diffusers / gradio / smolagents 提交 PR |
| 翻译文档 | 翻译 HF 官方文档和教程为中文 |
| 自定义 Gradio 组件 | 开发图像编辑/标注相关的 Gradio 自定义组件 |

---

## 技术栈演进路线

```
当前（已完成）             短期                        中期
──────────────────────────────────────────────────────────────
Gradio + Inference API  → + ZeroGPU 本地推理      → + 自定义训练
13 个功能模块            → + ControlNet / 3D       → + 微调自有模型
Smolagents 基础 Agent   → + 多轮对话 Agent        → + 多 Agent 协作
CPU Space               → + ZeroGPU (H200 70GB)   → + Dedicated GPU
```

---

## 升级为 ZeroGPU 的改造要点

当前项目全部使用 Inference API 远程调用，升级 ZeroGPU 后可改为本地推理：

```python
import spaces
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0"
)

@spaces.GPU  # 仅在此函数执行时占用 GPU
def text_to_image(prompt, ...):
    return pipe(prompt=prompt, ...).images[0]
```

优势：推理速度提升 5-10x，不受 API 速率限制，可使用更大模型。
