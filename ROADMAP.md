# HF Image Studio — 研究方向与实现状态

基于当前项目的技术栈和 HF 平台能力，规划的扩展方向及实现状态。

---

## 最新方向：AI 辅助数据标注预标注平台

HF Image Studio 正在向 **AI 辅助数据标注预标注平台** 垂直化，定位为 Label Studio / CVAT 的上游预标注引擎。

新增 **HF Auto Annotator**（`annotator_app.py`），用户上传图片，系统自动完成分割、检测、描述等预标注，导出标准 COCO JSON / YOLO TXT 格式，可直接导入标注平台进行人工校正。

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
| 14 | AI 预标注引擎 (Auto Annotator) | 已实现 | 标注工具 / annotator_app.py |

---

## HF Auto Annotator 技术栈

### 核心能力

| 能力 | 技术 | 状态 |
|------|------|------|
| 零样本目标检测 | OWLv2 (google/owlv2-base-patch16-ensemble) | 已实现 |
| 语义/全景分割 | SegFormer-b5 / DETR Panoptic | 已实现 |
| 图像描述生成 | GIT-Large-COCO / BLIP-Large | 已实现 |
| VQA 标注质检 | ViLT VQA | 已实现 |
| COCO JSON 导出 | 标准格式，Label Studio / FiftyOne 兼容 | 已实现 |
| YOLO TXT 导出 | 归一化坐标，YOLO 训练直接可用 | 已实现 |
| CSV 导出 | 表格格式，便于数据分析 | 已实现 |
| 批量处理 + ZIP 下载 | 最多 20 张，ZIP 打包下载 | 已实现 |

### 竞品对比

| 竞品 | 弱点 | HF Auto Annotator 优势 |
|------|------|------------------------|
| Label Studio | AI 预标注需自建 ML Backend，配置复杂 | 零配置，HF API 一键调用 |
| CVAT | 模型市场有限，部署重 | 可调用 HF Hub 上千模型 |
| EISeg（百度） | 绑定 PaddleSeg 生态 | 开放灵活，不绑定 |
| Roboflow | $249/月起步 | 免费开源，HF Spaces 零成本部署 |

---

## 未来可继续拓展的方向

### 预标注引擎增强

| 方向 | 说明 | 前置条件 |
|------|------|----------|
| Grounded SAM 2 集成 | 结合 Grounding DINO + SAM2 实现精细分割 | ZeroGPU / 付费 API |
| 实例分割多边形优化 | 用 cv2.findContours 输出精细多边形 | 添加 opencv-python 依赖 |
| 关键点检测 | 人体姿态 / 骨骼关键点预标注 | ViTPose 或 MediaPipe |
| 多轮交互式标注 | 基于 SAM2 的点击式分割 | ZeroGPU |
| 标注版本管理 | 记录多次修改历史 | 数据库或 Git 后端 |

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
当前（已完成）                    短期                        中期
──────────────────────────────────────────────────────────────────────
Gradio + Inference API          → + ZeroGPU 本地推理      → + 自定义训练
13 个功能模块                    → + ControlNet / 3D       → + 微调自有模型
Smolagents 基础 Agent           → + 多轮对话 Agent        → + 多 Agent 协作
Auto Annotator (COCO/YOLO 导出) → + SAM2 精细分割         → + 标注平台深度集成
CPU Space                       → + ZeroGPU (H200 70GB)   → + Dedicated GPU
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

