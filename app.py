"""
HF Image Studio — 全功能图像工具箱
13 个功能模块，全部通过 HF Inference API 调用，无需本地 GPU
"""

import os
import io
import tempfile
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import InferenceClient

# ── Smolagents（可选） ────────────────────────────────────────
try:
    from smolagents import ToolCallingAgent, tool, InferenceClientModel

    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

# ── 客户端初始化 ──────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

# ── 模型配置 ──────────────────────────────────────────────────
TEXT2IMG_MODELS = {
    "FLUX.1-schnell (快速)": "black-forest-labs/FLUX.1-schnell",
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "SD 3.5 Large Turbo": "stabilityai/stable-diffusion-3.5-large-turbo",
}

IMG2IMG_MODELS = {
    "FLUX.1-schnell": "black-forest-labs/FLUX.1-schnell",
    "SD XL Refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
}

INPAINT_MODELS = {
    "SD XL Inpainting": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "RunwayML Inpainting": "runwayml/stable-diffusion-inpainting",
}

SR_MODELS = {
    "SwinSR x4 (真实感)": "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    "SwinSR x2 (经典)": "caidas/swin2SR-classical-sr-x2-64",
}

RMBG_MODELS = {
    "RMBG-2.0 (推荐)": "briaai/RMBG-2.0",
    "RMBG-1.4": "briaai/RMBG-1.4",
}

SEG_MODELS = {
    "SegFormer (场景分割)": "nvidia/segformer-b5-finetuned-ade-640-640",
    "DETR (全景分割)": "facebook/detr-resnet-50-panoptic",
}

CAPTION_MODELS = {
    "GIT-Large-COCO": "microsoft/git-large-coco",
    "BLIP-Large": "Salesforce/blip-image-captioning-large",
}

VQA_MODELS = {
    "ViLT VQA": "dandelin/vilt-b32-finetuned-vqa",
}

OCR_MODELS = {
    "TrOCR Large (印刷体)": "microsoft/trocr-large-printed",
    "TrOCR Large (手写体)": "microsoft/trocr-large-handwritten",
}

VIDEO_MODELS = {
    "AnimateDiff Lightning": "ByteDance/AnimateDiff-Lightning",
}

CHAT_MODELS = {
    "Llama 3.1 8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Mistral 7B v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct",
}


# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def _pil_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def _bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


# ══════════════════════════════════════════════════════════════
#  1. 文生图 Text-to-Image
# ══════════════════════════════════════════════════════════════

def text_to_image(prompt, negative_prompt, model_name, width, height, guidance, steps):
    if not prompt.strip():
        raise gr.Error("请输入图像描述")
    image = client.text_to_image(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt.strip() else None,
        model=TEXT2IMG_MODELS[model_name],
        width=int(width),
        height=int(height),
        guidance_scale=guidance,
        num_inference_steps=int(steps),
    )
    return image


# ══════════════════════════════════════════════════════════════
#  2. Prompt 优化助手
# ══════════════════════════════════════════════════════════════

PROMPT_SYSTEM = """You are an expert prompt engineer for AI image generation.
Given a simple description, expand it into a detailed, high-quality prompt.
Include: subject details, art style, lighting, composition, quality keywords.
Output ONLY the enhanced prompt, nothing else. Keep it under 200 words."""


def enhance_prompt(simple_prompt, chat_model):
    if not simple_prompt.strip():
        raise gr.Error("请输入简单描述")
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": simple_prompt},
        ],
        model=CHAT_MODELS[chat_model],
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def enhance_and_generate(simple_prompt, chat_model, img_model, width, height):
    enhanced = enhance_prompt(simple_prompt, chat_model)
    image = client.text_to_image(
        prompt=enhanced,
        model=TEXT2IMG_MODELS[img_model],
        width=int(width),
        height=int(height),
    )
    return enhanced, image


# ══════════════════════════════════════════════════════════════
#  3. 图生视频 Image/Text-to-Video
# ══════════════════════════════════════════════════════════════

def text_to_video(prompt, model_name):
    if not prompt.strip():
        raise gr.Error("请输入视频描述")
    video_bytes = client.text_to_video(
        prompt=prompt,
        model=VIDEO_MODELS[model_name],
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video_bytes)
    tmp.flush()
    return tmp.name


# ══════════════════════════════════════════════════════════════
#  4. 风格转换 Style Transfer (Image-to-Image)
# ══════════════════════════════════════════════════════════════

def style_transfer(image, prompt, model_name, strength):
    if image is None:
        raise gr.Error("请上传图片")
    if not prompt.strip():
        raise gr.Error("请输入风格描述")
    return client.image_to_image(
        image=image,
        prompt=prompt,
        model=IMG2IMG_MODELS[model_name],
        strength=strength,
    )


# ══════════════════════════════════════════════════════════════
#  5. 图像编辑 Inpainting
# ══════════════════════════════════════════════════════════════

def inpaint(editor_value, prompt, model_name, strength):
    if editor_value is None:
        raise gr.Error("请上传图片并涂抹要编辑的区域")
    if not prompt.strip():
        raise gr.Error("请输入编辑描述")

    # 从 ImageEditor 提取原图和遮罩
    bg = editor_value.get("background")
    layers = editor_value.get("layers", [])
    if bg is None:
        raise gr.Error("请上传底图")

    # 合并所有图层为遮罩
    if isinstance(bg, np.ndarray):
        bg = Image.fromarray(bg)
    mask = Image.new("L", bg.size, 0)
    for layer in layers:
        if isinstance(layer, np.ndarray):
            layer = Image.fromarray(layer)
        if layer.mode == "RGBA":
            alpha = layer.split()[-1]
            mask = Image.composite(Image.new("L", bg.size, 255), mask, alpha)
        else:
            gray = layer.convert("L")
            mask = Image.composite(Image.new("L", bg.size, 255), mask, gray)

    # 使用 image-to-image 模拟 inpainting
    result = client.image_to_image(
        image=bg,
        prompt=prompt,
        model=INPAINT_MODELS[model_name],
        strength=strength,
    )
    return result


# ══════════════════════════════════════════════════════════════
#  6. 超分辨率 Super Resolution
# ══════════════════════════════════════════════════════════════

def super_resolution(image, model_name):
    if image is None:
        raise gr.Error("请上传图片")
    img_bytes = _pil_to_bytes(image)
    response = client.post(model=SR_MODELS[model_name], data=img_bytes)
    result = _bytes_to_pil(response)
    orig_w, orig_h = image.size
    new_w, new_h = result.size
    info = f"原始: {orig_w}x{orig_h} → 增强后: {new_w}x{new_h}"
    return result, info


# ══════════════════════════════════════════════════════════════
#  7. 背景移除 Background Removal
# ══════════════════════════════════════════════════════════════

def remove_background(image, model_name):
    if image is None:
        raise gr.Error("请上传图片")
    img_bytes = _pil_to_bytes(image)
    response = client.post(model=RMBG_MODELS[model_name], data=img_bytes)
    result = _bytes_to_pil(response)
    return result


# ══════════════════════════════════════════════════════════════
#  8. 图像描述 Image Captioning
# ══════════════════════════════════════════════════════════════

def image_caption(image, model_name):
    if image is None:
        raise gr.Error("请上传图片")
    result = client.image_to_text(image=image, model=CAPTION_MODELS[model_name])
    if isinstance(result, list):
        return result[0].get("generated_text", str(result))
    if isinstance(result, dict):
        return result.get("generated_text", str(result))
    return str(result)


# ══════════════════════════════════════════════════════════════
#  9. 视觉问答 Visual QA
# ══════════════════════════════════════════════════════════════

def visual_qa(image, question, model_name):
    if image is None:
        raise gr.Error("请上传图片")
    if not question.strip():
        raise gr.Error("请输入问题")
    result = client.visual_question_answering(
        image=image,
        question=question,
        model=VQA_MODELS[model_name],
    )
    if isinstance(result, list):
        lines = [f"• {item['answer']}（置信度 {item['score']:.2%}）" for item in result[:5]]
        return "\n".join(lines)
    return str(result)


# ══════════════════════════════════════════════════════════════
# 10. OCR 文字识别
# ══════════════════════════════════════════════════════════════

def ocr_recognize(image, model_name):
    if image is None:
        raise gr.Error("请上传图片")
    img_bytes = _pil_to_bytes(image)
    result = client.image_to_text(image=image, model=OCR_MODELS[model_name])
    if isinstance(result, list):
        return result[0].get("generated_text", str(result))
    if isinstance(result, dict):
        return result.get("generated_text", str(result))
    return str(result)


# ══════════════════════════════════════════════════════════════
# 11. 图像分割 Segmentation
# ══════════════════════════════════════════════════════════════

SEGMENT_COLORS = [
    (255, 0, 0, 120), (0, 255, 0, 120), (0, 0, 255, 120),
    (255, 255, 0, 120), (255, 0, 255, 120), (0, 255, 255, 120),
    (255, 128, 0, 120), (128, 0, 255, 120), (0, 128, 255, 120),
    (255, 0, 128, 120), (128, 255, 0, 120), (0, 255, 128, 120),
]


def image_segmentation(image, model_name):
    if image is None:
        raise gr.Error("请上传图片")
    segments = client.image_segmentation(
        image=image, model=SEG_MODELS[model_name]
    )

    # 可视化
    overlay = image.copy().convert("RGBA")
    labels = []
    for i, seg in enumerate(segments):
        label = seg.get("label", f"区域{i}")
        score = seg.get("score", 0)
        labels.append(f"• {label}（{score:.1%}）")

        mask = seg.get("mask")
        if mask is None:
            continue
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        mask_l = mask.convert("L")

        color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
        color_layer = Image.new("RGBA", overlay.size, color)
        overlay = Image.composite(color_layer, overlay, mask_l)

    result = Image.alpha_composite(
        image.convert("RGBA"), Image.blend(
            Image.new("RGBA", image.size, (0, 0, 0, 0)), overlay, 0.5
        )
    )
    return result.convert("RGB"), "\n".join(labels) if labels else "未检测到分割区域"


# ══════════════════════════════════════════════════════════════
# 12. AI Agent（Smolagents）
# ══════════════════════════════════════════════════════════════

def _save_tmp_image(img: Image.Image, name: str = "agent_out") -> str:
    path = os.path.join(tempfile.gettempdir(), f"{name}.png")
    img.save(path)
    return path


def build_agent():
    if not AGENT_AVAILABLE:
        return None

    @tool
    def generate_image(prompt: str) -> str:
        """Generate an image from a text description.
        Args:
            prompt: Text description of the image to generate.
        Returns:
            File path of the generated image.
        """
        img = client.text_to_image(
            prompt=prompt, model="black-forest-labs/FLUX.1-schnell"
        )
        return _save_tmp_image(img, "agent_gen")

    @tool
    def describe_image(image_path: str) -> str:
        """Describe the contents of an image.
        Args:
            image_path: File path of the image.
        Returns:
            Text description of the image.
        """
        img = Image.open(image_path)
        result = client.image_to_text(image=img, model="microsoft/git-large-coco")
        if isinstance(result, (list, dict)):
            return str(result)
        return result

    @tool
    def remove_bg(image_path: str) -> str:
        """Remove background from an image.
        Args:
            image_path: File path of the image.
        Returns:
            File path of the result image with background removed.
        """
        img = Image.open(image_path)
        response = client.post(model="briaai/RMBG-2.0", data=_pil_to_bytes(img))
        result = _bytes_to_pil(response)
        return _save_tmp_image(result, "agent_rmbg")

    @tool
    def upscale(image_path: str) -> str:
        """Upscale and enhance an image to higher resolution.
        Args:
            image_path: File path of the image.
        Returns:
            File path of the upscaled image.
        """
        img = Image.open(image_path)
        response = client.post(
            model="caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
            data=_pil_to_bytes(img),
        )
        result = _bytes_to_pil(response)
        return _save_tmp_image(result, "agent_sr")

    @tool
    def apply_style(image_path: str, style_prompt: str) -> str:
        """Apply an artistic style to an image.
        Args:
            image_path: File path of the image.
            style_prompt: Description of the style to apply.
        Returns:
            File path of the stylized image.
        """
        img = Image.open(image_path)
        result = client.image_to_image(
            image=img,
            prompt=style_prompt,
            model="black-forest-labs/FLUX.1-schnell",
            strength=0.65,
        )
        return _save_tmp_image(result, "agent_style")

    try:
        model = InferenceClientModel(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            token=HF_TOKEN,
        )
        return ToolCallingAgent(
            tools=[generate_image, describe_image, remove_bg, upscale, apply_style],
            model=model,
            max_steps=6,
        )
    except Exception:
        return None


_agent = build_agent()


def run_agent(instruction):
    if _agent is None:
        return "Agent 不可用。请确保已安装 smolagents 并设置 HF_TOKEN。", None
    if not instruction.strip():
        raise gr.Error("请输入指令")
    try:
        result = _agent.run(instruction)
        # 尝试加载结果图片
        if isinstance(result, str) and os.path.isfile(result):
            return f"执行完成，结果已保存: {result}", Image.open(result)
        return str(result), None
    except Exception as e:
        return f"执行出错: {e}", None


# ══════════════════════════════════════════════════════════════
# 13. 批量处理 Batch Processing
# ══════════════════════════════════════════════════════════════

BATCH_OPS = ["背景移除", "超分辨率", "图像描述"]


def batch_process(files, operation):
    if not files:
        raise gr.Error("请上传至少一张图片")

    results = []
    captions = []

    for f in files[:10]:  # 限制最多 10 张
        img = Image.open(f).convert("RGB")
        try:
            if operation == "背景移除":
                resp = client.post(model="briaai/RMBG-2.0", data=_pil_to_bytes(img))
                results.append(_bytes_to_pil(resp))
                captions.append("背景已移除")
            elif operation == "超分辨率":
                resp = client.post(
                    model="caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
                    data=_pil_to_bytes(img),
                )
                results.append(_bytes_to_pil(resp))
                captions.append(f"增强完成")
            elif operation == "图像描述":
                text = client.image_to_text(
                    image=img, model="microsoft/git-large-coco"
                )
                if isinstance(text, (list, dict)):
                    text = str(text)
                results.append(img)
                captions.append(str(text))
        except Exception as e:
            results.append(img)
            captions.append(f"处理失败: {e}")

    gallery = [(img, cap) for img, cap in zip(results, captions)]
    return gallery


# ══════════════════════════════════════════════════════════════
# 14. 标注工具 Auto Annotator (Quick Entry)
# ══════════════════════════════════════════════════════════════

_DETECTION_MODEL = "google/owlv2-base-patch16-ensemble"


def quick_annotate(image, text_prompt):
    """快速预标注入口：零样本目标检测 + 图像描述"""
    if image is None:
        raise gr.Error("请上传图片")
    if not text_prompt.strip():
        raise gr.Error("请输入目标类别（如 cat, dog）")

    labels = [lb.strip() for lb in text_prompt.split(",") if lb.strip()]
    status_lines = []
    vis = image.copy().convert("RGB")
    draw = ImageDraw.Draw(vis)
    colors = ["#FF4444", "#44FF44", "#4444FF", "#FFFF44", "#FF44FF", "#44FFFF"]

    # 零样本检测
    try:
        detections = client.zero_shot_object_detection(
            image=image, candidate_labels=labels, model=_DETECTION_MODEL
        )
        for i, det in enumerate(detections or []):
            box = det.get("box", {})
            xmin, ymin = box.get("xmin", 0), box.get("ymin", 0)
            xmax, ymax = box.get("xmax", 0), box.get("ymax", 0)
            label = det.get("label", "")
            score = det.get("score", 0)
            color = colors[i % len(colors)]
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(r, g, b), width=2)
            text = f"{label} {score:.2f}"
            text_x, text_y = xmin + 2, max(ymin - 13, 0)
            draw.rectangle(
                [text_x, text_y, text_x + len(text) * 6 + 4, text_y + 13],
                fill=(r, g, b),
            )
            draw.text((text_x + 2, text_y + 1), text, fill=(255, 255, 255))
        status_lines.append(f"✅ 检测到 {len(detections or [])} 个目标")
    except Exception as exc:
        status_lines.append(f"❌ 检测失败：{exc}")

    # 图像描述
    try:
        caption = client.image_to_text(image=image, model="microsoft/git-large-coco")
        if isinstance(caption, list):
            caption = caption[0].get("generated_text", str(caption))
        elif isinstance(caption, dict):
            caption = caption.get("generated_text", str(caption))
        status_lines.append(f"✅ 描述：{str(caption)[:80]}")
    except Exception as exc:
        status_lines.append(f"❌ 描述失败：{exc}")

    return vis, "\n".join(status_lines)


# ══════════════════════════════════════════════════════════════
#  Gradio 界面
# ══════════════════════════════════════════════════════════════

with gr.Blocks(title="HF Image Studio") as demo:
    gr.Markdown(
        """
        # HF Image Studio
        全功能图像工具箱 — 13 个 AI 功能，全部基于 HF Inference API，**无需本地 GPU**
        """
    )

    # ┌─────────────────────────────────────────────────────────
    # │ 生成
    # └─────────────────────────────────────────────────────────
    with gr.Tab("生成"):
        with gr.Tabs():

            # ── 1. 文生图 ──
            with gr.Tab("文生图"):
                with gr.Row():
                    with gr.Column():
                        t2i_prompt = gr.Textbox(label="图像描述", lines=3,
                            placeholder="A cat on the moon, digital art...")
                        t2i_neg = gr.Textbox(label="反向提示词", lines=2,
                            placeholder="blurry, low quality...")
                        t2i_model = gr.Dropdown(list(TEXT2IMG_MODELS), value="FLUX.1-schnell (快速)", label="模型")
                        with gr.Row():
                            t2i_w = gr.Slider(256, 1024, 512, step=64, label="宽度")
                            t2i_h = gr.Slider(256, 1024, 512, step=64, label="高度")
                        with gr.Row():
                            t2i_cfg = gr.Slider(1, 20, 7.5, step=0.5, label="引导强度")
                            t2i_steps = gr.Slider(1, 50, 4, step=1, label="推理步数")
                        t2i_btn = gr.Button("生成", variant="primary")
                    with gr.Column():
                        t2i_out = gr.Image(label="结果", type="pil")
                t2i_btn.click(text_to_image,
                    [t2i_prompt, t2i_neg, t2i_model, t2i_w, t2i_h, t2i_cfg, t2i_steps], t2i_out)

            # ── 2. Prompt 助手 ──
            with gr.Tab("Prompt 助手"):
                gr.Markdown("输入简单描述 → LLM 优化为专业 Prompt → 一键生成图像")
                with gr.Row():
                    with gr.Column():
                        pa_input = gr.Textbox(label="简单描述", lines=2, placeholder="一只猫坐在窗台上")
                        pa_chat = gr.Dropdown(list(CHAT_MODELS), value="Qwen 2.5 7B", label="LLM 模型")
                        pa_img_model = gr.Dropdown(list(TEXT2IMG_MODELS), value="FLUX.1-schnell (快速)", label="图像模型")
                        with gr.Row():
                            pa_w = gr.Slider(256, 1024, 512, step=64, label="宽度")
                            pa_h = gr.Slider(256, 1024, 512, step=64, label="高度")
                        with gr.Row():
                            pa_btn1 = gr.Button("仅优化 Prompt")
                            pa_btn2 = gr.Button("优化 + 生成图像", variant="primary")
                    with gr.Column():
                        pa_enhanced = gr.Textbox(label="优化后的 Prompt", lines=6)
                        pa_image = gr.Image(label="生成结果", type="pil")
                pa_btn1.click(enhance_prompt, [pa_input, pa_chat], pa_enhanced)
                pa_btn2.click(enhance_and_generate,
                    [pa_input, pa_chat, pa_img_model, pa_w, pa_h], [pa_enhanced, pa_image])

            # ── 3. 图生视频 ──
            with gr.Tab("图生视频"):
                gr.Markdown("通过文字描述生成短视频（需要模型支持，部分模型可能需要 PRO 订阅）")
                with gr.Row():
                    with gr.Column():
                        vid_prompt = gr.Textbox(label="视频描述", lines=3,
                            placeholder="A rocket launching into space, cinematic...")
                        vid_model = gr.Dropdown(list(VIDEO_MODELS), value="AnimateDiff Lightning", label="模型")
                        vid_btn = gr.Button("生成视频", variant="primary")
                    with gr.Column():
                        vid_out = gr.Video(label="生成结果")
                vid_btn.click(text_to_video, [vid_prompt, vid_model], vid_out)

    # ┌─────────────────────────────────────────────────────────
    # │ 编辑
    # └─────────────────────────────────────────────────────────
    with gr.Tab("编辑"):
        with gr.Tabs():

            # ── 4. 风格转换 ──
            with gr.Tab("风格转换"):
                with gr.Row():
                    with gr.Column():
                        st_img = gr.Image(label="上传图片", type="pil")
                        st_prompt = gr.Textbox(label="风格描述", lines=2,
                            placeholder="in the style of Van Gogh, oil painting...")
                        st_model = gr.Dropdown(list(IMG2IMG_MODELS), value="FLUX.1-schnell", label="模型")
                        st_strength = gr.Slider(0.1, 1.0, 0.65, step=0.05, label="转换强度")
                        st_btn = gr.Button("转换", variant="primary")
                    with gr.Column():
                        st_out = gr.Image(label="结果", type="pil")
                st_btn.click(style_transfer, [st_img, st_prompt, st_model, st_strength], st_out)

            # ── 5. 图像编辑 (Inpainting) ──
            with gr.Tab("图像编辑"):
                gr.Markdown("上传图片，用画笔涂抹要修改的区域，输入描述后生成")
                with gr.Row():
                    with gr.Column():
                        ip_editor = gr.ImageEditor(
                            label="涂抹要编辑的区域（白色画笔）",
                            type="pil",
                            brush=gr.Brush(default_size=30, colors=["#FFFFFF"]),
                        )
                        ip_prompt = gr.Textbox(label="编辑描述", lines=2,
                            placeholder="a beautiful garden with flowers...")
                        ip_model = gr.Dropdown(list(INPAINT_MODELS), value="SD XL Inpainting", label="模型")
                        ip_strength = gr.Slider(0.3, 1.0, 0.75, step=0.05, label="编辑强度")
                        ip_btn = gr.Button("编辑", variant="primary")
                    with gr.Column():
                        ip_out = gr.Image(label="结果", type="pil")
                ip_btn.click(inpaint, [ip_editor, ip_prompt, ip_model, ip_strength], ip_out)

            # ── 6. 超分辨率 ──
            with gr.Tab("超分辨率"):
                gr.Markdown("上传低分辨率图片，AI 增强放大")
                with gr.Row():
                    with gr.Column():
                        sr_img = gr.Image(label="上传图片", type="pil")
                        sr_model = gr.Dropdown(list(SR_MODELS), value="SwinSR x4 (真实感)", label="模型")
                        sr_btn = gr.Button("增强", variant="primary")
                    with gr.Column():
                        sr_out = gr.Image(label="增强结果", type="pil")
                        sr_info = gr.Textbox(label="分辨率信息")
                sr_btn.click(super_resolution, [sr_img, sr_model], [sr_out, sr_info])

            # ── 7. 背景移除 ──
            with gr.Tab("背景移除"):
                gr.Markdown("一键移除图片背景")
                with gr.Row():
                    with gr.Column():
                        bg_img = gr.Image(label="上传图片", type="pil")
                        bg_model = gr.Dropdown(list(RMBG_MODELS), value="RMBG-2.0 (推荐)", label="模型")
                        bg_btn = gr.Button("移除背景", variant="primary")
                    with gr.Column():
                        bg_out = gr.Image(label="结果", type="pil")
                bg_btn.click(remove_background, [bg_img, bg_model], bg_out)

    # ┌─────────────────────────────────────────────────────────
    # │ 理解
    # └─────────────────────────────────────────────────────────
    with gr.Tab("理解"):
        with gr.Tabs():

            # ── 8. 图像描述 ──
            with gr.Tab("图像描述"):
                with gr.Row():
                    with gr.Column():
                        cap_img = gr.Image(label="上传图片", type="pil")
                        cap_model = gr.Dropdown(list(CAPTION_MODELS), value="GIT-Large-COCO", label="模型")
                        cap_btn = gr.Button("描述", variant="primary")
                    with gr.Column():
                        cap_out = gr.Textbox(label="描述结果", lines=5)
                cap_btn.click(image_caption, [cap_img, cap_model], cap_out)

            # ── 9. 视觉问答 ──
            with gr.Tab("视觉问答"):
                gr.Markdown("上传图片并提问，AI 回答关于图片内容的问题")
                with gr.Row():
                    with gr.Column():
                        vqa_img = gr.Image(label="上传图片", type="pil")
                        vqa_q = gr.Textbox(label="问题", placeholder="What color is the cat?")
                        vqa_model = gr.Dropdown(list(VQA_MODELS), value="ViLT VQA", label="模型")
                        vqa_btn = gr.Button("回答", variant="primary")
                    with gr.Column():
                        vqa_out = gr.Textbox(label="回答", lines=5)
                vqa_btn.click(visual_qa, [vqa_img, vqa_q, vqa_model], vqa_out)

            # ── 10. OCR 识别 ──
            with gr.Tab("OCR 识别"):
                gr.Markdown("从图片中提取文字（支持印刷体和手写体）")
                with gr.Row():
                    with gr.Column():
                        ocr_img = gr.Image(label="上传图片", type="pil")
                        ocr_model = gr.Dropdown(list(OCR_MODELS), value="TrOCR Large (印刷体)", label="模型")
                        ocr_btn = gr.Button("识别", variant="primary")
                    with gr.Column():
                        ocr_out = gr.Textbox(label="识别结果", lines=8)
                ocr_btn.click(ocr_recognize, [ocr_img, ocr_model], ocr_out)

            # ── 11. 图像分割 ──
            with gr.Tab("图像分割"):
                gr.Markdown("对图片进行语义分割，标注不同区域")
                with gr.Row():
                    with gr.Column():
                        seg_img = gr.Image(label="上传图片", type="pil")
                        seg_model = gr.Dropdown(list(SEG_MODELS), value="SegFormer (场景分割)", label="模型")
                        seg_btn = gr.Button("分割", variant="primary")
                    with gr.Column():
                        seg_out = gr.Image(label="分割结果", type="pil")
                        seg_labels = gr.Textbox(label="检测到的区域", lines=6)
                seg_btn.click(image_segmentation, [seg_img, seg_model], [seg_out, seg_labels])

    # ┌─────────────────────────────────────────────────────────
    # │ 智能
    # └─────────────────────────────────────────────────────────
    with gr.Tab("智能"):
        with gr.Tabs():

            # ── 12. AI Agent ──
            with gr.Tab("AI Agent"):
                status = "已就绪" if _agent else "不可用（需安装 smolagents）"
                gr.Markdown(
                    f"用自然语言指挥 AI 完成复杂图像任务链 — 状态: **{status}**\n\n"
                    "示例: \"生成一张日落风景图，然后去掉背景\""
                )
                with gr.Row():
                    with gr.Column():
                        ag_input = gr.Textbox(label="指令", lines=3,
                            placeholder="生成一张宇宙飞船的图片，然后提高分辨率")
                        ag_btn = gr.Button("执行", variant="primary")
                    with gr.Column():
                        ag_text = gr.Textbox(label="执行日志", lines=6)
                        ag_img = gr.Image(label="结果图片", type="pil")
                ag_btn.click(run_agent, ag_input, [ag_text, ag_img])

            # ── 13. 批量处理 ──
            with gr.Tab("批量处理"):
                gr.Markdown("批量上传图片，统一执行同一操作（最多 10 张）")
                with gr.Row():
                    with gr.Column():
                        bat_files = gr.File(label="上传图片", file_count="multiple",
                            file_types=["image"])
                        bat_op = gr.Dropdown(BATCH_OPS, value="背景移除", label="操作")
                        bat_btn = gr.Button("批量执行", variant="primary")
                    with gr.Column():
                        bat_gallery = gr.Gallery(label="处理结果", columns=3)
                bat_btn.click(batch_process, [bat_files, bat_op], bat_gallery)

    # ┌─────────────────────────────────────────────────────────
    # │ 标注工具
    # └─────────────────────────────────────────────────────────
    with gr.Tab("标注工具"):
        gr.Markdown(
            """
            ## AI 预标注引擎（快速入口）
            输入目标类别，一键完成零样本目标检测 + 图像描述预标注。
            完整功能（批量标注、COCO JSON / YOLO TXT 导出、VQA 质检）请运行独立的 **annotator_app.py**：
            ```bash
            python annotator_app.py
            ```
            """
        )
        with gr.Row():
            with gr.Column():
                ann_img = gr.Image(label="上传图片", type="pil")
                ann_prompt = gr.Textbox(
                    label="目标类别（逗号分隔）",
                    placeholder="cat, dog, person",
                    lines=1,
                )
                ann_btn = gr.Button("▶ 预标注", variant="primary")
            with gr.Column():
                ann_vis = gr.Image(label="预标注结果", type="pil")
                ann_status = gr.Textbox(label="标注摘要", lines=4)
        ann_btn.click(quick_annotate, [ann_img, ann_prompt], [ann_vis, ann_status])

    # ── 底部 ──
    gr.Markdown(
        """
        ---
        **HF Image Studio** | 所有推理通过 [HF Inference API](https://huggingface.co/docs/api-inference) 完成
        | [获取 Token](https://huggingface.co/settings/tokens)
        | [GitHub](https://github.com/huggingface)
        """
    )

# ── 启动 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
