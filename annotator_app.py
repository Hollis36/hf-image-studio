"""
HF Auto Annotator — AI 预标注引擎
零配置，通过 HF Inference API 一键完成目标检测、语义分割、图像描述预标注
导出标准 COCO JSON / YOLO TXT 格式，可直接导入 Label Studio / CVAT
"""

import os
import io
import json
import csv
import zipfile
import tempfile
from datetime import datetime

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import InferenceClient

# ── 客户端初始化 ──────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(token=HF_TOKEN)

# ── 模型配置 ──────────────────────────────────────────────────
DETECTION_MODEL = "google/owlv2-base-patch16-ensemble"

SEG_MODELS = {
    "SegFormer (场景分割)": "nvidia/segformer-b5-finetuned-ade-640-640",
    "DETR (全景分割)": "facebook/detr-resnet-50-panoptic",
}

CAPTION_MODELS = {
    "GIT-Large-COCO": "microsoft/git-large-coco",
    "BLIP-Large": "Salesforce/blip-image-captioning-large",
}

VQA_MODEL = "dandelin/vilt-b32-finetuned-vqa"

COLORS = [
    "#FF4444", "#44FF44", "#4444FF", "#FFFF44", "#FF44FF", "#44FFFF",
    "#FF8844", "#44FF88", "#8844FF", "#FF4488", "#88FF44", "#4488FF",
]

BATCH_LIMIT = 20


# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def _hex_to_rgba(hex_color: str, alpha: int = 120) -> tuple:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (r, g, b, alpha)


def _mask_to_bbox_polygon(mask) -> tuple:
    """
    从二值掩码提取边界框和简化多边形（矩形轮廓）。
    返回 (polygon, area, bbox)，其中 polygon 为 [x1,y1,x2,y2,...] 格式。
    """
    if isinstance(mask, Image.Image):
        mask_array = np.array(mask.convert("L"))
    else:
        mask_array = np.asarray(mask)

    rows = np.any(mask_array > 127, axis=1)
    cols = np.any(mask_array > 127, axis=0)
    if not np.any(rows):
        return None, 0, [0, 0, 0, 0]

    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

    # 矩形多边形（顺时针 4 个顶点）
    polygon = [cmin, rmin, cmax, rmin, cmax, rmax, cmin, rmax]
    area = int((cmax - cmin) * (rmax - rmin))
    bbox = [cmin, rmin, cmax - cmin, rmax - rmin]
    return polygon, area, bbox


# ══════════════════════════════════════════════════════════════
#  核心推理函数
# ══════════════════════════════════════════════════════════════

def run_detection(image: Image.Image, text_prompt: str) -> list:
    """零样本目标检测（OWLv2）"""
    labels = [lb.strip() for lb in text_prompt.split(",") if lb.strip()]
    if not labels:
        return []
    results = client.zero_shot_object_detection(
        image=image,
        candidate_labels=labels,
        model=DETECTION_MODEL,
    )
    return results if results else []


def run_segmentation(image: Image.Image, seg_model_key: str) -> list:
    """语义分割（SegFormer / DETR）"""
    segments = client.image_segmentation(
        image=image, model=SEG_MODELS[seg_model_key]
    )
    return segments if segments else []


def run_caption(image: Image.Image, caption_model_key: str) -> str:
    """图像描述（GIT-Large / BLIP）"""
    result = client.image_to_text(
        image=image, model=CAPTION_MODELS[caption_model_key]
    )
    if isinstance(result, list):
        return result[0].get("generated_text", str(result)) if result else ""
    if isinstance(result, dict):
        return result.get("generated_text", str(result))
    return str(result)


def run_vqa_check(image: Image.Image, label: str) -> tuple:
    """
    用 VQA 验证检测标签是否存在于图片中。
    返回 (answer, confidence_score)。
    """
    question = f"Is there a {label} in this image?"
    results = client.visual_question_answering(
        image=image, question=question, model=VQA_MODEL
    )
    if not results:
        return "unknown", 0.0
    if isinstance(results, list):
        for item in results:
            ans = item.get("answer", "").lower()
            if ans in ("yes", "no"):
                return ans, float(item.get("score", 0.0))
        top = results[0]
        return top.get("answer", "unknown"), float(top.get("score", 0.0))
    return "unknown", 0.0


# ══════════════════════════════════════════════════════════════
#  可视化
# ══════════════════════════════════════════════════════════════

def visualize_annotations(image: Image.Image, detections: list, segments: list) -> Image.Image:
    """在图片上绘制检测框和分割掩码"""
    vis = image.copy().convert("RGBA")

    # 分割掩码叠加
    for i, seg in enumerate(segments):
        mask = seg.get("_mask")
        if mask is None:
            continue
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        mask_l = mask.convert("L")
        color_rgba = _hex_to_rgba(COLORS[i % len(COLORS)], alpha=100)
        color_layer = Image.new("RGBA", vis.size, color_rgba)
        vis = Image.composite(color_layer, vis, mask_l)

    # 检测框叠加
    overlay = Image.new("RGBA", vis.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i, det in enumerate(detections):
        box = det.get("box", {})
        xmin = box.get("xmin", 0)
        ymin = box.get("ymin", 0)
        xmax = box.get("xmax", 0)
        ymax = box.get("ymax", 0)
        label = det.get("label", "")
        score = det.get("score", 0)
        color = COLORS[i % len(COLORS)]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

        draw.rectangle([xmin, ymin, xmax, ymax], outline=(r, g, b, 220), width=2)
        text = f"{label} {score:.2f}"
        text_x = max(xmin, 2)
        text_y = max(ymin - 14, 2)
        draw.rectangle(
            [text_x, text_y, text_x + len(text) * 6 + 4, text_y + 14],
            fill=(r, g, b, 180),
        )
        draw.text((text_x + 2, text_y + 1), text, fill=(255, 255, 255, 255))

    vis = Image.alpha_composite(vis, overlay)
    return vis.convert("RGB")


# ══════════════════════════════════════════════════════════════
#  核心预标注函数
# ══════════════════════════════════════════════════════════════

def auto_annotate(image: Image.Image, text_prompt: str, tasks: list,
                  seg_model: str, caption_model: str) -> tuple:
    """
    对单张图片执行多任务预标注。

    Returns:
        (vis_image, annotations_dict, status_text)
    """
    annotations = {
        "detections": [],
        "segments": [],
        "caption": "",
        "vqa_checks": [],
    }
    status_lines = []

    # ── 检测 ──
    if "检测" in tasks or "全部" in tasks:
        if text_prompt.strip():
            try:
                dets = run_detection(image, text_prompt)
                annotations["detections"] = dets
                status_lines.append(f"✅ 目标检测：检测到 {len(dets)} 个目标")
            except Exception as exc:
                status_lines.append(f"❌ 目标检测失败：{exc}")
        else:
            status_lines.append("⚠️ 目标检测：请输入目标类别文本")

    # ── 分割 ──
    if "分割" in tasks or "全部" in tasks:
        try:
            segs = run_segmentation(image, seg_model)
            annotations["segments"] = [
                {
                    "label": s.get("label", ""),
                    "score": float(s.get("score", 0)),
                    "_mask": s.get("mask"),
                }
                for s in segs
            ]
            status_lines.append(f"✅ 语义分割：分割 {len(segs)} 个区域")
        except Exception as exc:
            status_lines.append(f"❌ 语义分割失败：{exc}")

    # ── 描述 ──
    if "描述" in tasks or "全部" in tasks:
        try:
            cap = run_caption(image, caption_model)
            annotations["caption"] = cap
            preview = cap[:60] + ("..." if len(cap) > 60 else "")
            status_lines.append(f"✅ 图像描述：{preview}")
        except Exception as exc:
            status_lines.append(f"❌ 图像描述失败：{exc}")

    # ── VQA 自动质检（针对检测到的标签） ──
    detected_labels = list({d.get("label", "") for d in annotations["detections"] if d.get("label")})
    for label in detected_labels[:5]:
        try:
            answer, score = run_vqa_check(image, label)
            annotations["vqa_checks"].append({
                "label": label,
                "answer": answer,
                "score": score,
                "low_confidence": score < 0.5,
            })
        except Exception:
            pass

    vis = visualize_annotations(image, annotations["detections"], annotations["segments"])
    return vis, annotations, "\n".join(status_lines)


# ══════════════════════════════════════════════════════════════
#  导出函数
# ══════════════════════════════════════════════════════════════

def export_coco_json(images_data: list) -> str:
    """生成标准 COCO JSON 格式"""
    coco = {
        "info": {
            "description": "HF Auto Annotator Export",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    category_map: dict[str, int] = {}
    cat_id = 1
    ann_id = 1

    for img_id, (img_name, img_size, annotations) in enumerate(images_data, 1):
        w, h = img_size
        coco["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": w,
            "height": h,
        })

        # 检测框注释
        for det in annotations.get("detections", []):
            label = det.get("label", "object")
            score = float(det.get("score", 0))
            box = det.get("box", {})
            xmin = int(box.get("xmin", 0))
            ymin = int(box.get("ymin", 0))
            xmax = int(box.get("xmax", 0))
            ymax = int(box.get("ymax", 0))
            bw = xmax - xmin
            bh = ymax - ymin

            if label not in category_map:
                category_map[label] = cat_id
                coco["categories"].append({
                    "id": cat_id,
                    "name": label,
                    "supercategory": "object",
                })
                cat_id += 1

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_map[label],
                "bbox": [xmin, ymin, bw, bh],
                "area": bw * bh,
                "segmentation": [],
                "score": score,
                "iscrowd": 0,
            })
            ann_id += 1

        # 分割注释
        for seg in annotations.get("segments", []):
            label = seg.get("label", "region")
            score = float(seg.get("score", 0))
            mask = seg.get("_mask")

            if label not in category_map:
                category_map[label] = cat_id
                coco["categories"].append({
                    "id": cat_id,
                    "name": label,
                    "supercategory": "object",
                })
                cat_id += 1

            polygon, area, bbox = None, 0, [0, 0, 0, 0]
            if mask is not None:
                polygon, area, bbox = _mask_to_bbox_polygon(mask)

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_map[label],
                "bbox": bbox,
                "area": area,
                "segmentation": [polygon] if polygon else [],
                "score": score,
                "iscrowd": 0,
            })
            ann_id += 1

    return json.dumps(coco, ensure_ascii=False, indent=2)


def export_yolo_txt(images_data: list) -> dict:
    """生成 YOLO TXT 格式，坐标归一化到 [0,1]"""
    files: dict[str, str] = {}
    category_map: dict[str, int] = {}
    cat_id = 0

    for img_name, img_size, annotations in images_data:
        w, h = img_size
        lines = []

        for det in annotations.get("detections", []):
            label = det.get("label", "object")
            score = float(det.get("score", 0))
            box = det.get("box", {})
            xmin = float(box.get("xmin", 0))
            ymin = float(box.get("ymin", 0))
            xmax = float(box.get("xmax", 0))
            ymax = float(box.get("ymax", 0))

            if label not in category_map:
                category_map[label] = cat_id
                cat_id += 1

            cx = ((xmin + xmax) / 2) / w
            cy = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            lines.append(
                f"{category_map[label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {score:.6f}"
            )

        base_name = os.path.splitext(img_name)[0]
        files[f"{base_name}.txt"] = "\n".join(lines)

    # classes.txt
    classes = sorted(category_map.keys(), key=lambda x: category_map[x])
    files["classes.txt"] = "\n".join(classes)
    return files


def export_csv(images_data: list) -> str:
    """生成 CSV 格式"""
    headers = [
        "image_name", "label",
        "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
        "confidence", "caption",
    ]
    rows = []

    for img_name, _img_size, annotations in images_data:
        caption = annotations.get("caption", "")
        detections = annotations.get("detections", [])

        if detections:
            for det in detections:
                label = det.get("label", "")
                score = float(det.get("score", 0))
                box = det.get("box", {})
                rows.append([
                    img_name, label,
                    box.get("xmin", ""), box.get("ymin", ""),
                    box.get("xmax", ""), box.get("ymax", ""),
                    f"{score:.4f}", caption,
                ])
        else:
            rows.append([img_name, "", "", "", "", "", "", caption])

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    writer.writerows(rows)
    return buf.getvalue()


def _build_zip(export_format: str, images_data: list) -> str:
    """将导出内容打包为 ZIP，返回临时文件路径"""
    tmp_dir = tempfile.gettempdir()

    if export_format == "COCO JSON":
        content = export_coco_json(images_data)
        zip_path = os.path.join(tmp_dir, "annotations_coco.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("annotations.json", content)

    elif export_format == "YOLO TXT":
        files_dict = export_yolo_txt(images_data)
        zip_path = os.path.join(tmp_dir, "annotations_yolo.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname, content in files_dict.items():
                zf.writestr(fname, content)

    else:  # CSV
        content = export_csv(images_data)
        zip_path = os.path.join(tmp_dir, "annotations_csv.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("annotations.csv", content)

    return zip_path


# ══════════════════════════════════════════════════════════════
#  Gradio 回调函数
# ══════════════════════════════════════════════════════════════

def single_annotate(image, text_prompt, tasks, seg_model, caption_model, export_fmt):
    """Tab 1 回调：单图预标注"""
    if image is None:
        raise gr.Error("请上传图片")
    if not tasks:
        raise gr.Error("请至少选择一个任务")

    vis, annotations, status = auto_annotate(image, text_prompt, tasks, seg_model, caption_model)

    # 构建标注详情文本
    detail_lines = []
    if annotations["detections"]:
        detail_lines.append("【目标检测结果】")
        for det in annotations["detections"]:
            box = det.get("box", {})
            detail_lines.append(
                f"  • {det.get('label')}  置信度 {det.get('score', 0):.3f}  "
                f"bbox [{box.get('xmin')}, {box.get('ymin')}, {box.get('xmax')}, {box.get('ymax')}]"
            )
    if annotations["segments"]:
        detail_lines.append("【分割结果】")
        for seg in annotations["segments"]:
            detail_lines.append(f"  • {seg.get('label')}  置信度 {seg.get('score', 0):.3f}")
    if annotations["caption"]:
        detail_lines.append(f"【图像描述】\n  {annotations['caption']}")
    if annotations["vqa_checks"]:
        detail_lines.append("【VQA 质检】")
        for chk in annotations["vqa_checks"]:
            flag = "⚠️ 低置信" if chk["low_confidence"] else "✅"
            detail_lines.append(
                f"  {flag} {chk['label']}：{chk['answer']}  置信度 {chk['score']:.3f}"
            )

    detail_text = "\n".join(detail_lines) if detail_lines else "无标注结果"

    # 导出单图
    images_data = [("image.jpg", image.size, annotations)]
    zip_path = _build_zip(export_fmt, images_data)

    return vis, status + "\n\n" + detail_text, zip_path


def batch_run(files, text_prompt, tasks, export_format, seg_model, caption_model, progress=gr.Progress()):
    """Tab 2 回调：批量预标注"""
    if not files:
        raise gr.Error("请上传图片")
    if len(files) > BATCH_LIMIT:
        raise gr.Error(f"最多支持 {BATCH_LIMIT} 张图片（免费 API 限制）")
    if not tasks:
        raise gr.Error("请至少选择一个任务")

    images_data = []
    gallery = []

    for idx, f in enumerate(files):
        progress((idx + 1) / len(files), desc=f"正在处理 {idx + 1}/{len(files)}…")
        try:
            img = Image.open(f.name if hasattr(f, "name") else f)
            img_name = os.path.basename(f.name if hasattr(f, "name") else str(f))
            vis, annotations, _status = auto_annotate(img, text_prompt, tasks, seg_model, caption_model)
            images_data.append((img_name, img.size, annotations))
            gallery.append((vis, img_name))
        except Exception as exc:
            img = Image.open(f.name if hasattr(f, "name") else f)
            img_name = os.path.basename(f.name if hasattr(f, "name") else str(f))
            gallery.append((img, f"❌ {img_name}: {exc}"))
            images_data.append((img_name, img.size, {
                "detections": [], "segments": [], "caption": "", "vqa_checks": []
            }))

    zip_path = _build_zip(export_format, images_data)
    summary = f"共处理 {len(files)} 张图片，导出格式：{export_format}"
    return gallery, summary, zip_path


def quality_check_run(image, annotation_text):
    """Tab 3 回调：VQA 标注质检"""
    if image is None:
        raise gr.Error("请上传图片")
    if not annotation_text.strip():
        raise gr.Error("请粘贴标注 JSON 或输入标签（逗号分隔）")

    # 尝试解析 COCO JSON
    labels = []
    try:
        data = json.loads(annotation_text)
        # 支持 COCO 格式 annotations + categories
        cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}
        for ann in data.get("annotations", []):
            cat_name = cat_map.get(ann.get("category_id"), "")
            if cat_name and cat_name not in labels:
                labels.append(cat_name)
    except (json.JSONDecodeError, AttributeError):
        # 降级：直接解析逗号分隔的标签
        labels = [lb.strip() for lb in annotation_text.split(",") if lb.strip()]

    if not labels:
        raise gr.Error("未能解析到任何标签，请检查输入格式")

    report_lines = ["# VQA 标注质检报告", f"检测标签数：{len(labels)}", ""]
    low_confidence_count = 0

    for label in labels:
        try:
            answer, score = run_vqa_check(image, label)
            is_low = score < 0.5
            if is_low:
                low_confidence_count += 1
            flag = "⚠️ 建议人工复核" if is_low else "✅ 通过"
            report_lines.append(
                f"• **{label}**  答案：{answer}  置信度：{score:.3f}  {flag}"
            )
        except Exception as exc:
            report_lines.append(f"• **{label}**  检查失败：{exc}")

    report_lines.append("")
    if low_confidence_count:
        report_lines.append(
            f"⚠️ 共 {low_confidence_count} 个标签置信度 < 0.5，建议人工重点审核"
        )
    else:
        report_lines.append("✅ 所有标签质检通过")

    return "\n".join(report_lines)


# ══════════════════════════════════════════════════════════════
#  Gradio UI
# ══════════════════════════════════════════════════════════════

with gr.Blocks(title="HF Auto Annotator") as annotator_demo:
    gr.Markdown(
        """
        # HF Auto Annotator — AI 预标注引擎
        **零配置预标注** | 基于 HF Inference API | 导出 COCO JSON / YOLO TXT / CSV
        可直接导入 Label Studio / CVAT / Roboflow 进行人工校正
        """
    )

    # ── Tab 1: 单图预标注 ──────────────────────────────────────
    with gr.Tab("单图预标注"):
        gr.Markdown("上传图片，输入目标类别，自动完成检测 / 分割 / 描述预标注。")
        with gr.Row():
            with gr.Column(scale=1):
                s_image = gr.Image(label="上传图片", type="pil")
                s_prompt = gr.Textbox(
                    label="目标类别（逗号分隔）",
                    placeholder="cat, dog, person",
                    lines=1,
                )
                s_tasks = gr.CheckboxGroup(
                    choices=["检测", "分割", "描述", "全部"],
                    value=["全部"],
                    label="执行任务",
                )
                with gr.Row():
                    s_seg_model = gr.Dropdown(
                        list(SEG_MODELS), value="SegFormer (场景分割)", label="分割模型"
                    )
                    s_cap_model = gr.Dropdown(
                        list(CAPTION_MODELS), value="GIT-Large-COCO", label="描述模型"
                    )
                s_export_fmt = gr.Radio(
                    ["COCO JSON", "YOLO TXT", "CSV"],
                    value="COCO JSON",
                    label="导出格式",
                )
                s_btn = gr.Button("▶ 运行预标注", variant="primary")

            with gr.Column(scale=1):
                s_vis = gr.Image(label="预标注可视化", type="pil")
                s_status = gr.Textbox(label="标注详情", lines=12)
                s_download = gr.File(label="下载标注文件")

        s_btn.click(
            single_annotate,
            inputs=[s_image, s_prompt, s_tasks, s_seg_model, s_cap_model, s_export_fmt],
            outputs=[s_vis, s_status, s_download],
        )

    # ── Tab 2: 批量预标注 ──────────────────────────────────────
    with gr.Tab("批量预标注"):
        gr.Markdown(f"批量上传图片（最多 {BATCH_LIMIT} 张），一键完成预标注并下载 ZIP 包。")
        with gr.Row():
            with gr.Column(scale=1):
                b_files = gr.File(
                    label="上传图片（最多 20 张）",
                    file_count="multiple",
                    file_types=["image"],
                )
                b_prompt = gr.Textbox(
                    label="目标类别（逗号分隔）",
                    placeholder="cat, dog, person",
                    lines=1,
                )
                b_tasks = gr.CheckboxGroup(
                    choices=["检测", "分割", "描述", "全部"],
                    value=["全部"],
                    label="执行任务",
                )
                with gr.Row():
                    b_seg_model = gr.Dropdown(
                        list(SEG_MODELS), value="SegFormer (场景分割)", label="分割模型"
                    )
                    b_cap_model = gr.Dropdown(
                        list(CAPTION_MODELS), value="GIT-Large-COCO", label="描述模型"
                    )
                b_export_fmt = gr.Radio(
                    ["COCO JSON", "YOLO TXT", "CSV"],
                    value="COCO JSON",
                    label="导出格式",
                )
                b_btn = gr.Button("▶ 一键批量标注", variant="primary")

            with gr.Column(scale=1):
                b_gallery = gr.Gallery(label="预标注预览", columns=3)
                b_summary = gr.Textbox(label="处理摘要", lines=3)
                b_download = gr.File(label="下载标注 ZIP 包")

        b_btn.click(
            batch_run,
            inputs=[b_files, b_prompt, b_tasks, b_export_fmt, b_seg_model, b_cap_model],
            outputs=[b_gallery, b_summary, b_download],
        )

    # ── Tab 3: 标注质检 ────────────────────────────────────────
    with gr.Tab("标注质检"):
        gr.Markdown(
            "上传图片并粘贴已有标注（COCO JSON 或逗号分隔标签），"
            "使用 VQA 自动验证标注正确性，标记低置信度结果供人工审核。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                q_image = gr.Image(label="上传图片", type="pil")
                q_ann = gr.Textbox(
                    label="标注内容（粘贴 COCO JSON 或输入标签，如 cat, dog）",
                    lines=8,
                    placeholder='{"categories": [{"id": 1, "name": "cat"}], "annotations": [{"category_id": 1}]}\n\n或直接输入：cat, dog, person',
                )
                q_btn = gr.Button("▶ 运行质检", variant="primary")
            with gr.Column(scale=1):
                q_report = gr.Textbox(label="质检报告", lines=18)

        q_btn.click(
            quality_check_run,
            inputs=[q_image, q_ann],
            outputs=[q_report],
        )

    # ── 底部 ──────────────────────────────────────────────────
    gr.Markdown(
        """
        ---
        **HF Auto Annotator** | 基于 [HF Inference API](https://huggingface.co/docs/api-inference)
        | 零样本检测：OWLv2 | 分割：SegFormer / DETR | 描述：GIT-Large / BLIP | 质检：ViLT VQA
        | [获取 HF Token](https://huggingface.co/settings/tokens)
        """
    )


if __name__ == "__main__":
    annotator_demo.launch(theme=gr.themes.Soft())
