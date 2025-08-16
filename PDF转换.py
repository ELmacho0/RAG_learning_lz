import os
import re
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

# ========== 基本参数 ==========
INPUT_DIR = r"D:\智能知识库\PDF原始数据"  # 批量PDF输入目录
OUTPUT_DIR = r"D:\智能知识库\PDF转换输出"  # 输出目录（每个PDF创建同名子目录）
POPPLER_BIN = r""  # 例如 r"C:\USER_PROGRAM\poppler-24.08.0\Library\bin"；若已配PATH可留空
DPI = 300  # 页面渲染DPI（线条细时可升到400）
DEBUG_SAVE_INTERMEDIATE = 0  # 是否保存二值图/线图/标注图（便于调参）

# ========== 表格检测参数（针对竖版A4默认值） ==========
BIN_METHOD = "adaptive"  # "adaptive" 或 "otsu"
ADAPTIVE_BLOCK_SIZE = 35  # 必须奇数；越大越“粗犷”
ADAPTIVE_C = 15  # 偏移量；大→更黑
GAUSS_BLUR = 3  # 预模糊核（0或1=不模糊）

VERT_KERNEL_RATIO = 0.018  # 竖线核比例（相对宽度）
HORZ_KERNEL_RATIO = 0.015  # 横线核比例（相对高度）
MORPH_ITER = 2  # 形态学迭代次数（线断就加大）

MIN_TABLE_AREA_RATIO = 0.01  # 最小表格面积占比（过多小框→调大）
MIN_W_RATIO = 0.12  # 表格最小宽度占比
MIN_H_RATIO = 0.06  # 表格最小高度占比
MERGE_IOU_THRESHOLD = 0.15  # 近邻合并 IoU 阈值
MERGE_PIX_GAP = 12  # 近邻合并像素间隙

# ========== 初始化 OCR ==========
ocr_engine = RapidOCR()


# ---------- 工具函数 ----------
def safe_imwrite(path: str, bgr_image) -> bool:
    """安全写图：优先 imencode，失败用 PIL 兜底，兼容中文路径"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower() or ".png"
    try:
        ok, buf = cv2.imencode(ext if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"] else ".png", bgr_image)
        if ok:
            with open(path, "wb") as f:
                f.write(buf.tobytes())
            return True
    except Exception:
        pass
    try:
        if bgr_image.ndim == 3 and bgr_image.shape[2] == 3:
            from PIL import Image
            Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)).save(path)
        else:
            from PIL import Image
            Image.fromarray(bgr_image).save(path)
        return True
    except Exception:
        return False


def pil_to_bgr(img_pil: Image.Image):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def binarize(gray):
    if GAUSS_BLUR and GAUSS_BLUR > 1:
        gray = cv2.GaussianBlur(gray, (GAUSS_BLUR, GAUSS_BLUR), 0)
    if BIN_METHOD == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
            ADAPTIVE_BLOCK_SIZE | 1, ADAPTIVE_C
        )
    else:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return th


def extract_lines(bin_img):
    h, w = bin_img.shape[:2]
    vk = max(1, int(w * VERT_KERNEL_RATIO))
    hk = max(1, int(h * HORZ_KERNEL_RATIO))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    vertical = cv2.erode(bin_img, vertical_kernel, iterations=MORPH_ITER)
    vertical = cv2.dilate(vertical, vertical_kernel, iterations=MORPH_ITER)
    horizontal = cv2.erode(bin_img, horizontal_kernel, iterations=MORPH_ITER)
    horizontal = cv2.dilate(horizontal, horizontal_kernel, iterations=MORPH_ITER)
    table_lines = cv2.bitwise_or(vertical, horizontal)
    table_lines = cv2.dilate(table_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return table_lines



def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0: return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / float(area_a + area_b - inter_area + 1e-6)


def merge_boxes(boxes, gap=MERGE_PIX_GAP, iou_th=MERGE_IOU_THRESHOLD):
    changed = True
    boxes = boxes[:]
    while changed:
        changed = False
        new = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]: continue
            ax1, ay1, ax2, ay2 = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]: continue
                bx1, by1, bx2, by2 = boxes[j]
                near = (abs(ax1 - bx1) <= gap or abs(ax2 - bx2) <= gap or
                        abs(ay1 - by1) <= gap or abs(ay2 - by2) <= gap)
                if near or iou((ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2)) >= iou_th:
                    ax1, ay1 = min(ax1, bx1), min(ay1, by1)
                    ax2, ay2 = max(ax2, bx2), max(ay2, by2)
                    used[j] = True
                    changed = True
            used[i] = True
            new.append((ax1, ay1, ax2, ay2))
        boxes = new
    return boxes


def detect_tables_on_bgr(bgr):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bin_img = binarize(gray)
    table_lines = extract_lines(bin_img)
    contours, _ = cv2.findContours(table_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    min_area = int(MIN_TABLE_AREA_RATIO * w * h)
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww * hh < min_area: continue
        if ww < int(MIN_W_RATIO * w) or hh < int(MIN_H_RATIO * h): continue
        boxes.append((x, y, x + ww, y + hh))
    boxes = merge_boxes(boxes)
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes, table_lines, bin_img


def mask_tables_on_bgr(bgr, boxes, margin_px=2):
    """将表格区域涂白，避免 OCR 进去；margin_px 给一点边界裕量"""
    h, w = bgr.shape[:2]
    masked = bgr.copy()
    for (x1, y1, x2, y2) in boxes:
        x1 = max(0, x1 - margin_px);
        y1 = max(0, y1 - margin_px)
        x2 = min(w, x2 + margin_px);
        y2 = min(h, y2 + margin_px)
        cv2.rectangle(masked, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
    return masked


def clean_and_merge_lines(text):
    """按中文标点断句，去掉PDF硬换行；遇到 。，！？ 才换行"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    joined = " ".join(lines)  # 先把硬换行变空格
    # 用标点 + 空格 进行分割，再拼回去
    # 保留标点本身（用捕获组）
    parts = re.split(r"([。！？])", joined)
    rebuilt = []
    buf = ""
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        if not seg and not punct:
            continue
        buf = (seg + punct).strip()
        if punct:
            rebuilt.append(buf)
            buf = ""
    if buf:
        rebuilt.append(buf)
    return rebuilt


def ocr_image_pil(image_pil: Image.Image) -> str:
    """对整页（已遮盖表格）做 OCR"""
    ocr_result, _ = ocr_engine(image_pil)
    if not ocr_result:
        return ""
    return "".join([line[1] for line in ocr_result])


def pdf_page_to_image(pdf_path, page_num=1, dpi=DPI, poppler_bin=POPPLER_BIN) -> Image.Image:
    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=page_num,
        last_page=page_num,
        poppler_path=poppler_bin if poppler_bin else None
    )
    return images[0]  # PIL.Image


# ---------- 主流程 ----------
def process_pdf(pdf_path: str, output_root: str):
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = os.path.join(output_root, base)
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"{base}.txt")
    print(f"[START] {pdf_path} -> {out_dir}")

    # 获取页数（用 pdf2image 一页页渲染）
    # 更高效的方式可以先用 pdfinfo，但这里直接尝试逐页直到失败不太优雅；
    # 简单做法：先渲染一次拿到总页数
    try:
        pages = convert_from_path(pdf_path, dpi=50, poppler_path=POPPLER_BIN if POPPLER_BIN else None)
        total_pages = len(pages)
    except Exception:
        # 回退：大多数PDF都能渲染，这里保底当做1页
        total_pages = 1

    with open(txt_path, "w", encoding="utf-8") as fout:
        for page_num in range(1, total_pages + 1):
            pil_img = pdf_page_to_image(pdf_path, page_num=page_num)
            bgr = pil_to_bgr(pil_img)

            # 1) 检测表格
            boxes, lines_img, bin_img = detect_tables_on_bgr(bgr)

            # 2) 保存表格截图（只表格，不保存普通文字）
            for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
                crop = bgr[y1:y2, x1:x2]
                crop_path = os.path.join(out_dir, f"page_{page_num}_table_{i}.png")
                ok = safe_imwrite(crop_path, crop)
                print(f"[{'INFO' if ok else 'ERROR'}] 表格图: {crop_path} ({x1},{y1},{x2},{y2})")

            # 3) 遮蔽表格区域（白色），避免 OCR 到表格内文字
            masked_bgr = mask_tables_on_bgr(bgr, boxes, margin_px=2)
            masked_pil = Image.fromarray(cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB))

            # 4) OCR 非表格区域
            text_raw = ocr_image_pil(masked_pil)
            lines = clean_and_merge_lines(text_raw)

            # 5) 写入 txt（每行末尾追加 &&文件名 第N页&&）
            for line in lines:
                fout.write(f"{line} &&{base} 第{page_num}页&&\n")

            # 6) 调试中间图
            if DEBUG_SAVE_INTERMEDIATE:
                anno = bgr.copy()
                for (x1, y1, x2, y2) in boxes:
                    cv2.rectangle(anno, (x1, y1), (x2, y2), (0, 0, 255), 3)
                safe_imwrite(os.path.join(out_dir, f"page_{page_num}_binary.png"), bin_img)
                safe_imwrite(os.path.join(out_dir, f"page_{page_num}_lines.png"), lines_img)
                safe_imwrite(os.path.join(out_dir, f"page_{page_num}_tables_annotated.png"), anno)

    print(f"[DONE] 结果：{txt_path}")


def batch_process(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(input_dir):
        if name.lower().endswith(".pdf"):
            process_pdf(os.path.join(input_dir, name), output_dir)


if __name__ == "__main__":
    batch_process()
