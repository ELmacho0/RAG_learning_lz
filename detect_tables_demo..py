import os
import math
import numpy as np
import cv2
from pdf2image import convert_from_path
from PIL import Image

# ====== 可调参数（重点看这里） ======
POPPLER_BIN = r""  # 例如 r"C:\USER_PROGRAM\poppler-24.08.0\Library\bin"；若已配好PATH可留空
DPI = 300  # 渲染清晰度；表格线很细时可升到 400
BIN_METHOD = "adaptive"  # "adaptive" 或 "otsu"
ADAPTIVE_BLOCK_SIZE = 35  # 必须是奇数，越大越“粗犷”
ADAPTIVE_C = 15  # 越大越偏向黑
GAUSS_BLUR = 3  # 预模糊核大小，降噪；0或1代表不模糊

# 线结构元素尺寸按页面尺寸比例设置（适合竖版A4）
VERT_KERNEL_RATIO = 0.018  # 竖线宽度比例（相对宽度），例如 0.018 ≈ 1.8%
HORZ_KERNEL_RATIO = 0.015  # 横线高度比例（相对高度）
MORPH_ITER = 2  # 腐蚀/膨胀迭代次数；线条细就加大

# 候选表格过滤阈值（相对整页）
MIN_TABLE_AREA_RATIO = 0.01  # 最小面积（表格很小时可降到 0.005）
MIN_W_RATIO = 0.12  # 最小宽度比例（避免把段落框当表格）
MIN_H_RATIO = 0.06  # 最小高度比例

# 近邻框合并（避免表格被分成多块）
MERGE_IOU_THRESHOLD = 0.15  # 两框重叠超过该阈值就合并
MERGE_PIX_GAP = 12  # 框之间像素间隙小于这个就视为相邻，需要合并


# ===============================


def pil_to_cv(img_pil: Image.Image):
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
    # 竖线核： (height, 1)
    vk = max(1, int(w * VERT_KERNEL_RATIO))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    vertical = cv2.erode(bin_img, vertical_kernel, iterations=MORPH_ITER)
    vertical = cv2.dilate(vertical, vertical_kernel, iterations=MORPH_ITER)

    # 横线核： (1, width)
    hk = max(1, int(h * HORZ_KERNEL_RATIO))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    horizontal = cv2.erode(bin_img, horizontal_kernel, iterations=MORPH_ITER)
    horizontal = cv2.dilate(horizontal, horizontal_kernel, iterations=MORPH_ITER)

    # 合并横竖线
    table_lines = cv2.bitwise_or(vertical, horizontal)
    # 轻微膨胀，连通断裂线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_lines = cv2.dilate(table_lines, kernel, iterations=1)
    return table_lines


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
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
            if used[i]:
                continue
            ax1, ay1, ax2, ay2 = boxes[i]
            merged = False
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = boxes[j]
                # 近邻判断（允许小间隙）
                near = (abs(ax1 - bx1) <= gap or abs(ax2 - bx2) <= gap or
                        abs(ay1 - by1) <= gap or abs(ay2 - by2) <= gap)
                # IoU 判断
                if near or iou((ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2)) >= iou_th:
                    nx1, ny1 = min(ax1, bx1), min(ay1, by1)
                    nx2, ny2 = max(ax2, bx2), max(ay2, by2)
                    ax1, ay1, ax2, ay2 = nx1, ny1, nx2, ny2
                    used[j] = True
                    merged = True
                    changed = True
            used[i] = True
            new.append((ax1, ay1, ax2, ay2))
        boxes = new
    return boxes


def detect_tables_on_image(bgr):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bin_img = binarize(gray)
    table_lines = extract_lines(bin_img)

    # 找轮廓
    contours, _ = cv2.findContours(table_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    min_area = int(MIN_TABLE_AREA_RATIO * w * h)
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww * hh < min_area:
            continue
        # 宽高比例过滤（避免把段落框/小装饰当表格）
        if ww < int(MIN_W_RATIO * w) or hh < int(MIN_H_RATIO * h):
            continue
        boxes.append((x, y, x + ww, y + hh))

    # 合并近邻/重叠框
    boxes = merge_boxes(boxes)
    # 由上到下、左到右排序
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes, table_lines, bin_img


def save_crops_and_debug(bgr, boxes, out_dir, page_num):
    # 确保输出目录存在并打印绝对路径
    os.makedirs(out_dir, exist_ok=True)
    out_dir_abs = os.path.abspath(out_dir)
    print(f"[DEBUG] 输出目录: {out_dir_abs}")

    h, w = bgr.shape[:2]
    print(f"[DEBUG] 页面尺寸: width={w}, height={h}")
    if not boxes:
        print("[WARN] 没有可保存的表格框。")

    annotated = bgr.copy()
    saved_any = False

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        # 打印原始坐标
        print(f"[DEBUG] 原始框{i}: ({x1}, {y1}, {x2}, {y2})")
        # 钳位到图像范围
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            print(f"[WARN] 框{i} 钳位后无效，跳过。")
            continue

        # 画框
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # 裁剪
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"[WARN] 框{i} 裁剪结果为空，跳过。")
            continue

        # 保存文件（先用 OpenCV，失败再用 PIL）
        out_path = os.path.join(out_dir_abs, f"page_{page_num}_table_{i}.png")
        ok = cv2.imwrite(out_path, crop)
        if not ok:
            print(f"[WARN] cv2.imwrite 失败，改用 PIL 保存：{out_path}")
            try:
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(out_path)
                print(f"[INFO] 已用 PIL 保存：{out_path}")
                saved_any = True
            except Exception as e:
                print(f"[ERROR] PIL 保存仍失败：{e}")
        else:
            print(f"[INFO] 保存表格截图：{out_path}")
            saved_any = True

    # 不管怎样都保存一张标注图
    anno_path = os.path.join(out_dir_abs, f"page_{page_num}_tables_annotated.png")
    cv2.imwrite(anno_path, annotated)
    print(f"[INFO] 已保存标注图：{anno_path}")

    if not saved_any and boxes:
        print("[HINT] 检测到框但未成功保存裁剪图："
              "1) 看看标注图里框是否在页面外；2) 检查磁盘权限/路径；"
              "3) 若框贴边，可适当放宽 MIN_W_RATIO/MIN_H_RATIO 或增大 MERGE_PIX_GAP。")


def pdf_page_to_image(pdf_path, page_num=1, dpi=DPI, poppler_bin=POPPLER_BIN):
    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=page_num,
        last_page=page_num,
        poppler_path=poppler_bin if poppler_bin else None
    )
    return images[0]  # PIL.Image


def main(pdf_path, output_dir, page_num=1):
    from pdf2image import convert_from_path

    def pdf_page_to_image(pdf_path, page_num=1, dpi=300, poppler_bin=""):
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_num,
            last_page=page_num,
            poppler_path=poppler_bin if poppler_bin else None
        )
        return images[0]

    # === 你的 POPPLER_BIN / DPI 等参数照旧 ===
    pil_img = pdf_page_to_image(pdf_path, page_num=page_num, dpi=DPI, poppler_bin=POPPLER_BIN)
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 检测
    boxes, table_lines, bin_img = detect_tables_on_image(bgr)
    print(f"[INFO] 检测到 {len(boxes)} 个表格区域。")
    # 保存裁剪和标注
    save_crops_and_debug(bgr, boxes, output_dir, page_num)

    # 额外保存中间过程图（便于调参）
    bin_path = os.path.join(os.path.abspath(output_dir), f"page_{page_num}_binary.png")
    lines_path = os.path.join(os.path.abspath(output_dir), f"page_{page_num}_lines.png")
    cv2.imwrite(bin_path, bin_img)
    cv2.imwrite(lines_path, table_lines)
    print(f"[INFO] 中间图已保存：{bin_path}")
    print(f"[INFO] 中间图已保存：{lines_path}")

if __name__ == "__main__":
    # 示例：调整为你的路径
    PDF_PATH = r"D:\智能知识库\PDF原始数据\《行政办公费用使用及管理工作细则》（图片）.pdf"
    OUTPUT_DIR = r"D:\智能知识库\pdf_output\demo"
    PAGE = 7  # 要调参就先选有表格的一页
    main(PDF_PATH, OUTPUT_DIR, PAGE)
