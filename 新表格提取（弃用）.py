# table_image_to_json.py
# 通用：图片/扫描件表格 -> JSON（几何层级 + 方案B：表头嵌入键名）
# 依赖：opencv-python, numpy, pillow, rapidocr-onnxruntime

import os
import re
import json
import cv2
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
from rapidocr_onnxruntime import RapidOCR


# ============== 基础工具 ==============

def ignore_whitespace(s: str) -> str:
    """忽略换行与多余空白（保持你之前的做法，不做额外业务清洗）。"""
    if not s:
        return ""
    s = re.sub(r"\s+", "", s)
    s = s.strip("·,，。.:：;；|丨/／\\")
    return s


def unique_sorted(vals: List[int], tol: int = 2) -> List[int]:
    vals = sorted(vals)
    merged = []
    for v in vals:
        if not merged or abs(v - merged[-1]) > tol:
            merged.append(v)
    return merged


# ============== 数据结构 ==============

@dataclass
class Cell:
    r0: int
    c0: int
    rowspan: int
    colspan: int
    x1: int
    y1: int
    x2: int
    y2: int
    text: str = ""


# ============== 预处理与线提取 ==============

def try_adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    try:
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 15
        )
        return th
    except Exception:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return th


def extract_lines(bin_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = bin_img.shape
    horiz_len = max(10, w // 40)
    vert_len = max(10, h // 40)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))

    h_e = cv2.erode(bin_img, hk, iterations=1)
    h_lines = cv2.dilate(h_e, hk, iterations=1)

    v_e = cv2.erode(bin_img, vk, iterations=1)
    v_lines = cv2.dilate(v_e, vk, iterations=1)

    grid = cv2.bitwise_or(h_lines, v_lines)
    return h_lines, v_lines, grid


def detect_grid_xy(h_lines: np.ndarray, v_lines: np.ndarray) -> Tuple[List[int], List[int]]:
    y_proj = h_lines.sum(axis=1)
    x_proj = v_lines.sum(axis=0)
    y_cand = [i for i, v in enumerate(y_proj) if v > 0]
    x_cand = [i for i, v in enumerate(x_proj) if v > 0]
    y_coords = unique_sorted(y_cand, tol=3)
    x_coords = unique_sorted(x_cand, tol=3)

    def thin(coords: List[int], min_gap=4):
        out, last = [], -10 ** 9
        for c in coords:
            if c - last >= min_gap:
                out.append(c);
                last = c
        return out

    return thin(y_coords, 4), thin(x_coords, 4)


# ============== 合并单元格识别 ==============

def grid_presence(grid: np.ndarray, x1, y1, x2, y2, side: str, thr: float = 0.6) -> bool:
    h, w = grid.shape
    pad = 2
    if side == 'left':
        xs = slice(max(0, x1 - pad), min(w, x1 + pad + 1));
        ys = slice(y1, y2)
    elif side == 'right':
        xs = slice(max(0, x2 - pad), min(w, x2 + pad + 1));
        ys = slice(y1, y2)
    elif side == 'top':
        xs = slice(x1, x2);
        ys = slice(max(0, y1 - pad), min(h, y1 + pad + 1))
    else:
        xs = slice(x1, x2);
        ys = slice(max(0, y2 - pad), min(h, y2 + pad + 1))

    roi = grid[ys, xs]
    return (roi.mean() / 255.0) > thr


def merge_spanned_cells(y_coords: List[int], x_coords: List[int], grid: np.ndarray) -> List[Cell]:
    R, C = len(y_coords) - 1, len(x_coords) - 1
    used = np.zeros((R, C), dtype=bool)
    cells: List[Cell] = []

    for r in range(R):
        for c in range(C):
            if used[r, c]:
                continue
            r2, c2 = r, c
            # 向右扩展
            while c2 + 1 < C:
                x1, y1 = x_coords[c], y_coords[r]
                x2, y2 = x_coords[c2 + 1], y_coords[r2 + 1]
                if grid_presence(grid, x1, y1, x2, y2, 'right'):
                    break
                c2 += 1
            # 向下扩展
            expanded = True
            while expanded:
                expanded = False
                if r2 + 1 < R:
                    x1, y1 = x_coords[c], y_coords[r]
                    x2, y2 = x_coords[c2 + 1], y_coords[r2 + 1]
                    if not grid_presence(grid, x1, y1, x2, y2, 'bottom'):
                        r2 += 1;
                        expanded = True

            used[r:r2 + 1, c:c2 + 1] = True
            x1, y1 = x_coords[c], y_coords[r]
            x2, y2 = x_coords[c2 + 1], y_coords[r2 + 1]
            cells.append(Cell(r, c, r2 - r + 1, c2 - c + 1, x1, y1, x2, y2))
    return cells


# ============== OCR（RapidOCR）与安全裁剪 ==============

class OCR:
    def __init__(self, use_cuda: bool = False):
        self.engine = RapidOCR(device_id=0, use_cuda=use_cuda)

    def safe_crop(self, img: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
        best = img[max(0, y1 + 3):max(0, y2 - 3), max(0, x1 + 3):max(0, x2 - 3)]
        for inset in (3, 4, 5, 6, 7, 8):
            xi1 = max(0, x1 + inset);
            yi1 = max(0, y1 + inset)
            xi2 = max(xi1 + 1, min(img.shape[1], x2 - inset))
            yi2 = max(yi1 + 1, min(img.shape[0], y2 - inset))
            crop = img[yi1:yi2, xi1:xi2]
            if crop.size == 0:
                continue
            border = 2
            ring = np.concatenate([
                crop[:border, :].reshape(-1, 3), crop[-border:, :].reshape(-1, 3),
                crop[:, :border].reshape(-1, 3), crop[:, -border:].reshape(-1, 3)
            ], axis=0)
            if ring.size == 0:
                best = crop;
                break
            gray = cv2.cvtColor(ring.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).reshape(-1)
            dark_ratio = (gray < 40).mean()
            best = crop
            if dark_ratio < 0.15:
                break
        return best

    def text(self, img_bgr: np.ndarray) -> str:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res, _ = self.engine(img_rgb)
        if not res:
            return ""
        txt = "".join([it[1] for it in res])
        return ignore_whitespace(txt)


def ocr_cells(img: np.ndarray, cells: List[Cell], ocr: OCR) -> None:
    for c in cells:
        crop = ocr.safe_crop(img, c.x1, c.y1, c.x2, c.y2)
        c.text = ocr.text(crop)


# ============== 覆盖矩阵与表头估计（结构启发式） ==============

def build_coverage(cells: List[Cell], n_rows: int, n_cols: int) -> List[List[Optional[Cell]]]:
    cover = [[None for _ in range(n_cols)] for __ in range(n_rows)]
    for cell in cells:
        for rr in range(cell.r0, cell.r0 + cell.rowspan):
            for cc in range(cell.c0, cell.c0 + cell.colspan):
                cover[rr][cc] = cell
    return cover


def guess_header_rows(cells: List[Cell], n_cols: int, max_scan_rows: int = 3) -> int:
    by_row: Dict[int, List[Cell]] = {}
    for c in cells:
        by_row.setdefault(c.r0, []).append(c)
    rows = sorted(by_row.keys())
    skip = 0
    for r in rows:
        if r >= max_scan_rows:
            break
        starts = by_row[r]
        if not starts:
            continue
        avg_colspan = sum(c.colspan for c in starts) / len(starts)
        short_ratio = sum(1 for c in starts if 0 < len(c.text) <= 6) / len(starts)
        many_starts = len(starts) >= max(2, n_cols // 3)
        if avg_colspan >= 2 or (short_ratio >= 0.6 and many_starts):
            skip += 1
        else:
            break
    return skip


# ============== 方案B：表头嵌入键名的整形器 ==============

def split_rightmost_value(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[／/]", s)
    return [p for p in (t.strip() for t in parts) if p]


def extract_headers_from_top(cells: List[Cell], n_cols: int, skip_header_rows: int) -> List[str]:
    headers = [""] * n_cols
    # 先按 skip_header_rows 范围内找
    for col in range(n_cols):
        cand = [(c.r0, c.text) for c in cells if c.c0 == col and c.r0 < max(skip_header_rows, 1) and c.text]
        if cand:
            cand.sort(key=lambda x: x[0])
            headers[col] = cand[0][1]
        else:
            # 回退：直接取该列最上方的非空文本
            cand2 = [(c.r0, c.text) for c in cells if c.c0 == col and c.text]
            if cand2:
                cand2.sort(key=lambda x: x[0])
                headers[col] = cand2[0][1]
            else:
                headers[col] = f"col_{col}"
    return headers


def build_structure_B(cells: List[Cell], n_rows: int, n_cols: int, cover, headers: List[str], skip_header_rows: int):
    # ---- 工具：取父节点右侧第一列、父行跨度内的子起点 ----
    def next_starts_in_span(parent_cell: Cell):
        col_next = parent_cell.c0 + parent_cell.colspan
        if col_next >= n_cols:
            return {}
        starts = {}
        rr = parent_cell.r0
        r_end = parent_cell.r0 + parent_cell.rowspan - 1
        while rr <= r_end:
            ch = cover[rr][col_next]
            if ch is None:
                rr += 1
                continue
            if ch.c0 == col_next and ch.r0 == rr:
                starts[rr] = ch
                rr += ch.rowspan
            else:
                rr += 1
        return starts

    # ---- 递归：构建从某个节点向右的分支对象 ----
    def build_branch(node: Cell):
        branch_obj = {}
        current = node
        while True:
            children = next_starts_in_span(current)
            if not children:
                # 最右列：把值按 / 或 ／ 拆成数组
                return split_rightmost_value(current.text)

            next_col = current.c0 + current.colspan
            hdr = headers[next_col] or f"col_{next_col}"

            if len(children) == 1:
                only_child = next(iter(children.values()))
                # 如果 child 已经是最右列，直接作为值数组返回；否则继续单链条推进
                if only_child.c0 + only_child.colspan >= n_cols:
                    branch_obj[hdr] = split_rightmost_value(only_child.text)
                    return branch_obj
                else:
                    branch_obj[hdr] = only_child.text or ""
                    current = only_child
                    continue
            else:
                # 并列：在下一列表头下展开 {label: 子分支}
                mapping = {(ch.text or ""): build_branch(ch) for ch in children.values()}
                branch_obj[hdr] = mapping
                return branch_obj

    # ---- 根节点选择（稳健回退）----
    result = {}

    if not cells:
        return result

    # 先按“最左列 + 跳过表头”找根
    cols_sorted = sorted({c.c0 for c in cells})
    min_c0 = cols_sorted[0]
    roots = sorted([c for c in cells if c.c0 == min_c0 and c.r0 >= skip_header_rows], key=lambda x: x.r0)

    # 回退1：如果没有根，尝试不跳过表头（skip=0）
    if not roots:
        roots = sorted([c for c in cells if c.c0 == min_c0 and c.r0 >= 0], key=lambda x: x.r0)

    # 回退2：如果这一列确实没有任何起点，再向右找下一列
    if not roots:
        for cand_col in cols_sorted[1:]:
            roots = sorted([c for c in cells if c.c0 == cand_col and c.r0 >= skip_header_rows], key=lambda x: x.r0)
            if roots:
                min_c0 = cand_col
                break

    # 回退3：仍为空，彻底兜底——全表最上方的起点当根
    if not roots:
        roots = sorted(cells, key=lambda x: (x.r0, x.c0))

    if not roots:
        return result  # 理论上到不了这里

    # ---- 顶层展开 ----
    if len(roots) > 1:
        # 多根并列：在该列的表头名下做 {label: 分支}
        hdr0 = headers[min_c0] or f"col_{min_c0}"
        mapping = {(r.text or ""): build_branch(r) for r in roots}
        result[hdr0] = mapping
        return result

    # 单根：连续单链条滚动，直至出现并列或终止
    cur = roots[0]
    result[headers[cur.c0] or f"col_{cur.c0}"] = cur.text or ""

    while True:
        kids = next_starts_in_span(cur)
        if not kids:
            break

        next_col = cur.c0 + cur.colspan
        hdr = headers[next_col] or f"col_{next_col}"

        if len(kids) == 1:
            only_child = next(iter(kids.values()))
            if only_child.c0 + only_child.colspan >= n_cols:
                result[hdr] = split_rightmost_value(only_child.text)
                break
            else:
                result[hdr] = only_child.text or ""
                cur = only_child
                continue
        else:
            mapping = {(ch.text or ""): build_branch(ch) for ch in kids.values()}
            result[hdr] = mapping
            break

    return result


# ============== 主函数（供外部调用） ==============

def table_image_to_json(
        image_path: str,
        out_json: str,
        use_cuda: bool = False,
        header_auto: bool = True,
        header_scan_rows: int = 3
) -> Dict[str, Any]:
    """
    :param image_path: 输入图片路径
    :param out_json: 输出 JSON 文件路径
    :param use_cuda: RapidOCR 是否用 GPU
    :param header_auto: 是否启用表头行数启发式估计
    :param header_scan_rows: 顶部扫描行数上限
    :return: 解析后的 JSON 对象（方案B结构）
    """
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    mx = max(img.shape[:2])
    if mx < 1600:
        scale = 1600.0 / mx
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_img = try_adaptive_threshold(gray)

    h_lines, v_lines, grid = extract_lines(bin_img)
    y_coords, x_coords = detect_grid_xy(h_lines, v_lines)

    if len(y_coords) < 2 or len(x_coords) < 2:
        raise RuntimeError("未检测到足够的表格线，请提高图像清晰度或对比度。")

    cells = merge_spanned_cells(y_coords, x_coords, grid)
    n_rows, n_cols = len(y_coords) - 1, len(x_coords) - 1

    # OCR
    ocr = OCR(use_cuda=use_cuda)
    ocr_cells(img, cells, ocr)

    # 覆盖矩阵 & 表头行估计
    cover = build_coverage(cells, n_rows, n_cols)
    skip_rows = guess_header_rows(cells, n_cols, header_scan_rows) if header_auto else 0

    # 防止 skip 过大导致找不到根（把它夹在 0 ~ n_rows-2 之间）
    if skip_rows < 0:
        skip_rows = 0
    if skip_rows > max(0, n_rows - 2):
        skip_rows = max(0, n_rows - 2)

    # 表头提取（不做业务词写死）
    headers = extract_headers_from_top(cells, n_cols, skip_rows)

    # 方案B结构化
    data = build_structure_B(cells, n_rows, n_cols, cover, headers, skip_rows)

    # 轻度清理空键/空值（不触碰实际文本与百分号）
    def prune(x):
        if isinstance(x, dict):
            return {k: prune(v) for k, v in x.items() if k != "" or v not in ("", None, [], {})}
        if isinstance(x, list):
            return [prune(v) for v in x if v not in ("", None, [], {})]
        return x

    data = prune(data)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


# ============== 示例直呼（仅本文件单独运行时） ==============
if __name__ == "__main__":
    # 作为库被 import 时，不会执行这里；以下仅示例。
    example_image = "ccc.png"
    example_out = "ccc.json"
    if os.path.exists(example_image):
        res = table_image_to_json(
            image_path=example_image,
            out_json=example_out,
            use_cuda=False,
            header_auto=True,
            header_scan_rows=3
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print("示例：请把待识别图片命名为 input.png 放在同目录后直接运行此脚本。")
