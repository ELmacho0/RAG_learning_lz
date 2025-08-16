# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 01:44:22 2025
@author: clearloveX
"""

import os
import fitz
from cnocr import CnOcr
from docx import Document
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from threading import Thread
import queue
import time


class PDFConverterApp:
    def __init__(self, master):
        self.master = master
        master.title("PDF_word转换工具")
        master.geometry("800x600")

        # 配置参数
        self.DPI = 200
        self.OCR_CONFIG = {
            'det_model_name': 'db_shufflenet_v2',
            'rec_model_name': 'densenet_lite_136-gru',
            'context': 'cpu'
        }

        self.running = False
        self.progress_queue = queue.Queue()

        self.create_widgets()
        self.update_progress()

    def create_widgets(self):
        """创建界面组件"""
        # 顶部控制面板
        control_frame = ttk.Frame(self.master, padding=10)
        control_frame.pack(fill=tk.X)

        self.btn_select = ttk.Button(control_frame, text="选择文件夹", command=self.select_folder)
        self.btn_select.pack(side=tk.LEFT, padx=5)

        self.btn_start = ttk.Button(control_frame, text="开始转换", command=self.start_conversion)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_cancel = ttk.Button(control_frame, text="取消", command=self.cancel_conversion, state=tk.DISABLED)
        self.btn_cancel.pack(side=tk.LEFT, padx=5)

        # 路径显示
        self.lbl_path = ttk.Label(control_frame, text="未选择文件夹")
        self.lbl_path.pack(side=tk.LEFT, padx=10)

        # 进度条
        self.progress = ttk.Progressbar(self.master, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)

        # 日志窗口
        self.log_area = scrolledtext.ScrolledText(self.master, wrap=tk.WORD)
        self.log_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.master, textvariable=self.status_var)
        self.status_bar.pack(fill=tk.X, padx=10, pady=5)

    def select_folder(self):
        """选择目标文件夹"""
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.lbl_path.config(text=folder_path)
            self.log("已选择文件夹：" + folder_path)

    def log(self, message):
        """记录日志信息"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_area.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_area.see(tk.END)
        self.master.update_idletasks()

    def update_progress(self):
        """更新进度和界面"""
        try:
            while True:
                msg = self.progress_queue.get_nowait()
                if msg == 'PROGRESS_UPDATE':
                    self.progress['value'] += 1
                elif msg == 'COMPLETE':
                    self.conversion_complete()
                elif isinstance(msg, dict):
                    self.status_var.set(msg.get('status', ''))
        except queue.Empty:
            pass
        self.master.after(100, self.update_progress)

    def start_conversion(self):
        """开始转换"""
        if not self.running and self.lbl_path['text'] != "未选择文件夹":
            self.running = True
            self.btn_start['state'] = tk.DISABLED
            self.btn_cancel['state'] = tk.NORMAL
            self.progress['value'] = 0
            self.log_area.delete(1.0, tk.END)

            target_folder = self.lbl_path['text']
            Thread(target=self.process_folder, args=(target_folder,), daemon=True).start()

    def cancel_conversion(self):
        """取消转换"""
        self.running = False
        self.log("转换已取消")
        self.status_var.set("操作已取消")

    def conversion_complete(self):
        """转换完成处理"""
        self.running = False
        self.btn_start['state'] = tk.NORMAL
        self.btn_cancel['state'] = tk.DISABLED
        self.status_var.set("转换完成")

    def process_folder(self, folder_path):
        """处理文件夹（线程中运行）"""
        try:
            pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
            self.progress['maximum'] = len(pdf_files)
            self.progress_queue.put({'status': '正在扫描PDF文件...'})

            for filename in pdf_files:
                if not self.running:
                    break

                base_name = os.path.splitext(filename)[0]
                pdf_path = os.path.join(folder_path, filename)
                output_path = os.path.join(folder_path, f"{base_name}.docx")

                self.progress_queue.put({'status': f'正在处理: {filename}'})
                self.log(f"开始处理文件: {filename}")

                try:
                    if os.path.exists(output_path):
                        self.log(f"跳过已存在文件: {filename}")
                        continue

                    self.convert_pdf_to_docx(pdf_path, output_path)
                    self.log(f"成功转换: {filename}")

                except Exception as e:
                    error_msg = f"处理 {filename} 失败: {type(e).__name__} - {str(e)}"
                    self.log(error_msg)
                    with open(os.path.join(folder_path, "conversion_errors.log"), "a") as f:
                        f.write(error_msg + "\n")

                self.progress_queue.put('PROGRESS_UPDATE')

            self.progress_queue.put('COMPLETE')
        except Exception as e:
            self.log(f"发生未知错误: {str(e)}")
            self.progress_queue.put('COMPLETE')

    def convert_pdf_to_docx(self, pdf_path, output_path):
        """转换单个PDF文件"""
        ocr = CnOcr(**self.OCR_CONFIG)
        doc = Document()

        with tempfile.TemporaryDirectory() as temp_dir:
            images = self.pdf_to_images(pdf_path, temp_dir)

            for idx, img_path in enumerate(images):
                if not self.running:
                    break

                try:
                    paragraphs = self.ocr_image(img_path, ocr)
                    for para in paragraphs:
                        doc.add_paragraph(para)
                    if idx != len(images) - 1:
                        doc.add_page_break()
                except Exception as e:
                    self.log(f"页面 {idx + 1} 识别失败: {str(e)}")

        if self.running:
            doc.save(output_path)

    def pdf_to_images(self, pdf_path, temp_dir):
        """生成PDF页面图像"""
        doc = fitz.open(pdf_path)
        images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=self.DPI)
            img_path = os.path.join(temp_dir, f"page_{page_num}.png")
            pix.save(img_path)
            images.append(img_path)
        return images

    # 修正ocr_image方法中的调用方式
    @staticmethod
    def ocr_image(image_path, ocr_engine):
        """执行OCR识别"""
        result = ocr_engine.ocr(image_path)
        raw_lines = [item["text"] for item in result]
        return PDFConverterApp.merge_paragraphs(raw_lines)  # 改为通过类名调用

    # 保持merge_paragraphs为静态方法

    def merge_paragraphs(lines):
        """合并段落"""
        paragraphs = []
        current_para = []
        END_PUNCTS = ('。', '！', '？', '”', '…', '；', '》', '.', '!', '?')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if current_para:
                if current_para[-1][-1] in END_PUNCTS:
                    paragraphs.append(''.join(current_para))
                    current_para = [line]
                else:
                    current_para.append(line)
            else:
                current_para.append(line)

        if current_para:
            paragraphs.append(''.join(current_para))

        return paragraphs


if __name__ == "__main__":
    root = tk.Tk()
    app = PDFConverterApp(root)
    root.mainloop()