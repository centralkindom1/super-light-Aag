# pdf_structure_parser.py
import pdfplumber
import pytesseract
from pytesseract import Output
import re
import pandas as pd
from PIL import Image
from day1_shared_config import RAGConfig 
from image_preprocessing import preprocess_image_for_ocr

class DocumentLine:
    def __init__(self, text, font_size, is_bold=False, page_num=0, is_ocr=False):
        self.text = text.strip()
        self.font_size = float(font_size)
        self.is_bold = is_bold
        self.page_num = page_num
        self.is_ocr = is_ocr
        self.role = "BODY" 

class PDFStructureParser:
    def __init__(self, filepath, use_ocr=True):
        self.filepath = filepath
        self.use_ocr = use_ocr
        self.parsed_lines = []
        self.body_font_size = 10.5 

    def parse(self, callback_signal=None):
        raw_lines = []
        try:
            with pdfplumber.open(self.filepath) as pdf:
                total_pages = len(pdf.pages)
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    
                    if self.use_ocr:
                        # --- 强制 OCR 视觉识别模式 ---
                        if callback_signal:
                            callback_signal.emit(f"📸 第 {page_num} 页: 正在进行视觉布局分析...", int((page_num/total_pages)*100))
                        
                        img = page.to_image(resolution=RAGConfig.OCR_DPI).original
                        processed_img = preprocess_image_for_ocr(img)
                        
                        # ✨ 关键：使用 image_to_data 获取结构化字典
                        ocr_data = pytesseract.image_to_data(processed_img, lang='chi_sim', output_type=Output.DICT)
                        
                        # 将 OCR 碎片组合成“行”并估算高度
                        page_lines = self._process_ocr_data_to_lines(ocr_data, page_num)
                        raw_lines.extend(page_lines)
                    
                    else:
                        # --- 矢量提取模式 ---
                        if callback_signal:
                            callback_signal.emit(f"📄 第 {page_num} 页: 提取矢量文本...", int((page_num/total_pages)*100))
                        
                        text_objects = page.extract_words(extra_attrs=["size", "fontname"])
                        for obj in text_objects:
                            font_name = str(obj.get('fontname', '')).lower()
                            is_bold = any(kw in font_name for kw in ["bold", "black", "heavy"])
                            raw_lines.append(DocumentLine(obj['text'], obj['size'], is_bold, page_num))

            self.parsed_lines = self._analyze_structure(raw_lines)
            return self.parsed_lines
            
        except Exception as e:
            raise Exception(f"解析错误: {str(e)}")

    def _process_ocr_data_to_lines(self, data, page_num):
        """将 Tesseract 返回的碎词聚合为行，并提取视觉高度"""
        lines = []
        n_boxes = len(data['text'])
        
        current_line_text = []
        current_line_heights = []
        last_top = -1
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if not text: continue
            
            top = data['top'][i]
            height = data['height'][i]
            
            # 判断是否还在同一行 (允许 10 像素的垂直偏差)
            if last_top == -1 or abs(top - last_top) <= 10:
                current_line_text.append(text)
                current_line_heights.append(height)
            else:
                # 换行了，保存上一行
                if current_line_text:
                    avg_height = sum(current_line_heights) / len(current_line_heights)
                    full_text = "".join(current_line_text)
                    # 模拟字号：OCR 的像素高度需要换算或直接作为参考
                    lines.append(DocumentLine(full_text, avg_height, False, page_num, is_ocr=True))
                
                current_line_text = [text]
                current_line_heights = [height]
            
            last_top = top
            
        # 最后一行
        if current_line_text:
            avg_height = sum(current_line_heights) / len(current_line_heights)
            lines.append(DocumentLine("".join(current_line_text), avg_height, False, page_num, is_ocr=True))
            
        return lines

    def _analyze_structure(self, lines):
        if not lines: return []
        
        # 分别计算矢量和 OCR 模式下的基准高度
        sizes = [round(l.font_size, 1) for l in lines if l.text.strip()]
        if not sizes: return []
        self.body_font_size = max(set(sizes), key=sizes.count)
        
        for line in lines:
            # 视觉判定逻辑
            # 1. 显著比正文高 (如果是 OCR 模式，像素高度差通常比较明显)
            # 2. 符合特定正则 (如 第1章, 1.1)
            size_diff = line.font_size - self.body_font_size
            
            # 针对 OCR 模式，高度差异阈值需要调优
            threshold = RAGConfig.HEADER_SIZE_THRESHOLD 
            if line.is_ocr:
                threshold = threshold * 2 # OCR 像素波动大，阈值翻倍
            
            if size_diff > threshold + 5:
                line.role = "H1"
            elif size_diff > threshold:
                line.role = "H2"
            elif re.match(r'^(第[一二三四五六七八九十\d]+[章节]|[1-9]\.[1-9])', line.text):
                line.role = "H2"
            else:
                line.role = "BODY"
        
        # 合并逻辑
        merged = []
        if not lines: return []
        curr = lines[0]
        for nxt in lines[1:]:
            if nxt.role == curr.role and nxt.page_num == curr.page_num:
                connector = "" if re.search(r'[\u4e00-\u9fa5]', curr.text) else " "
                curr.text += connector + nxt.text
            else:
                merged.append(curr)
                curr = nxt
        merged.append(curr)
        return merged

    def build_tree_structure(self):
        root = []
        curr_h1, curr_h2 = None, None
        for line in self.parsed_lines:
            item = {'type': line.role, 'text': line.text[:60], 'full_text': line.text, 'page': line.page_num, 'children': []}
            if line.role == 'H1':
                curr_h1 = item; curr_h2 = None; root.append(curr_h1)
            elif line.role == 'H2':
                curr_h2 = item
                if curr_h1: curr_h1['children'].append(curr_h2)
                else: root.append(curr_h2)
            else:
                if curr_h2: curr_h2['children'].append(item)
                elif curr_h1: curr_h1['children'].append(item)
                else: root.append(item)
        return root