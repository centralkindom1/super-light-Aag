import os
import requests
import json
import pytesseract
from pytesseract import Output
import pdfplumber
from PIL import Image
from image_preprocessing import preprocess_image_for_ocr

# --- 硬编码局域网配置 (参考 llmarena2.py) ---
API_URL = "https://www.deepseek.com:18080/v1/chat/completions"
API_KEY = "your api key"

class HybridOCRParser:
    def __init__(self, dpi=300):
        self.dpi = dpi

    def call_local_llm(self, messy_text):
        """调用局域网模型进行文本洗白与结构化"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "Host": "aiplus.airchina.com.cn:18080"
        }
        
        # 选用 DeepSeek-V3 或 Xinghuo-X1-70B 
        # 理由：V3 对破碎文本的逻辑推理能力最强
        payload = {
            "model": "DeepSeek-V3", 
            "messages": [
                {"role": "system", "content": "你是一个OCR文档修复专家。我会给你一段带有坐标聚合后的、可能存在断句或错别字的原始文本，请你：\n1. 修复由于OCR导致的错别字和断句问题。\n2. 使用Markdown格式输出，识别并保留一级标题(#)、二级标题(##)和正文。\n3. 不要输出任何解释，直接输出修复后的Markdown文本。"},
                {"role": "user", "content": f"以下是原始碎片文本：\n\n{messy_text}"}
            ],
            "temperature": 0.3,
            "stream": False
        }

        try:
            # 禁用 SSL 验证 (verify=False) 因为是局域网环境
            response = requests.post(API_URL, headers=headers, json=payload, verify=False, timeout=60)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            return f"LLM 修复失败: {response.text}"
        except Exception as e:
            return f"网络调用错误: {str(e)}"

    def extract_spatial_text(self, pil_img):
        """第一步：获取坐标级碎片并按物理行合并"""
        processed_img = preprocess_image_for_ocr(pil_img)
        # 获取详细的坐标字典
        d = pytesseract.image_to_data(processed_img, lang='chi_sim', output_type=Output.DICT)
        
        lines = []
        current_line = []
        last_top = -1
        
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            text = d['text'][i].strip()
            if not text: continue # 跳过空白
            
            top = d['top'][i]
            height = d['height'][i]
            
            # 逻辑：如果当前词的 Top 与上一词差值小于高度的一半，视为同一物理行
            if last_top == -1 or abs(top - last_top) < (height / 2):
                current_line.append(text)
            else:
                lines.append("".join(current_line))
                current_line = [text]
            last_top = top
            
        if current_line:
            lines.append("".join(current_line))
            
        return "\n".join(lines)

    def parse_pdf(self, pdf_path, callback=None):
        full_markdown = ""
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                if callback: callback(f"正在处理第 {i+1} 页...", int((i+1)/total*100))
                
                # 1. 视觉化
                img = page.to_image(resolution=self.dpi).original
                
                # 2. 提取带有空间逻辑的原始文本 (Local OCR)
                raw_spatial_text = self.extract_spatial_text(img)
                
                # 3. 语义修复与结构化 (Local LLM)
                # 注意：如果单页内容极多，建议分块发送，这里演示整页发送
                fixed_content = self.call_local_llm(raw_spatial_text)
                
                full_markdown += f"\n\n\n{fixed_content}"
                
        return full_markdown

# --- 测试代码 ---
if __name__ == "__main__":
    parser = HybridOCRParser()
    # 模拟运行
    # result = parser.parse_pdf("your_file.pdf")
    # print(result)