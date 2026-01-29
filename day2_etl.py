# 文件名: day2_etl.py
import json
import uuid
import os
import re

# --- 配置区域 ---
# 建议切片长度 (对应 BGE-M3 的最佳窗口)
CHUNK_SIZE = 800
# 切片重叠长度 (防止一句话被切断)
CHUNK_OVERLAP = 100

class ContextAssembler:
    def __init__(self, input_file):
        self.input_file = input_file
        self.raw_data = []
        self.chunks = []
        
        # 状态机指针
        self.current_h1 = "无一级标题"
        self.current_h2 = ""
        self.doc_name = os.path.basename(input_file).replace(".json", ".pdf")

    def load_data(self):
        """读取 Day 1 的中间态 JSON"""
        if not os.path.exists(self.input_file):
            print(f"[Error] 找不到文件: {self.input_file}")
            return False
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        print(f"[Info] 成功加载 {len(self.raw_data)} 行原始数据")
        return True

    def run_etl(self):
        """执行上下文重组与切片"""
        print("[Info] 开始 ETL 处理...")
        
        for item in self.raw_data:
            role = item.get('role', 'BODY')
            text = item.get('text', '').strip()
            page = item.get('page', 0)
            
            # 清理 Markdown 符号
            clean_text = text.replace('# ', '').replace('## ', '').replace('**', '')
            
            # 状态机逻辑
            if role == 'H1':
                self.current_h1 = clean_text
                self.current_h2 = "" # 遇到新 H1，清空 H2
                continue # 标题本身不单独作为向量块，除非你需要
                
            elif role == 'H2':
                self.current_h2 = clean_text
                continue
            
            elif role == 'BODY':
                if not clean_text: continue
                
                # 构建增强上下文 (Prompt Engineering for Embedding)
                # 格式： 文档名 + 路径 + 内容
                context_header = f"文档来源：{self.doc_name}\n章节路径：{self.current_h1} > {self.current_h2}\n内容正文："
                
                # 判断是否需要切片
                if len(clean_text) > CHUNK_SIZE:
                    sub_chunks = self.smart_split(clean_text)
                    for idx, sub_txt in enumerate(sub_chunks):
                        self.create_chunk_entry(sub_txt, page, context_header, chunk_idx=idx)
                else:
                    self.create_chunk_entry(clean_text, page, context_header)

    def smart_split(self, text):
        """长文本智能切分 (滑动窗口)"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + CHUNK_SIZE
            
            # 如果没到结尾，尝试寻找最近的句号/分号断句，避免切断句子
            if end < text_len:
                # 在 end 附近往前找标点符号
                look_back = text[max(0, end-50):end]
                split_pos = -1
                for punct in ['。', '；', '！', '?', '\n']:
                    pos = look_back.rfind(punct)
                    if pos != -1:
                        split_pos = max(0, end-50) + pos + 1
                        break
                
                if split_pos != -1:
                    end = split_pos
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            # 滑动窗口：下一次起点 = 当前终点 - 重叠量
            start = end - CHUNK_OVERLAP
            if start >= text_len: break
            
        return chunks

    def create_chunk_entry(self, text, page, context_header, chunk_idx=0):
        """生成符合 Vector DB 要求的 JSON 对象"""
        
        # 核心策略：Embedding 字段包含所有上下文
        full_embedding_text = f"{context_header}\n{text}"
        
        chunk_obj = {
            "id": str(uuid.uuid4()),
            "metadata": {
                "source": self.doc_name,
                "page": page,
                "h1": self.current_h1,
                "h2": self.current_h2,
                "char_len": len(text),
                "is_split": chunk_idx > 0
            },
            # 关键分离：embedding_text 用于向量化，pure_text 用于 LLM 回答
            "content": {
                "embedding_text": full_embedding_text, 
                "pure_text": text
            }
        }
        self.chunks.append(chunk_obj)

    def save_output(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        print(f"[Success] ETL 完成！共生成 {len(self.chunks)} 个知识块。")
        print(f"[Output] 结果已保存至: {output_path}")

# --- 执行入口 ---
if __name__ == "__main__":
    # 假设 Day 1 导出的文件名为 'day1_export.json'
    # 您可以手动修改这里的文件名，或者通过命令行参数传入
    input_json = "day1_export.json" 
    output_json = "day2_vector_ready.json"
    
    # 创建模拟数据文件以便您直接测试 (如果目录下没有文件)
    if not os.path.exists(input_json):
        print(f"提示: 未找到 {input_json}，正在创建测试数据...")
        mock_data = [
            {"role": "H1", "text": "第一章 航空安全规定", "page": 1},
            {"role": "H2", "text": "1.1 电子设备使用", "page": 1},
            {"role": "BODY", "text": "在飞机滑行、起飞、下降和着陆阶段，禁止使用所有发射无线电信号的便携式电子设备。", "page": 1},
            {"role": "BODY", "text": "这里是一段非常长的文本用于测试切片功能..." * 50, "page": 2}
        ]
        with open(input_json, 'w', encoding='utf-8') as f:
            json.dump(mock_data, f, ensure_ascii=False)
            
    etl = ContextAssembler(input_json)
    if etl.load_data():
        etl.run_etl()
        etl.save_output(output_json)