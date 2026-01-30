import sys
import os
import json
import time
import uuid
import sqlite3
import logging
import requests
import urllib3
import numpy as np
import cv2
import pdfplumber
import pytesseract
from pytesseract import Output
from concurrent.futures import ThreadPoolExecutor

# PyQt5 UI 库
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLineEdit, QLabel, QFileDialog, 
                             QTextEdit, QTreeWidget, QTreeWidgetItem, 
                             QSplitter, QMessageBox, QTabWidget,
                             QSpinBox, QGroupBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QColor, QFont

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================================================================
#  1. 配置中心
# ==============================================================================
class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TESSERACT_CMD = r'D:\Python\Scripts\tesseract.exe'
    TESSDATA_DIR = r'D:\Python\Scripts\tessdata'
    
    # API 配置
    API_BASE_URL = "https://www.deepseekandsiconflow:18080/v1"
    API_KEY = "YOUR API KEY"
    
    # 模型配置
    MODEL_PARSER = "DeepSeek-V3"
    MODEL_REWRITE = "DeepSeek-V3"
    MODEL_GEN = "DeepSeek-R1"
    MODEL_EMBED = "bge-m3"
    MODEL_RERANK = "bge-reranker-v2-m3"

    # ETL 配置
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100

    @staticmethod
    def setup_tesseract():
        if os.path.exists(Config.TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD
        if os.path.exists(Config.TESSDATA_DIR):
            os.environ['TESSDATA_PREFIX'] = Config.TESSDATA_DIR

Config.setup_tesseract()

# ==============================================================================
#  2. 核心业务逻辑层
# ==============================================================================
class RAGCoreService:
    def __init__(self, db_path=None):
        self.db_path = db_path

    def init_db(self, db_path):
        """初始化指定路径的 SQLite 数据库"""
        self.db_path = db_path
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks_full_index (
                id TEXT PRIMARY KEY,
                doc_title TEXT,
                page_num INTEGER,
                chapter_path TEXT,
                embedding_text TEXT,
                pure_text TEXT,
                vector_blob BLOB,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def _api_post(self, endpoint, payload, timeout=120):
        url = f"{Config.API_BASE_URL}/{endpoint}"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {Config.API_KEY}"}
        try:
            response = requests.post(url, headers=headers, json=payload, verify=False, timeout=timeout)
            if response.status_code == 200: return response.json()
            print(f"API Error ({endpoint}): {response.status_code} - {response.text}")
            return None
        except Exception as e:
            print(f"Network Exception ({endpoint}): {e}")
            return None

    # --- OCR / Parser ---
    def preprocess_image(self, pil_image):
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def parse_pdf_page(self, page_img):
        processed_img = self.preprocess_image(page_img)
        d = pytesseract.image_to_data(processed_img, lang='chi_sim', output_type=Output.DICT)
        lines = []
        curr_text = []
        last_top = -1
        for i in range(len(d['text'])):
            text = d['text'][i].strip()
            if not text: continue
            top = d['top'][i]
            height = d['height'][i]
            if last_top == -1 or abs(top - last_top) < (height / 2):
                curr_text.append(text)
            else:
                lines.append("".join(curr_text))
                curr_text = [text]
            last_top = top
        if curr_text: lines.append("".join(curr_text))
        return "\n".join(lines)

    def repair_text_with_llm(self, messy_text):
        prompt = "你是一个OCR文档解析专家。请将以下破碎的文本修复为整洁的Markdown格式。只输出内容。"
        payload = {"model": Config.MODEL_PARSER, "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": messy_text}], "temperature": 0.3}
        res = self._api_post("chat/completions", payload)
        return res['choices'][0]['message']['content'] if res else messy_text

    # --- ETL ---
    def smart_split(self, text, chunk_size=800, overlap=100):
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + chunk_size
            if end < text_len:
                look_back = text[max(0, end-50):end]
                for punct in ['。', '；', '！', '?', '\n']:
                    pos = look_back.rfind(punct)
                    if pos != -1:
                        end = max(0, end-50) + pos + 1
                        break
            chunks.append(text[start:end])
            start = end - overlap
            if start >= text_len: break
        return chunks

    def process_etl(self, parsed_data_list, filename="doc"):
        final_chunks = []
        curr_h1 = "无一级标题"
        curr_h2 = ""
        for item in parsed_data_list:
            role = item.get('role', 'BODY')
            text = item.get('text', '').strip()
            page = item.get('page', 0)
            clean_text = text.replace('# ', '').replace('## ', '')
            
            if role == 'H1': curr_h1 = clean_text; curr_h2 = ""
            elif role == 'H2': curr_h2 = clean_text
            elif role == 'BODY' and clean_text:
                header = f"文档：{filename}\n章节：{curr_h1} > {curr_h2}\n正文："
                subs = self.smart_split(clean_text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP) if len(clean_text) > Config.CHUNK_SIZE else [clean_text]
                for sub in subs:
                    full_emb = f"{header}\n{sub}"
                    final_chunks.append({
                        "id": str(uuid.uuid4()),
                        "metadata": {"source": filename, "page": page, "h1": curr_h1, "h2": curr_h2},
                        "content": {"embedding_text": full_emb, "pure_text": sub}
                    })
        return final_chunks

    # --- Vectors & DB ---
    def get_embeddings(self, texts):
        payload = {"model": Config.MODEL_EMBED, "input": texts, "encoding_format": "float"}
        res = self._api_post("embeddings", payload)
        return [d['embedding'] for d in res['data']] if res else None

    def save_vectors_to_db(self, chunks_with_vectors, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for item in chunks_with_vectors:
            vec_blob = np.array(item['vector'], dtype=np.float32).tobytes()
            cursor.execute('''
                INSERT OR REPLACE INTO chunks_full_index 
                (id, doc_title, page_num, chapter_path, embedding_text, pure_text, vector_blob, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item['id'], item['metadata']['source'], item['metadata']['page'],
                f"{item['metadata']['h1']} > {item['metadata']['h2']}",
                item['content']['embedding_text'], item['content']['pure_text'],
                vec_blob, json.dumps(item['metadata'])
            ))
        conn.commit()
        conn.close()

    # --- RAG ---
    def rewrite_query(self, query):
        payload = {"model": Config.MODEL_REWRITE, "messages": [{"role": "user", "content": f"将此用户提问重写为更好的搜索引擎检索词: {query}"}]}
        res = self._api_post("chat/completions", payload)
        return res['choices'][0]['message']['content'] if res else query

    def vector_search(self, query_text, top_k=20):
        q_vecs = self.get_embeddings([query_text])
        if not q_vecs: return []
        q_vec = np.array(q_vecs[0], dtype=np.float32)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT pure_text, vector_blob, doc_title FROM chunks_full_index")
        rows = cursor.fetchall()
        conn.close()

        results = []
        for text, blob, title in rows:
            db_vec = np.frombuffer(blob, dtype=np.float32)
            if q_vec.shape != db_vec.shape: continue
            score = np.dot(q_vec, db_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(db_vec))
            results.append({"content": text, "score": float(score), "title": title})
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def rerank(self, query, docs, top_n=5):
        if not docs: return []
        payload = {"model": Config.MODEL_RERANK, "query": query, "documents": [d['content'] for d in docs], "top_n": top_n}
        res = self._api_post("rerank", payload)
        if res and 'results' in res:
            ranked = []
            for item in res['results']:
                idx = item['index']
                docs[idx]['rerank_score'] = item['relevance_score']
                ranked.append(docs[idx])
            return ranked
        return docs[:top_n]

    def generate_answer(self, query, context_docs):
        ctx_str = "\n".join([f"资料{i+1}: {d['content']}" for i, d in enumerate(context_docs)])
        prompt = f"基于以下资料回答问题:\n{ctx_str}\n\n问题: {query}"
        payload = {"model": Config.MODEL_GEN, "messages": [{"role": "user", "content": prompt}]}
        res = self._api_post("chat/completions", payload)
        return res['choices'][0]['message']['content'] if res else "生成失败"

# ==============================================================================
#  3. 异步工作线程
# ==============================================================================

class PipelineThread(QThread):
    """
    统一流水线线程：根据传入的 flag 执行不同的步骤，严格遵守文件命名约定。
    """
    log_signal = pyqtSignal(str)
    step1_done = pyqtSignal(object) # 传递解析后的数据
    step2_done = pyqtSignal()

    def __init__(self, mode, file_paths):
        super().__init__()
        self.mode = mode # 'parse' or 'vectorize'
        self.paths = file_paths # 字典，包含 input_pdf, step1_json, step2_json, step3_json, step3_db
        self.engine = RAGCoreService()

    def run(self):
        if self.mode == 'parse':
            self._run_parsing()
        elif self.mode == 'vectorize':
            self._run_vectorize()

    def _run_parsing(self):
        try:
            pdf_path = self.paths['input_pdf']
            json_path = self.paths['step1_json']
            
            self.log_signal.emit(f"📄 [Step 1] 开始解析: {os.path.basename(pdf_path)}")
            results = []
            with pdfplumber.open(pdf_path) as pdf:
                total = len(pdf.pages)
                for i, page in enumerate(pdf.pages):
                    self.log_signal.emit(f"正在 OCR 第 {i+1}/{total} 页...")
                    img = page.to_image(resolution=300).original
                    raw = self.engine.parse_pdf_page(img)
                    fixed = self.engine.repair_text_with_llm(raw)
                    for line in fixed.split('\n'):
                        if not line.strip(): continue
                        role = "H1" if line.startswith('# ') else "H2" if line.startswith('## ') else "BODY"
                        results.append({"role": role, "text": line, "page": i+1})
            
            # 保存 Step 1 JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.log_signal.emit(f"✅ Step 1 完成！已生成: {os.path.basename(json_path)}")
            self.step1_done.emit(results)
            
        except Exception as e:
            self.log_signal.emit(f"❌ 解析错误: {str(e)}")

    def _run_vectorize(self):
        try:
            # 1. 读取 Step 1 的 JSON
            step1_path = self.paths['step1_json']
            step2_path = self.paths['step2_json'] # ..._vector_ready.json
            final_json_path = self.paths['step3_json'] # ..._final_vectors.json
            db_path = self.paths['step3_db']    # ..._rag_production.db
            
            with open(step1_path, 'r', encoding='utf-8') as f:
                parsed_data = json.load(f)

            # 2. ETL 处理
            self.log_signal.emit("🔨 [Step 2] 开始 ETL 切片...")
            filename = os.path.basename(self.paths['input_pdf'])
            chunks = self.engine.process_etl(parsed_data, filename)
            
            # 保存 Step 2 JSON (Vector Ready)
            with open(step2_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            self.log_signal.emit(f"✅ Step 2 完成！已生成: {os.path.basename(step2_path)}")

            # 3. 向量化 & 入库
            self.log_signal.emit(f"🧠 [Step 2.5] 开始向量化 ({len(chunks)} 块)...")
            self.engine.init_db(db_path) # 初始化特定的 DB 文件
            
            batch_size = 8
            total = len(chunks)
            for i in range(0, total, batch_size):
                batch = chunks[i : i+batch_size]
                texts = [c['content']['embedding_text'] for c in batch]
                vectors = self.engine.get_embeddings(texts)
                if vectors:
                    for j, vec in enumerate(vectors):
                        batch[j]['vector'] = vec
                    self.engine.save_vectors_to_db(batch, db_path)
                    self.log_signal.emit(f"  -> 进度: {min(i+batch_size, total)}/{total}")
            
            # 保存 Step 3 JSON (Final with Vectors - Optional backup)
            with open(final_json_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2) # 这里包含 vector 数组，文件会很大
            
            self.log_signal.emit(f"🎉 全部完成！")
            self.log_signal.emit(f"📂 生成 DB: {os.path.basename(db_path)}")
            self.step2_done.emit()

        except Exception as e:
            self.log_signal.emit(f"❌ 向量化错误: {str(e)}")

class RAGQueryThread(QThread):
    result_signal = pyqtSignal(str, str)

    def __init__(self, query, db_path, top_k):
        super().__init__()
        self.query = query
        self.engine = RAGCoreService(db_path)
        self.top_k = top_k

    def run(self):
        log = []
        try:
            log.append(f"📚 连接知识库: {os.path.basename(self.engine.db_path)}")
            log.append(f"Q: {self.query}")
            rw_q = self.engine.rewrite_query(self.query)
            log.append(f"🔄 重写: {rw_q}")
            
            docs = self.engine.vector_search(rw_q, top_k=20)
            log.append(f"🔍 召回: {len(docs)} 条")
            
            ranked = self.engine.rerank(rw_q, docs, top_n=self.top_k)
            log.append(f"⚖️  精排: {len(ranked)} 条")
            
            ans = self.engine.generate_answer(self.query, ranked)
            self.result_signal.emit(ans, "\n".join(log))
        except Exception as e:
            self.result_signal.emit(f"Error: {str(e)}", str(e))

# ==============================================================================
#  4. 主界面
# ==============================================================================

class UnifiedWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Super Light RAG - 全流程流水线")
        self.resize(1200, 800)
        self.file_paths = {} # 存储当前 PDF 的整套路径链
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        title = QLabel("RAG 生产流水线控制台")
        title.setFont(QFont("微软雅黑", 14, QFont.Bold))
        layout.addWidget(title)

        self.tabs = QTabWidget()
        self.tab_ingest = QWidget()
        self.tab_chat = QWidget()
        
        self.tabs.addTab(self.tab_ingest, "🏭 A. 数据生产线")
        self.tabs.addTab(self.tab_chat, "💬 B. 检索问答")
        
        self._setup_ingest_tab()
        self._setup_chat_tab()
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def _setup_ingest_tab(self):
        layout = QHBoxLayout(self.tab_ingest)
        left = QWidget(); l_layout = QVBoxLayout(left)
        
        # 文件选择
        f_grp = QGroupBox("1. 输入源")
        f_lay = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("请选择 PDF 文件...")
        btn = QPushButton("📂 选择 PDF"); btn.clicked.connect(self.browse_pdf)
        f_lay.addWidget(self.path_edit); f_lay.addWidget(btn)
        f_grp.setLayout(f_lay)
        
        # 文件链预览
        p_grp = QGroupBox("2. 自动生成的文件链 (Lineage)")
        p_lay = QVBoxLayout()
        self.lbl_step1 = QLabel("Step 1 JSON: 等待选择...")
        self.lbl_step2 = QLabel("Step 2 JSON: 等待选择...")
        self.lbl_db = QLabel("Step 3 DB: 等待选择...")
        self.lbl_step1.setStyleSheet("color: gray"); self.lbl_step2.setStyleSheet("color: gray"); self.lbl_db.setStyleSheet("color: gray")
        p_lay.addWidget(self.lbl_step1); p_lay.addWidget(self.lbl_step2); p_lay.addWidget(self.lbl_db)
        p_grp.setLayout(p_lay)

        # 动作
        act_grp = QGroupBox("3. 执行")
        a_lay = QVBoxLayout()
        self.btn_parse = QPushButton("▶ 启动 Step 1 (Parsing)"); self.btn_parse.clicked.connect(self.run_step1); self.btn_parse.setEnabled(False)
        self.btn_vect = QPushButton("▶ 启动 Step 2 & 2.5 (Vectorize)"); self.btn_vect.clicked.connect(self.run_step2); self.btn_vect.setEnabled(False)
        a_lay.addWidget(self.btn_parse); a_lay.addWidget(self.btn_vect)
        act_grp.setLayout(a_lay)
        
        # 日志
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background: #222; color: #0f0; font-family: Consolas;")
        
        l_layout.addWidget(f_grp); l_layout.addWidget(p_grp); l_layout.addWidget(act_grp); l_layout.addWidget(self.log_box)
        
        right = QWidget(); r_lay = QVBoxLayout(right)
        self.tree = QTreeWidget(); self.tree.setHeaderLabels(["Role", "Text", "Page"])
        self.tree.setColumnWidth(0, 60); self.tree.setColumnWidth(2, 40)
        r_lay.addWidget(QLabel("中间结果预览:")); r_lay.addWidget(self.tree)
        
        splitter = QSplitter(Qt.Horizontal); splitter.addWidget(left); splitter.addWidget(right); splitter.setSizes([450, 750])
        layout.addWidget(splitter)

    def _setup_chat_tab(self):
        layout = QVBoxLayout(self.tab_chat)
        
        d_lay = QHBoxLayout()
        self.db_select_lbl = QLabel("当前挂载 DB: 未就绪")
        self.db_select_lbl.setStyleSheet("color: red; font-weight: bold;")
        d_lay.addWidget(QLabel("知识库状态:")); d_lay.addWidget(self.db_select_lbl)
        layout.addLayout(d_lay)

        self.chat_history = QTextEdit(); self.chat_history.setReadOnly(True)
        layout.addWidget(self.chat_history)
        
        i_lay = QHBoxLayout()
        self.query_edit = QLineEdit(); self.query_edit.returnPressed.connect(self.run_query)
        self.spin_k = QSpinBox(); self.spin_k.setValue(5); self.spin_k.setPrefix("Top-")
        btn = QPushButton("发送"); btn.clicked.connect(self.run_query)
        i_lay.addWidget(self.query_edit); i_lay.addWidget(self.spin_k); i_lay.addWidget(btn)
        layout.addLayout(i_lay)
        
        self.rag_log = QTextEdit(); self.rag_log.setMaximumHeight(100); 
        self.rag_log.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(QLabel("调试日志:")); layout.addWidget(self.rag_log)

    # --- Logic ---

    def browse_pdf(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select PDF", "", "PDF (*.pdf)")
        if f:
            self.path_edit.setText(f)
            self.generate_filenames(f)
            self.update_status()

    def generate_filenames(self, pdf_path):
        """核心逻辑：生成文件链"""
        dirname = os.path.dirname(pdf_path)
        basename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # 1. Step 1 Export
        s1 = os.path.join(dirname, f"{basename}_day1_export.json")
        
        # 2. Step 2 Ready (叠加命名)
        # 注意 Windows 路径长度限制，如果名字太长需要截断，但为了完全符合你的需求，这里先完整拼接
        s2 = os.path.join(dirname, f"{basename}_day1_export_vector_ready.json")
        
        # 3. Step 3 Final & DB (继续叠加)
        s3_json = os.path.join(dirname, f"{basename}_day1_export_vector_ready_final_vectors.json")
        s3_db = os.path.join(dirname, f"{basename}_day1_export_vector_ready_rag_production.db")
        
        self.file_paths = {
            "input_pdf": pdf_path,
            "step1_json": s1,
            "step2_json": s2,
            "step3_json": s3_json,
            "step3_db": s3_db
        }
        
        # 更新 UI 显示
        self.lbl_step1.setText(f"Step 1: ...{s1[-40:]}")
        self.lbl_step2.setText(f"Step 2: ...{s2[-40:]}")
        self.lbl_db.setText(f"Step 3: ...{s3_db[-40:]}")

    def update_status(self):
        # 检查文件是否存在来决定按钮状态
        has_pdf = os.path.exists(self.file_paths.get('input_pdf', ''))
        has_s1 = os.path.exists(self.file_paths.get('step1_json', ''))
        has_db = os.path.exists(self.file_paths.get('step3_db', ''))
        
        self.btn_parse.setEnabled(has_pdf)
        self.btn_vect.setEnabled(has_s1)
        
        if has_s1:
            self.lbl_step1.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.lbl_step1.setStyleSheet("color: gray;")
            
        if has_db:
            self.lbl_db.setStyleSheet("color: green; font-weight: bold;")
            self.db_select_lbl.setText(f"✅ 就绪: {os.path.basename(self.file_paths['step3_db'])}")
            self.db_select_lbl.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.db_select_lbl.setText("❌ 未找到数据库 (请先运行生产线)")
            self.db_select_lbl.setStyleSheet("color: red;")

    def log(self, msg):
        self.log_box.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def run_step1(self):
        self.tree.clear()
        self.btn_parse.setEnabled(False)
        self.thread = PipelineThread('parse', self.file_paths)
        self.thread.log_signal.connect(self.log)
        self.thread.step1_done.connect(self.on_step1_finished)
        self.thread.start()

    def on_step1_finished(self, results):
        self.update_status() # 刷新按钮状态
        for item in results:
            node = QTreeWidgetItem([item['role'], item['text'], str(item['page'])])
            if item['role'] == 'H1': node.setBackground(0, QColor("#3498db"))
            self.tree.addTopLevelItem(node)
        QMessageBox.information(self, "Step 1 Done", "解析完成，Json已生成。\n现在可以点击 Step 2。")

    def run_step2(self):
        self.btn_vect.setEnabled(False)
        self.thread = PipelineThread('vectorize', self.file_paths)
        self.thread.log_signal.connect(self.log)
        self.thread.step2_done.connect(self.on_step2_finished)
        self.thread.start()

    def on_step2_finished(self):
        self.update_status()
        QMessageBox.information(self, "All Done", "向量化完成！数据库已就绪。\n请切换到 'B. 检索问答' 标签页进行测试。")

    def run_query(self):
        q = self.query_edit.text().strip()
        db = self.file_paths.get('step3_db')
        if not q or not db or not os.path.exists(db):
            QMessageBox.warning(self, "Error", "没有问题 或 数据库未生成")
            return
        
        self.chat_history.append(f"\n🙋‍♂️ {q}")
        self.query_edit.clear()
        
        self.q_thread = RAGQueryThread(q, db, self.spin_k.value())
        self.q_thread.result_signal.connect(lambda ans, log: (self.chat_history.append(f"🤖 {ans}"), self.rag_log.setText(log)))
        self.q_thread.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UnifiedWindow()
    window.show()
    sys.exit(app.exec_())