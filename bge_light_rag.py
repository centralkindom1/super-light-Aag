import os
import json
import sqlite3
import requests
import urllib3
import time
import uuid
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 禁用 SSL 警告 (局域网 API 经常需要)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 1. 配置中心 (保留原结构，但在运行时由GUI动态修改) ---
class RagConfig:
    # 路径配置
    INPUT_JSON = "day2_vector_ready.json" # 默认改为用户要求的 day2
    OUTPUT_DB = "rag_production.db"
    BACKUP_JSON = "day3_final_vectors.json"
    
    # 模式选择: 'intranet' (内网) 或 'silicon' (硅基流动)
    ACTIVE_PROVIDER = 'intranet' 

    PROVIDERS = {
        'intranet': {
            'url': "https://www.siconvaly.com:18080/v1/embeddings",
            'key': "your api key",
            'model': "bge-m3",
            'name': "内网 BGE-M3"
        },
        'silicon': {
            'url': "https://api.siliconflow.cn/v1/embeddings",
            'key': "your api key",
            'model': "BAAI/bge-m3",
            'name': "硅基流动 BGE-M3"
        }
    }

    # 并发性能配置 (默认值)
    MAX_THREADS = 2      
    BATCH_SIZE = 8       
    EMBEDDING_DIM = 1024 

# --- 2. 数据库管理类 (保持不变) ---
class VectorDBManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # 创建表：存储元数据、原文及向量（BLOB存储）
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

    def save_batch(self, results):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for item in results:
            # 将 list 向量转为二进制 bytes 存储，节省空间
            vec_np = np.array(item['vector'], dtype=np.float32)
            vec_blob = vec_np.tobytes()
            
            cursor.execute('''
                INSERT OR REPLACE INTO chunks_full_index 
                (id, doc_title, page_num, chapter_path, embedding_text, pure_text, vector_blob, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item['id'],
                item['metadata']['source'],
                item['metadata']['page'],
                f"{item['metadata']['h1']} > {item['metadata']['h2']}",
                item['content']['embedding_text'],
                item['content']['pure_text'],
                vec_blob,
                json.dumps(item['metadata'])
            ))
        conn.commit()
        conn.close()

# --- 3. 向量化引擎 (保持不变) ---
class EmbeddingEngine:
    def __init__(self, provider_key):
        config = RagConfig.PROVIDERS[provider_key]
        self.url = config['url']
        self.headers = {
            "Authorization": f"Bearer {config['key']}",
            "Content-Type": "application/json"
        }
        self.model = config['model']

    def get_vector_batch(self, texts):
        """调用 API 获取一组文本的向量"""
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }
        
        for attempt in range(3): # 失败重试 3 次
            try:
                response = requests.post(self.url, headers=self.headers, json=payload, verify=False, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    # 提取结果向量，保持输入顺序
                    embeddings = [d['embedding'] for d in data['data']]
                    return embeddings
                else:
                    print(f"[API Error] Status: {response.status_code}, Msg: {response.text}")
            except Exception as e:
                print(f"[Network Error] 正在重试 ({attempt+1}/3): {e}")
                time.sleep(2)
        return None

# --- 4. 核心逻辑封装 (供GUI调用) ---
def run_pipeline_logic():
    """
    运行核心流水线。
    注意：此时 RagConfig 已经被 GUI 更新为用户设定的值。
    """
    print(f"🚀 BGE-M3 RAG 向量化流水线启动")
    print(f"📌 模式: {RagConfig.ACTIVE_PROVIDER}")
    print(f"📂 输入文件: {RagConfig.INPUT_JSON}")
    print(f"⚙️  线程数: {RagConfig.MAX_THREADS}, Batch大小: {RagConfig.BATCH_SIZE}")
    
    # 加载数据
    if not os.path.exists(RagConfig.INPUT_JSON):
        print(f"❌ 错误: 找不到输入文件 {RagConfig.INPUT_JSON}")
        return

    try:
        with open(RagConfig.INPUT_JSON, 'r', encoding='utf-8') as f:
            all_chunks = json.load(f)
    except Exception as e:
        print(f"❌ 读取JSON失败: {e}")
        return
    
    total_chunks = len(all_chunks)
    print(f"📦 已加载 {total_chunks} 个知识块。")

    db = VectorDBManager(RagConfig.OUTPUT_DB)
    engine = EmbeddingEngine(RagConfig.ACTIVE_PROVIDER)

    # 分批次 (Batching)
    batches = [all_chunks[i : i + RagConfig.BATCH_SIZE] for i in range(0, total_chunks, RagConfig.BATCH_SIZE)]
    
    final_processed_data = []
    
    print(f"🔄 开始并发处理...")
    
    # 定义单批处理函数用于多线程
    def process_batch(batch_data):
        texts_to_embed = [item['content']['embedding_text'] for item in batch_data]
        vectors = engine.get_vector_batch(texts_to_embed)
        
        if vectors:
            for i, vec in enumerate(vectors):
                batch_data[i]['vector'] = vec
            return batch_data
        else:
            print(f"⚠️  一批数据({len(batch_data)}条)向量化失败。")
            return []

    # 多线程并行
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=RagConfig.MAX_THREADS) as executor:
        futures = [executor.submit(process_batch, b) for b in batches]
        
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            if res:
                db.save_batch(res)
                final_processed_data.extend(res)
                completed += len(res)
                # 计算进度百分比
                percent = (completed / total_chunks) * 100
                print(f"✅ 进度: {completed}/{total_chunks} ({percent:.1f}%) 已入库...")

    # 保存 JSON 备份
    with open(RagConfig.BACKUP_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_processed_data, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    print("\n" + "="*30)
    print(f"🎉 任务圆满完成!")
    print(f"⏱️  总耗时: {end_time - start_time:.2f} 秒")
    print(f"📂 数据库文件: {RagConfig.OUTPUT_DB}")
    print(f"📝 备份文件: {RagConfig.BACKUP_JSON}")
    print("="*30)

# --- 5. GUI 界面类 ---

class TextRedirector:
    """重定向 stdout 到 Tkinter Text 控件"""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str_val):
        # 使用 after 方法在主线程更新 UI，防止线程冲突
        self.widget.after(0, self._append_text, str_val)

    def _append_text(self, str_val):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, str_val, self.tag)
        self.widget.see(tk.END)
        self.widget.configure(state='disabled')

    def flush(self):
        pass

class RagApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BGE-M3 向量化控制台")
        self.root.geometry("700x600")
        
        # 样式设置
        style = ttk.Style()
        style.configure("TButton", font=("Microsoft YaHei", 10))
        style.configure("TLabel", font=("Microsoft YaHei", 10))
        
        self.create_widgets()
        
        # 重定向输出
        self.original_stdout = sys.stdout
        sys.stdout = TextRedirector(self.console_text)

    def create_widgets(self):
        # 1. 文件选择区域
        file_frame = ttk.LabelFrame(self.root, text="输入设置", padding=10)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(file_frame, text="输入文件:").grid(row=0, column=0, padx=5, sticky="w")
        self.file_path_var = tk.StringVar(value=os.path.abspath("day2_vector_ready.json"))
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        self.file_entry.grid(row=0, column=1, padx=5)
        
        ttk.Button(file_frame, text="选择文件", command=self.browse_file).grid(row=0, column=2, padx=5)

        # 2. 参数设置区域
        settings_frame = ttk.LabelFrame(self.root, text="运行参数", padding=10)
        settings_frame.pack(fill="x", padx=10, pady=5)
        
        # 线程数
        ttk.Label(settings_frame, text="线程数 (2-10):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.thread_var = tk.IntVar(value=2)
        thread_spin = ttk.Spinbox(settings_frame, from_=2, to=10, textvariable=self.thread_var, width=10)
        thread_spin.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Batch数
        ttk.Label(settings_frame, text="Batch大小 (8-20):").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.batch_var = tk.IntVar(value=8)
        batch_spin = ttk.Spinbox(settings_frame, from_=8, to=20, textvariable=self.batch_var, width=10)
        batch_spin.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        # 3. 模型选择区域
        model_frame = ttk.LabelFrame(self.root, text="模型服务商", padding=10)
        model_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(model_frame, text="选择服务商:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # 构建下拉选项
        self.provider_map = {
            "局域网大模型 (Intranet)": "intranet",
            "硅基动力 (SiliconFlow)": "silicon"
        }
        self.provider_var = tk.StringVar(value="局域网大模型 (Intranet)")
        provider_combo = ttk.Combobox(model_frame, textvariable=self.provider_var, 
                                      values=list(self.provider_map.keys()), state="readonly", width=30)
        provider_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # 4. 操作按钮
        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.pack(fill="x", padx=10)
        
        self.start_btn = ttk.Button(btn_frame, text="🚀 开始向量化处理", command=self.start_processing)
        self.start_btn.pack(side="left", fill="x", expand=True, padx=5)
        
        # 5. 控制台输出区域
        console_frame = ttk.LabelFrame(self.root, text="控制台日志", padding=10)
        console_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.console_text = scrolledtext.ScrolledText(console_frame, state='disabled', height=15, 
                                                      font=("Consolas", 9), bg="#f0f0f0")
        self.console_text.pack(fill="both", expand=True)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="选择输入JSON文件",
            filetypes=(("JSON Files", "*.json"), ("All Files", "*.*"))
        )
        if filename:
            self.file_path_var.set(filename)

    def start_processing(self):
        # 锁定按钮防止重复点击
        self.start_btn.config(state="disabled")
        self.console_text.config(state="normal")
        self.console_text.delete(1.0, tk.END)
        self.console_text.config(state="disabled")
        
        # 获取界面参数并更新 Config
        input_file = self.file_path_var.get()
        threads = self.thread_var.get()
        batch = self.batch_var.get()
        provider_display = self.provider_var.get()
        provider_key = self.provider_map.get(provider_display, "intranet")
        
        # 简单的验证
        if not os.path.exists(input_file) and "day2_vector_ready.json" not in input_file: 
            # 如果是默认值但文件不存在，逻辑里会报错，这里先允许通过以便查看逻辑报错
            pass

        # 更新全局配置类
        RagConfig.INPUT_JSON = input_file
        RagConfig.MAX_THREADS = threads
        RagConfig.BATCH_SIZE = batch
        RagConfig.ACTIVE_PROVIDER = provider_key
        
        # 在新线程中运行，防止阻塞 GUI
        threading.Thread(target=self.run_thread, daemon=True).start()

    def run_thread(self):
        try:
            run_pipeline_logic()
        except Exception as e:
            print(f"\n❌ 发生严重错误: {e}")
        finally:
            # 任务结束后恢复按钮状态
            self.root.after(0, lambda: self.start_btn.config(state="normal"))

if __name__ == "__main__":
    root = tk.Tk()
    app = RagApp(root)

    root.mainloop()
