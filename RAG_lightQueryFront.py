import os
import json
import sqlite3
import requests
import urllib3
import time
import threading
import random
import math
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 1. 增强型配置中心
# ==========================================
class RAGConfig:
    # 基础 API 地址 (请确保与内网环境一致)
    BASE_URL = "https://WWW.DEEPSEEK.COM.cn:18080/v1"
    API_KEY = "YOUR API KEY"
    
    # 默认模型标识符
    # 注意：如果后端部署的 ID 不同，需在此修改
    MODEL_REWRITE = "DeepSeek-V3"
    MODEL_GEN = "DeepSeek-R1"
    MODEL_EMBED = "bge-m3"
    MODEL_RERANK = "bge-reranker-v2-m3"

# ==========================================
# 2. 粒子背景引擎 (Google Style)
# ==========================================
class ParticleEffect(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.particles = []
        self.num_particles = 45
        self.width = 1300
        self.height = 850
        self.bind("<Configure>", self.on_resize)
        self.create_particles()
        self.animate()

    def on_resize(self, event):
        self.width = event.width
        self.height = event.height

    def create_particles(self):
        for _ in range(self.num_particles):
            p = {
                "x": random.randint(0, self.width),
                "y": random.randint(0, self.height),
                "vx": random.uniform(-0.6, 0.6),
                "vy": random.uniform(-0.6, 0.6),
                "id": self.create_oval(0, 0, 3, 3, fill="#4285F4", outline="")
            }
            self.particles.append(p)

    def animate(self):
        self.delete("line")
        for i, p in enumerate(self.particles):
            p["x"] += p["vx"]; p["y"] += p["vy"]
            if p["x"] <= 0 or p["x"] >= self.width: p["vx"] *= -1
            if p["y"] <= 0 or p["y"] >= self.height: p["vy"] *= -1
            self.coords(p["id"], p["x"]-1.5, p["y"]-1.5, p["x"]+1.5, p["y"]+1.5)
            for p2 in self.particles[i+1:]:
                dist = math.sqrt((p["x"]-p2["x"])**2 + (p["y"]-p2["y"])**2)
                if dist < 140:
                    alpha = int(220 * (1 - dist/140))
                    color = f"#{alpha:02x}85F4"
                    self.create_line(p["x"], p["y"], p2["x"], p2["y"], fill=color, tags="line")
        self.after(35, self.animate)

# ==========================================
# 3. 增强型后端引擎 (含深度日志)
# ==========================================
class RAGBackend:
    def __init__(self, log_func):
        self.log = log_func
        self.db_path = None

    def _post(self, endpoint, payload):
        """核心请求器：带详细错误捕获"""
        url = f"{RAGConfig.BASE_URL}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {RAGConfig.API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            self.log(f"📡 Requesting: {endpoint} | Model: {payload.get('model')}")
            res = requests.post(url, headers=headers, json=payload, verify=False, timeout=60)
            
            if res.status_code == 200:
                data = res.json()
                # 检查是否包含预期的结果字段
                if 'choices' in data or 'data' in data or 'results' in data:
                    return data
                else:
                    self.log(f"⚠️  响应结构异常: {json.dumps(data)[:200]}...")
                    return None
            else:
                self.log(f"❌ API 拒绝 (Status {res.status_code}): {res.text}")
                return None
        except Exception as e:
            self.log(f"💥 网络层异常: {str(e)}")
            return None

    def rewrite(self, query, model):
        self.log(f"🔄 [Step 1] 语义重写中...")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": f"请将用户提问重写为更具描述性的检索词: {query}"}],
            "temperature": 0.3
        }
        res = self._post("chat/completions", payload)
        if res and 'choices' in res:
            new_q = res['choices'][0]['message']['content'].strip()
            self.log(f"✅ 重写结果: {new_q[:30]}...")
            return new_q
        return query

    def search(self, query, top_k):
        self.log(f"🔎 [Step 2] 向量召回 (Model: {RAGConfig.MODEL_EMBED})...")
        emb_res = self._post("embeddings", {"model": RAGConfig.MODEL_EMBED, "input": [query]})
        if not emb_res: return []
        
        q_vec = np.array(emb_res['data'][0]['embedding'], dtype=np.float32)

        if not self.db_path or not os.path.exists(self.db_path):
            self.log("❌ 召回中断: 数据库未挂载或文件不存在")
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT pure_text, vector_blob, doc_title FROM chunks_full_index")
        rows = cursor.fetchall()
        
        results = []
        for text, v_blob, title in rows:
            db_vec = np.frombuffer(v_blob, dtype=np.float32)
            # 维度校验
            if q_vec.shape != db_vec.shape: continue
            score = np.dot(q_vec, db_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(db_vec))
            results.append({"content": text, "score": float(score), "title": title})
        
        conn.close()
        results.sort(key=lambda x: x['score'], reverse=True)
        self.log(f"✅ 召回完成，库内匹配最高分: {results[0]['score']:.4f}" if results else "⚠️  库内无匹配")
        return results[:20]

    def rerank(self, query, docs, n):
        self.log(f"⚖️  [Step 3] BGE Reranker 重排中...")
        if not docs: return []
        payload = {
            "model": RAGConfig.MODEL_RERANK,
            "query": query,
            "documents": [d['content'] for d in docs],
            "top_n": n
        }
        res = self._post("rerank", payload)
        if not res: return docs[:n]
        return [docs[item['index']] for item in res['results']]

    def ask(self, query, context, model):
        self.log(f"💬 [Step 4] 大模型最终生成 ({model})...")
        ctx_text = "\n".join([f"资料{i+1}: {d['content']}" for i, d in enumerate(context)])
        prompt = f"请结合资料回答问题。\n\n资料：\n{ctx_text}\n\n提问：{query}"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5
        }
        res = self._post("chat/completions", payload)
        
        if res and 'choices' in res:
            content = res['choices'][0]['message'].get('content', "")
            if not content:
                self.log("⚠️  API 返回成功但内容为空")
                return "API 返回内容为空，请检查模型状态。"
            return content
        return "生成失败 (详情见右侧日志)"

# ==========================================
# 4. 工业化调试窗体 UI
# ==========================================
class IndustrialRAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG 全链路工业级深度调试平台")
        self.root.geometry("1300x900")
        self.root.attributes('-alpha', 0.96)
        
        self.backend = RAGBackend(self.write_log)
        self._setup_style()
        self._setup_layout()

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#F0F2F5")
        style.configure("TLabel", background="#F0F2F5", font=("微软雅黑", 9))

    def _setup_layout(self):
        # 粒子层
        self.bg_canvas = ParticleEffect(self.root, highlightthickness=0, bg="#F0F2F5")
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        # 主内容容器
        main_container = tk.Frame(self.root, bg="", highlightthickness=0)
        main_container.place(relx=0.02, rely=0.02, relwidth=0.96, relheight=0.92)
        
        self.paned = tk.PanedWindow(main_container, orient=tk.HORIZONTAL, bg="#CCCCCC", sashwidth=4)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # --- 左侧: 配置 + 聊天 ---
        left_frame = tk.Frame(self.paned, bg="#F7F9FC")
        self.paned.add(left_frame, width=780)

        # 配置面板
        cfg_box = ttk.LabelFrame(left_frame, text="RAG 链路控制参数")
        cfg_box.pack(fill=tk.X, padx=10, pady=5)

        # 数据库挂载控件
        db_frame = ttk.Frame(cfg_box)
        db_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(db_frame, text="数据库挂载:").pack(side=tk.LEFT)
        self.db_path_var = tk.StringVar(value="[ 未挂载 ]")
        ttk.Entry(db_frame, textvariable=self.db_path_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(db_frame, text="浏览库文件", command=self.mount_db).pack(side=tk.LEFT)

        # 模型与K值设置
        param_frame = ttk.Frame(cfg_box)
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(param_frame, text="重写模型:").grid(row=0, column=0)
        self.rewrite_ui = ttk.Combobox(param_frame, values=["DeepSeek-V3", "deepseek-chat"], width=15)
        self.rewrite_ui.set("DeepSeek-V3")
        self.rewrite_ui.grid(row=0, column=1, padx=5)

        ttk.Label(param_frame, text="生成模型:").grid(row=0, column=2)
        self.gen_ui = ttk.Combobox(param_frame, values=["DeepSeek-R1", "DeepSeek-V3"], width=15)
        self.gen_ui.set("DeepSeek-R1")
        self.gen_ui.grid(row=0, column=3, padx=5)

        ttk.Label(param_frame, text="Top-K:").grid(row=0, column=4)
        self.top_k_ui = ttk.Spinbox(param_frame, from_=1, to=15, width=5)
        self.top_k_ui.set(5)
        self.top_k_ui.grid(row=0, column=5, padx=5)

        # 聊天区域
        self.chat_area = scrolledtext.ScrolledText(left_frame, font=("微软雅黑", 11), bg="white", borderwidth=0)
        self.chat_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- 右侧: 提问 + 深度日志 ---
        right_frame = tk.Frame(self.paned, bg="#F7F9FC")
        self.paned.add(right_frame)

        # 提问区
        q_box = ttk.LabelFrame(right_frame, text="用户提问 (Prompt)")
        q_box.pack(fill=tk.X, padx=10, pady=5)
        self.q_input = tk.Text(q_box, height=8, font=("Consolas", 10), bg="#FFFFFF")
        self.q_input.pack(fill=tk.X, padx=5, pady=5)
        self.q_input.insert(tk.END, "请根据挂载的文档回答：行李损坏如何申请赔偿？")
        
        self.run_btn = tk.Button(q_box, text="⚡ 发送请求 (全链路监控)", bg="#4285F4", fg="white", 
                                 font=("微软雅黑", 10, "bold"), command=self.run_pipeline)
        self.run_btn.pack(fill=tk.X, padx=5, pady=5)

        # 日志区
        log_box = ttk.LabelFrame(right_frame, text="深度交互日志 (API & Logic)")
        log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_area = scrolledtext.ScrolledText(log_box, bg="#1E1E1E", fg="#DCDCDC", font=("Consolas", 9))
        self.log_area.pack(fill=tk.BOTH, expand=True)

        # 底部透明度调节
        bottom_bar = tk.Frame(self.root, bg="#E0E0E0")
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(bottom_bar, text="窗体透明度控制:").pack(side=tk.LEFT, padx=10)
        self.alpha_scale = ttk.Scale(bottom_bar, from_=0.4, to=1.0, value=0.96, command=lambda v: self.root.attributes('-alpha', float(v)))
        self.alpha_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)

    def write_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        def _append():
            self.log_area.insert(tk.END, f"[{ts}] {msg}\n")
            self.log_area.see(tk.END)
        self.root.after(0, _append)

    def mount_db(self):
        p = filedialog.askopenfilename(filetypes=[("SQLite Database", "*.db")])
        if p:
            self.db_path_var.set(p)
            self.backend.db_path = p
            self.write_log(f"✅ 成功挂载数据库: {os.path.basename(p)}")

    def run_pipeline(self):
        query = self.q_input.get("1.0", tk.END).strip()
        if not query: return
        self.run_btn.config(state=tk.DISABLED, text="处理中...")
        self.chat_area.delete("1.0", tk.END)
        self.chat_area.insert(tk.END, "🚀 RAG 链路已激活，正在检索中...\n")
        threading.Thread(target=self._worker, args=(query,), daemon=True).start()

    def _worker(self, query):
        try:
            # 1. 重写
            q_rewrite = self.backend.rewrite(query, self.rewrite_ui.get())
            # 2. 召回
            raw_docs = self.backend.search(q_rewrite, 20)
            if not raw_docs:
                self.root.after(0, lambda: self.chat_area.insert(tk.END, "❌ 召回阶段未找到相关内容，请检查 DB 是否正确向量化。"))
            else:
                # 3. 重排
                n = int(self.top_k_ui.get())
                final_docs = self.backend.rerank(q_rewrite, raw_docs, n)
                # 4. 生成
                ans = self.backend.ask(query, final_docs, self.gen_ui.get())
                self.root.after(0, lambda: self.chat_area.delete("1.0", tk.END))
                self.root.after(0, lambda: self.chat_area.insert(tk.END, ans))
            
            self.write_log("✨ 全链路任务执行完毕。")
        except Exception as e:
            self.write_log(f"💥 链路核心崩溃: {str(e)}")
        finally:
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL, text="⚡ 发送请求 (全链路监控)"))

if __name__ == "__main__":
    root = tk.Tk()
    app = IndustrialRAGApp(root)
    root.mainloop()