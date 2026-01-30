import os
import sys
import json
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

# --- 核心：动态挂载 RagLight 引擎 ---
RAG_ENGINE_PATH = r"C:\RagLight"
if RAG_ENGINE_PATH not in sys.path:
    sys.path.append(RAG_ENGINE_PATH)

# 尝试导入引擎，如果失败则报错（确保 C:\RagLight 下有 RAG_light_engine.py）
try:
    from RAG_light_engine import RAGCoreService, Config
except ImportError:
    print("CRITICAL ERROR: Could not import RAG_light_engine. Please check C:\\RagLight path.")
    RAGCoreService = None
    Config = None

router = APIRouter()

# --- 数据模型 ---
class ProcessStep1Request(BaseModel):
    filename: str

class ProcessStep2Request(BaseModel):
    filename: str

class ChatRequest(BaseModel):
    filename: str
    db_selection: Optional[str] = None  # 新增：用户手动选择的DB文件名
    query: str
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str
    logs: str
    docs: List[dict] = []

# --- 工具函数：文件名生成器 (保持与 Desktop 一致) ---
def get_file_chain(original_filename):
    """
    根据原始文件名生成完整的工序文件路径链
    """
    base_dir = RAG_ENGINE_PATH
    basename = os.path.splitext(original_filename)[0]
    
    # 路径定义
    pdf_path = os.path.join(base_dir, original_filename)
    step1_json = os.path.join(base_dir, f"{basename}_day1_export.json")
    step2_json = os.path.join(base_dir, f"{basename}_day1_export_vector_ready.json")
    step3_db = os.path.join(base_dir, f"{basename}_day1_export_vector_ready_rag_production.db")
    
    return {
        "pdf": pdf_path,
        "step1": step1_json,
        "step2": step2_json,
        "db": step3_db,
        "basename": basename
    }

# --- API 接口 ---

@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """
    1. 上传 PDF 到 C:\RagLight
    """
    if not RAGCoreService:
        raise HTTPException(status_code=500, detail="RAG Engine not loaded")
    
    try:
        save_path = os.path.join(RAG_ENGINE_PATH, file.filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "message": "Upload successful", 
            "filename": file.filename,
            "paths": get_file_chain(file.filename)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run_step1")
def run_step1(req: ProcessStep1Request):
    """
    2. 执行 OCR 和 LLM 解析
    注意：为了避免阻塞主线程太久，FastAPI 会在线程池中运行此同步函数
    """
    paths = get_file_chain(req.filename)
    if not os.path.exists(paths['pdf']):
        raise HTTPException(status_code=404, detail="PDF file not found")

    engine = RAGCoreService()
    results = []
    
    try:
        # 使用 pdfplumber 打开 (需导入 pdfplumber)
        import pdfplumber
        
        with pdfplumber.open(paths['pdf']) as pdf:
            for i, page in enumerate(pdf.pages):
                # OCR
                img = page.to_image(resolution=300).original
                raw_text = engine.parse_pdf_page(img)
                # LLM Fix
                fixed_text = engine.repair_text_with_llm(raw_text)
                
                # Parse Lines
                for line in fixed_text.split('\n'):
                    if not line.strip(): continue
                    role = "H1" if line.startswith('# ') else "H2" if line.startswith('## ') else "BODY"
                    results.append({"role": role, "text": line, "page": i+1})

        # Save JSON
        with open(paths['step1'], 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        return {"status": "success", "data": results, "next_file": os.path.basename(paths['step1'])}

    except Exception as e:
        print(f"Step 1 Error: {e}")
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")

@router.post("/run_step2")
def run_step2(req: ProcessStep2Request):
    """
    3. 执行 ETL 和 向量化入库
    """
    paths = get_file_chain(req.filename)
    if not os.path.exists(paths['step1']):
        raise HTTPException(status_code=400, detail="Step 1 data not found. Please run parsing first.")

    engine = RAGCoreService()
    
    try:
        # Load Step 1
        with open(paths['step1'], 'r', encoding='utf-8') as f:
            parsed_data = json.load(f)
            
        # ETL
        chunks = engine.process_etl(parsed_data, req.filename)
        
        # Save Step 2 JSON
        with open(paths['step2'], 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
        # Vectorize & DB
        engine.init_db(paths['db'])
        
        batch_size = 8
        total = len(chunks)
        
        # 简单循环处理 (Web端暂不使用复杂多线程，保证稳定性)
        for i in range(0, total, batch_size):
            batch = chunks[i : i+batch_size]
            texts = [c['content']['embedding_text'] for c in batch]
            vectors = engine.get_embeddings(texts)
            
            if vectors:
                for j, vec in enumerate(vectors):
                    batch[j]['vector'] = vec
                engine.save_vectors_to_db(batch, paths['db'])
        
        return {
            "status": "success", 
            "db_path": paths['db'],
            "chunk_count": total
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vectorization failed: {str(e)}")

@router.get("/list_dbs")
def list_dbs():
    """
    列出 C:\RagLight 下所有的 .db 文件
    """
    if not os.path.exists(RAG_ENGINE_PATH):
        return {"dbs": []}
    
    try:
        files = [f for f in os.listdir(RAG_ENGINE_PATH) if f.endswith('.db')]
        return {"dbs": files}
    except Exception as e:
        return {"dbs": [], "error": str(e)}

@router.post("/query")
def rag_query(req: ChatRequest):
    """
    4. RAG 问答
    支持通过 db_selection 指定特定库，或者通过 filename 自动推导库
    """
    # 确定数据库路径
    db_path = None
    
    if req.db_selection:
        # 如果用户选择了特定DB，直接使用该路径
        candidate_path = os.path.join(RAG_ENGINE_PATH, req.db_selection)
        if os.path.exists(candidate_path):
            db_path = candidate_path
    
    # 如果没有指定 db_selection 或者文件不存在，尝试回退到 filename 推导
    if not db_path and req.filename and req.filename != "manual_select":
        paths = get_file_chain(req.filename)
        if os.path.exists(paths['db']):
            db_path = paths['db']
            
    if not db_path:
        raise HTTPException(status_code=400, detail="Database file not found. Please select a database or run Step 2.")
        
    engine = RAGCoreService(db_path=db_path)
    logs = []
    
    try:
        # 1. Rewrite
        logs.append(f"Q: {req.query}")
        rw_q = engine.rewrite_query(req.query)
        logs.append(f"Rewrite: {rw_q}")
        
        # 2. Search
        docs = engine.vector_search(rw_q, top_k=20)
        logs.append(f"Recall: {len(docs)} docs from {os.path.basename(db_path)}")
        
        # 3. Rerank
        ranked = engine.rerank(rw_q, docs, top_n=req.top_k)
        logs.append(f"Rerank: Top {len(ranked)}")
        
        # 4. Generate
        ans = engine.generate_answer(req.query, ranked)
        
        return ChatResponse(
            answer=ans,
            logs="\n".join(logs),
            docs=ranked
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")