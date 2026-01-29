# 代码结构说明 (Code Structure)

## 1. 目录概览
```text
.
├── config.py                # 全局环境配置 (Tesseract 路径、RAG 参数)
├── image_preprocessing.py   # 图像处理工具集 (OpenCV 预处理流水线)
├── pdf_structure_parser.py  # 基础解析引擎 (PDF 物理布局分析)
├── pdf_hybrid_engine.py     # 混合动力引擎 (OCR + LLM 语义修复)
├── day1_ui.py               # 第一阶段：可视化解析与导出工具 (PyQt5)
├── day2_etl.py              # 第二阶段：结构化 ETL 与知识切片逻辑
├── bge_light_rag.py         # 第三阶段：向量化入库控制台 (Tkinter + 多线程)
├── RAG_lightQueryFront.py   # 第四阶段：全链路问答交互终端 (Debug 专用)
├── requirements.txt         # 项目依赖声明
├── rag_production.db        # 生产向量数据库 (SQLite)
└── *.json                   # 中间态数据备份 (Day 1-3)
```

## 2. 核心模块详解

### 2.1 数据获取层
- **`image_preprocessing.py`**: 负责将 PDF 页面图像转化为适合 OCR 的二值化图像。采用 `medianBlur` 降噪和 `OTSU` 阈值分割。
- **`pdf_structure_parser.py`**: 系统底层解析器。它不仅提取文本，还通过 `pytesseract.image_to_data` 捕获每个字符的坐标 (top, left, width, height)，并基于此推断逻辑行。

### 2.2 智能加工层
- **`pdf_hybrid_engine.py`**: 解析能力的升华。它将物理行发送给 LLM，利用大模型的推理能力修复由于 OCR 错误或 PDF 损坏导致的语义断裂，输出结构化的 Markdown。
- **`day2_etl.py`**: 负责“语义桥接”。它解析 Markdown 标签，提取 H1 和 H2，并在切片时将这些上下文“染色”到正文中，确保每个 Chunk 都携带完整的文档路径。

### 2.3 存储与检索引擎
- **`bge_light_rag.py`**: 向量化流水线控制器。管理 API 调用频率、多线程并发以及 SQLite 事务，确保大数据量下入库的稳定性。
- **`RAG_lightQueryFront.py`**: 集大成者。它实现了复杂的检索逻辑：
  - **Query Rewrite**: 解决用户输入模糊问题。
  - **Vector Search**: 基于欧氏距离/余弦相似度的初步筛选。
  - **Rerank**: 使用 `bge-reranker` 解决向量搜索“看似相关实则无关”的问题。

## 3. 设计模式应用
1. **策略模式 (Strategy)**: 在 `bge_light_rag.py` 中，通过 `PROVIDERS` 配置，系统可以灵活切换内网 BGE 服务与云端 API 服务。
2. **观察者/信号模式 (Signals)**: UI 层通过 PyQt 信号与工作线程通信，实现了耗时解析过程中的非阻塞进度更新。
3. **单例/全局配置**: `config.py` 作为全局状态中心，规范了跨模块的参数调用。
