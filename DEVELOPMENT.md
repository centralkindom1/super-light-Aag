# 开发与架构深度文档 (Development & Architecture)

## 1. 软件需求规格 (SRS)
*详细内容请参阅独立文档：[SRS.md](./SRS.md)*

## 2. 架构设计 (Google 工程师视角)

### 2.1 整体拓扑
系统采用典型的 **四层解耦架构**：
1. **数据接入层 (Ingestion)**：`pdfplumber` + `Tesseract` 物理提取。
2. **清洗转换层 (ETL)**：LLM 辅助语义对齐 + 结构感知切片。
3. **索引存储层 (Indexing)**：SQLite + Numpy 向量持久化。
4. **业务检索层 (Retrieval)**：Query Rewrite -> Vector Search -> Rerank -> Generation。

### 2.2 核心设计哲学
- **"Clean Data > Smart Model"**：通过 `pdf_hybrid_engine.py` 中的 LLM 修复逻辑，在入口处解决数据污染问题。
- **结构化上下文注入**：将文档路径（如 `安全规章 > 锂电池运输`）注入向量块，使嵌入模型能捕捉到层级语义。

## 3. 关键算法与流程

### 3.1 物理行合并算法 (Spatial Merging)
在 `pdf_structure_parser.py` 中，通过判断文本块的 `top` 坐标偏差是否小于 `height/2` 来实现碎词聚合成行，解决了 OCR 识别后文字断裂的问题。

### 3.2 智能切片 (Smart Split)
```python
# 核心逻辑：带重叠的滑动窗口 + 标点回溯
while start < text_len:
    # ... 在窗口末尾向前寻找 [。；！] 确保句子完整性
    # ... 维护 CHUNK_OVERLAP 防止语义截断
```

## 4. 关键数据结构

### 4.1 `DocumentLine` (内存对象)
```python
{
    "text": str,      # 文本内容
    "role": str,      # 角色: H1, H2, BODY
    "page_num": int,  # 页码
    "is_ocr": bool    # 是否来自 OCR
}
```

### 4.2 向量数据库表结构
```sql
CREATE TABLE chunks_full_index (
    id TEXT PRIMARY KEY,
    doc_title TEXT,
    chapter_path TEXT,  -- 存储 H1 > H2 路径
    vector_blob BLOB,   -- 1024维浮点向量
    pure_text TEXT      -- 原始正文用于生成
);
```

## 5. 可视化架构与流程图
*更多流程图、时序图及拓扑图请参阅：[DIAGRAMS.md](./DIAGRAMS.md)*

## 6. 代码组织与模块职责
*详细的文件级说明请参阅：[CODE_STRUCTURE.md](./CODE_STRUCTURE.md)*

## 7. 安全性分析
1. **API 安全**：当前 API Key 采用硬编码方式，存在泄露风险。*改进建议：迁移至 `.env` 文件。*
2. **数据隐私**：系统支持局域网模型部署，确保敏感 PDF 文档不流向公网。
3. **输入校验**：前端对 Query 长度及非法字符进行了初步过滤。

## 8. 可扩展性与性能优化
- **性能瓶颈**：当前向量检索采用 `Numpy` 线性扫描，在百万级数据下会变慢。*建议：引入 FAISS 或 HNSW 索引。*
- **可扩展性**：`RagConfig` 类在多个文件中重复定义，不利于集中管理。*建议：整合为单例模式。*
- **并发处理**：`bge_light_rag.py` 已实现 `ThreadPoolExecutor` 并发推送向量，显著提升了入库速度。

## 9. Google 软件工程师综合评价

### 核心亮点
1. **Pipeline 鲁棒性**：通过引入 LLM-based Cleanup 阶段，在数据源头通过语义对齐解决了 OCR 常见的错别字和布局碎片问题，这是目前工业级 RAG 系统的最佳实践。
2. **结构化上下文**：系统不仅存储文本，还维护了 `chapter_path`，这种“结构感知”的索引方式极大地缓解了长文档检索中常见的“上下文丢失”难题。
3. **闭环调试能力**：`RAG_lightQueryFront.py` 提供的深度日志功能对于算法调优至关重要，体现了开发者对 RAG 不确定性的深刻理解。

### 改进空间
1. **配置管理**：建议使用 `pydantic-settings` 或 `.env` 替换各文件中的硬编码常量，提升生产环境安全性。
2. **检索效率**：目前基于 Numpy 的点积计算属于 $O(N)$ 复杂度，对于海量知识库建议接入向量数据库（如 Milvus）或近似最近邻（ANN）索引。
3. **架构统一性**：由于项目跨越多个开发阶段，UI 框架在 PyQt5 和 Tkinter 之间切换，建议未来统一技术栈。

## 10. 总结与建议
本系统在 **RAG 精度控制** 上做得非常出色，特别是对 PDF 层级的保留。后续建议重点提升 **工程标准性**，包括引入虚拟环境管理、单元测试覆盖以及更规范的分布式任务调度机制。
