# 软件使用手册 (User Guide)

本手册指导您如何从零开始运行“视觉语义 RAG 系统”。

## 1. 环境准备

### 1.1 Python 环境
建议使用 Python 3.9+。安装依赖：
```bash
pip install -r requirements.txt
```

### 1.2 OCR 引擎安装
1. 下载并安装 [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki)。
2. 在 `config.py` 中修改 `TESSERACT_CMD` 为您的安装路径（例如 `C:\Program Files\Tesseract-OCR\tesseract.exe`）。
3. 确保已下载中文语言包 `chi_sim.traineddata` 并放入 `tessdata` 目录。

## 2. 操作流程

### 第一步：PDF 视觉解析与清洗
运行 `day1_ui.py`：
1. 点击“浏览 PDF”选择文件。
2. 点击“🚀 开始解析”，系统将进行 OCR 提取并调用大模型进行语义修复。
3. 解析完成后，点击“💾 导出结果”保存为 `day1_export.json`。

### 第二步：结构化 ETL 处理
运行 `day2_etl.py`：
- 该脚本会自动读取 `day1_export.json`。
- 它会将文本切分为适合向量化的块，并注入章节路径信息。
- 输出文件：`day2_vector_ready.json`。

### 第三步：向量化入库
运行 `bge_light_rag.py`：
1. 在界面选择输入文件 `day2_vector_ready.json`。
2. 配置服务商（局域网或硅基流动）。
3. 点击“🚀 开始向量化处理”，系统将并发生成向量并存入 `rag_production.db`。

### 第四步：智能问答交互
运行 `RAG_lightQueryFront.py`：
1. 点击“浏览库文件”挂载 `rag_production.db`。
2. 在右侧输入框提出您的问题。
3. 点击“⚡ 发送请求”，即可查看全链路 RAG 处理日志及最终生成的答案。

## 3. 常见问题 (FAQ)
- **Q: 为什么 OCR 识别不准？**
  - A: 请检查 `config.py` 中的 `OCR_DPI` 设置，扫描件建议设为 300 以上。
- **Q: 为什么连接不上大模型 API？**
  - A: 请检查各脚本顶部的 `API_KEY` 与 `API_URL` 是否配置正确，且网络是否可达。
- **Q: 数据库加载失败？**
  - A: 确保 `rag_production.db` 路径正确且文件未被其他程序占用。
