# 🏥 MediChat — 中文医疗科普问答系统

基于 RAG 检索增强生成的医疗科普助手。以默沙东诊疗手册为知识库，结合三层安全防线和 QLoRA 微调的 Qwen2-7B 模型，提供安全、可溯源的医疗知识问答。

> ⚠️ **免责声明：** 本系统仅提供医学科普，不提供诊断、处方或个人医疗建议。如有健康问题请及时就医。

## ✨ 功能亮点

- **检索增强生成（RAG）** — 基于默沙东诊疗手册，回答有据可查，每条回答附带文献来源
- **三层安全防线** — 关键词规则 → BERT 安全分类 → 输出后过滤，拒绝诊断类请求，主动输出免责声明
- **混合检索 + 重排序** — BM25 稀疏检索 + BGE 稠密向量，融合后经 BGE-Reranker 精排，兼顾语义和关键词匹配
- **QLoRA 微调** — 模型回答风格更通俗，主动拒绝诊断请求，回答末尾自动附加免责声明
- **SSE 流式输出** — 首个 token 即开始显示，无需等待完整回答
- **一键部署** — Docker Compose 一条命令启动，无需手动配置环境

## 🏗️ 架构

```
用户浏览器 (:7860)
       │
       ▼
FastAPI 编排层 (:8000)
  ├── 安全流水线（关键词 → BERT → 后过滤）
  ├── 混合检索（BM25 + BGE稠密 → 融合 → Reranker）
  └── Qwen2-7B (LoRA微调) + SSE 流式输出
       │
       ▼
Milvus 向量数据库（默沙东诊疗手册）
```

## 🚀 快速开始

### 前提条件

- Python 3.11+
- NVIDIA GPU（推荐 8GB+ 显存，Qwen2-7B FP16 约需 14GB）
- 或 CPU 模式（仅运行安全模块和检索，LLM 需调用云端 API）

### 1. 克隆仓库

```bash
git clone https://github.com/zfj-dev/medichat.git
cd medichat
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
# 或
pip install fastapi uvicorn transformers peft sentence-transformers pymilvus rank-bm25 jieba pydantic
```

### 3. 下载模型（二选一）

**方式 A：自动下载（推荐）**
```bash
bash scripts/download_models.sh
```

**方式 B：手动下载**
- Qwen2-7B-Instruct：从 ModelScope 下载到 `models/models/Qwen--Qwen2-7B-Instruct/`
- BGE-large-zh-v1.5：从 ModelScope 下载到 `models/models/BAAI--bge-large-zh-v1.5/`

### 4. 构建知识库

```bash
bash scripts/build_kb.sh
```

### 5. 启动服务

```bash
# API 服务
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 前端（另一个终端）
streamlit run server/streamlit_app.py
```

### 6. 打开浏览器

- 聊天界面：http://localhost:7860
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/healthz

### Docker 部署（可选）

```bash
docker build -t medichat -f docker/Dockerfile .
docker-compose -f docker/docker-compose.yml up -d
```

## 🛡️ 安全设计

详见 [ARCHITECTURE.md](./docs/ARCHITECTURE.md)

MediChat 采用三层安全防线：
1. **关键词规则引擎** — 最快拦截，0 延迟，覆盖常见危险请求
2. **BERT 安全分类器** — 语义级判断，识别变体表述
3. **输出后过滤** — 最后一道防线，追加免责声明

## 📊 项目结构

```
medichat/
├── src/
│   ├── api/          # FastAPI 接口层
│   ├── core/         # 核心逻辑（编排、LLM、检索、配置）
│   └── safety/       # 安全模块（三层防线）
├── server/           # 原始服务端代码
│   ├── orchestrator_v2.py
│   └── streamlit_app.py
├── safety/           # 安全模块
├── scripts/          # 运维脚本
├── data/             # 知识库数据
├── tests/            # 单元测试
├── docs/             # 文档
├── docker/           # Docker 部署
└── .github/          # CI/CD
```

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| 大模型 | Qwen2-7B-Instruct (LoRA 微调) |
| 向量模型 | BGE-large-zh-v1.5 |
| 向量数据库 | Milvus Lite |
| 后端框架 | FastAPI (SSE 流式) |
| 前端 | Streamlit |
| 微调框架 | PEFT (LoRA) |
| 部署 | Docker + Docker Compose |
| CI/CD | GitHub Actions |

## 📖 文档

- [架构设计](./docs/ARCHITECTURE.md)
- [API 文档](http://localhost:8000/docs)

## 📄 许可

[Apache License 2.0](./LICENSE)

## ⚠️ 免责声明

MediChat 是一个技术演示和教育项目。本系统提供的所有信息仅供参考，不能替代专业医疗诊断、建议或治疗。如果您有健康问题，请咨询合格的医疗专业人员。
