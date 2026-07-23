# MediChat Docker 镜像 —— 多阶段构建
FROM python:3.11-slim AS builder

WORKDIR /app
COPY pyproject.toml .

# 只安装核心依赖（不含 torch——运行时从宿主机挂载或用 CPU 版）
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] \
    sentence-transformers \
    pymilvus \
    peft \
    transformers \
    pydantic pydantic-settings \
    rank-bm25 jieba \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

# === 运行时 ===
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

RUN useradd -m medichat && chown -R medichat:medichat /app
USER medichat

EXPOSE 8000 7860

HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
