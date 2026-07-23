import sys; sys.path.insert(0, "/hy-tmp/medichat")
"""
MediChat FastAPI 主程序 —— SSE 流式输出 + 多轮对话 + 混合检索
"""
import json
import uuid
import sqlite3
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.config import settings
from src.core.llm import LLMEngine
from src.core.retriever import HybridRetriever
from src.safety.pipeline import SafetyPipeline
from safety.post_filter import filter as post_filter

# ---- 全局组件 ----
llm: LLMEngine = None
retriever: HybridRetriever = None
safety: SafetyPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, retriever, safety

    print("[1/4] 加载 LLM (含 LoRA adapter)...")
    llm = LLMEngine(
        model_path=settings.resolve(settings.model_path),
        lora_path=settings.resolve(settings.lora_adapter_path),
    )

    print("[2/4] 加载检索器...")
    from sentence_transformers import SentenceTransformer
    from pymilvus import connections, Collection

    embedder = SentenceTransformer(settings.resolve(settings.vector_model_path))
    connections.connect(uri=settings.resolve(settings.milvus_uri))
    collection = Collection(settings.collection_name)
    collection.load()

    # 尝试加载 reranker
    reranker = None
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(settings.resolve(settings.reranker_model_path))
        print("  Reranker 加载成功")
    except Exception:
        print("  Reranker 未安装，跳过重排序")

    retriever = HybridRetriever(embedder, collection, reranker)

    print("[3/4] 加载安全模块...")
    safety = SafetyPipeline()

    print("[4/4] 启动完成！")
    yield


app = FastAPI(title="MediChat", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ============================================================
# API 模型
# ============================================================
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    stream: bool = True


class ChatResponse(BaseModel):
    answer: str
    action: str = "normal"
    sources: list = []


# ============================================================
# 路由
# ============================================================
@app.get("/healthz")
async def health():
    return {
        "status": "healthy",
        "llm_loaded": llm is not None,
        "retriever_ready": retriever is not None and retriever.bm25 is not None,
        "safety_ready": safety is not None,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """普通聊天（非流式）"""
    # 安全检测
    action, resp = safety.check(req.message)
    if action == "emergency":
        return ChatResponse(answer=resp, action="emergency")

    # 检索
    docs = retriever.search(req.message, top_k=settings.rerank_top_k)

    # 生成
    answer = llm.generate(req.message, docs[:3])

    # 后过滤
    answer, _ = post_filter(answer)

    sources = [{"title": d["title"], "source": d["source"]} for d in docs[:3]]
    return ChatResponse(answer=answer, action="normal", sources=sources)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式聊天"""
    # 安全检测
    action, resp = safety.check(req.message)
    if action == "emergency":
        return ChatResponse(answer=resp, action="emergency")

    # 检索
    docs = retriever.search(req.message, top_k=settings.rerank_top_k)

    async def generate() -> AsyncGenerator[str, None]:
        full_answer = ""
        async for token in llm.generate_stream(req.message, docs[:3]):
            full_answer += token
            yield f"data: {json.dumps({'token': token})}\n\n"

        # 后过滤
        full_answer, _ = post_filter(full_answer)

        # 发送来源
        sources = [{"title": d["title"], "source": d["source"]} for d in docs[:3]]
        yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.api_port)
