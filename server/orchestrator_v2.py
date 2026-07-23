"""MediChat 编排层 v2 —— RAG + 安全 + 生成"""
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np

import sys
sys.path.insert(0, "/hy-tmp/medichat")
from safety.classifier.inference import load as load_classifier, predict as classify
from safety.post_filter import filter as post_filter

import torch

model = None
tokenizer = None
embedder = None
collection = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, embedder, collection

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    from pymilvus import connections, Collection

    MODEL_PATH = "/hy-tmp/medichat/models/models/Qwen--Qwen2-7B-Instruct/snapshots/master"

    print("加载主模型（FP16）...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("加载 LoRA adapter...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, "/hy-tmp/medichat/models/lora-adapter")
    model = model.to("cuda")
    print("LoRA adapter 加载完成")
    print("主模型加载完成")

    print("加载向量模型...")
    embedder = SentenceTransformer("/hy-tmp/medichat/models/models/BAAI--bge-large-zh-v1.5/snapshots/master/")
    connections.connect(uri="/hy-tmp/medichat/data/milvus.db")
    collection = Collection("medichat_knowledge")
    collection.load()
    print(f"知识库就绪: {collection.num_entities} 条")

    # 加载 BERT 分类器
    try:
        load_classifier()
    except Exception as e:
        print(f"BERT 分类器加载失败（可能还未训练）: {e}")
        
    yield

app = FastAPI(title="MediChat v2", lifespan=lifespan)

# === 安全规则 ===
CHEST_ZONES = ["胸口", "心脏", "心口", "心前区", "前胸"]
CHEST_FEELS = ["痛", "疼", "剧烈", "难受", "压迫", "不适", "不舒服", "紧缩", "刺痛"]

TEMPLATES = {
    "chest_pain": "您描述的症状（胸痛/胸闷/心脏不适）可能是需要紧急医疗处理的信号。请立即拨打120急救电话，前往最近医院急诊科。等待期间保持平静，解开紧身衣物，采取半卧位。请不要自行驾车。",
    "self_harm": "我们注意到您可能正在经历非常困难的时刻。请知道您不是一个人。全国24小时心理援助热线：400-161-9995。紧急情况请立即拨打120。",
    "breathing": "您描述的呼吸困难症状可能是需要紧急医疗处理的信号。请立即拨打120，采取坐姿或半卧位，不要平躺。松开领口和腰带。",
    "bleeding": "您描述的大出血症状需要立即急救。请立即拨打120，对出血部位用清洁布料用力按压止血。",
    "stroke": "您描述的症状可能是脑卒中征兆，抢救有黄金时间窗口。请立即拨打120，让患者平躺头偏一侧，不要喂水喂药。",
}

DISCLAIMER = "\n\n以上内容仅供参考，不能替代专业医疗诊断。如有健康问题，请咨询医生。"


def check_safety(text):
    # 第一层：关键词规则引擎（快速，无漏报优先）
    rules = [
        (["自杀", "不想活", "结束生命", "自残", "自伤", "割腕", "跳楼", "寻死"], "self_harm"),
        (["胸痛", "胸闷", "心绞痛", "心梗"], "chest_pain"),
        (["呼吸困难", "喘不过气", "窒息", "透不过气", "上不来气"], "breathing"),
        (["大出血", "大量出血", "咳血", "吐血", "便血"], "bleeding"),
        (["突然晕倒", "昏迷", "不省人事", "半身不遂", "口眼歪斜", "半身", "动不了"], "stroke"),
    ]
    for kws, tpl in rules:
        if any(kw in text for kw in kws):
            return "emergency", TEMPLATES.get(tpl, TEMPLATES["chest_pain"])

    if any(z in text for z in CHEST_ZONES) and any(f in text for f in CHEST_FEELS):
        return "emergency", TEMPLATES["chest_pain"]

    # 第二层：BERT 分类器（捕获关键词遗漏的情况）
    try:
        result = classify(text)
        if result["label"] == "urgent" :
            return "emergency", TEMPLATES.get("chest_pain",
                       "检测到可能的紧急情况。建议您立即就医。如有严重不适，请拨打120。")
        if result["label"] == "manipulative" :
            return "flag", ""  # 标记为诱导性提问，generate 里会加强约束
    except Exception:
        pass  # 分类器不可用时降级，仅靠关键词

    return "normal", ""


def retrieve(query, top_k=3):
    try:
        qv = embedder.encode([query], normalize_embeddings=True)
        results = collection.search(
            data=qv.tolist(), anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k, output_fields=["content", "title", "source"],
        )
        docs = []
        for hit in results[0]:
            if hit.score > 0.5:
                docs.append({
                    "content": hit.entity.get("content", ""),
                    "title": hit.entity.get("title", ""),
                    "source": hit.entity.get("source", ""),
                    "score": round(hit.score, 3),
                })
        return docs
    except Exception as e:
        print(f"检索异常: {e}")
        return []


def generate(message, docs):
    ctx = ""
    if docs:
        parts = [f"参考资料{i+1}: [{d['title']}]({d['source']})\n{d['content'][:500]}" for i, d in enumerate(docs[:3])]
        ctx = "\n\n".join(parts)

    prompt = f"<|im_start|>system\n你是医疗科普助手。用通俗中文回答。不提供诊断。如有参考资料请引用来源。\n{ctx}<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
    reply = tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    reply, was_blocked = post_filter(reply)
    return reply


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    action: str = "normal"
    sources: list = []


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    action, resp = check_safety(req.message)
    if action == "emergency":
        return ChatResponse(answer=resp, action="emergency")

    docs = retrieve(req.message)
    sources = [{"title": d["title"], "source": d["source"]} for d in docs]
    answer = generate(req.message, docs)
    return ChatResponse(answer=answer, action="normal", sources=sources)


@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
