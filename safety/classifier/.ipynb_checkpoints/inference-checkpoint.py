"""BERT 安全分类器推理模块"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

MODEL_PATH = "/hy-tmp/medichat/safety/classifier/saved_model"
LABELS = ["normal", "urgent", "manipulative"]

_model = None
_tokenizer = None


def load():
    """加载 BERT 分类器（仅调用一次）"""
    global _model, _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    _model.eval()
    if torch.cuda.is_available():
        _model = _model.to("cuda")
    print("BERT 分类器就绪")


def predict(text: str) -> dict:
    """
    返回 {"label": str, "confidence": float}
    """
    if _model is None:
        raise RuntimeError("分类器未加载，请先调用 load()")

    enc = _tokenizer(text, truncation=True, padding="max_length",
                     max_length=128, return_tensors="pt")
    if torch.cuda.is_available():
        enc = {k: v.to("cuda") for k, v in enc.items()}

    with torch.no_grad():
        logits = _model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    idx = int(np.argmax(probs))
    return {"label": LABELS[idx], "confidence": float(probs[idx])}