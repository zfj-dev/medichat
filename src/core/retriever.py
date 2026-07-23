"""
MediChat 混合检索引擎
BM25 稀疏检索 + BGE 稠密向量 + BGE-reranker 重排序
"""
import json
import numpy as np
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import jieba


class HybridRetriever:
    """混合检索器：BM25 + 稠密向量 → 加权融合 → reranker 精排"""

    def __init__(self, embedder, collection, reranker=None):
        self.embedder = embedder          # SentenceTransformer (BGE)
        self.collection = collection      # Milvus Collection
        self.reranker = reranker          # CrossEncoder (BGE-reranker)
        self.bm25 = None
        self.chunks: List[Dict] = []
        self._build_bm25_index()

    def _build_bm25_index(self):
        """从 Milvus 构建 BM25 索引"""
        try:
            results = self.collection.query(
                expr="id != ''",
                output_fields=["content", "title", "source"],
                limit=10000,
            )
            self.chunks = results
            if results:
                tokenized = [list(jieba.cut(c["content"])) for c in results]
                self.bm25 = BM25Okapi(tokenized)
        except Exception:
            self.bm25 = None

    def _dense_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """稠密向量检索"""
        qv = self.embedder.encode([query], normalize_embeddings=True)
        results = self.collection.search(
            data=qv.tolist(),
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["content", "title", "source"],
        )
        docs = []
        for hit in results[0]:
            docs.append({
                "content": hit.entity.get("content", ""),
                "title": hit.entity.get("title", ""),
                "source": hit.entity.get("source", ""),
                "dense_score": float(hit.score),
            })
        return docs

    def _sparse_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 稀疏检索"""
        if self.bm25 is None:
            return []
        tokenized = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized)

        # 归一化
        if scores.max() > 0:
            scores = scores / scores.max()

        # Top-K
        indices = np.argsort(scores)[::-1][:top_k]
        docs = []
        for idx in indices:
            if scores[idx] > 0:
                docs.append({
                    "content": self.chunks[idx]["content"],
                    "title": self.chunks[idx].get("title", ""),
                    "source": self.chunks[idx].get("source", ""),
                    "sparse_score": float(scores[idx]),
                })
        return docs

    def _fuse_results(
        self,
        dense_docs: List[Dict],
        sparse_docs: List[Dict],
        dense_weight: float = 0.7,
    ) -> List[Dict]:
        """加权融合稠密和稀疏结果"""
        fused = {}  # key = content[:100]

        for doc in dense_docs:
            key = doc["content"][:100]
            fused[key] = {
                **doc,
                "final_score": doc.get("dense_score", 0) * dense_weight,
            }

        for doc in sparse_docs:
            key = doc["content"][:100]
            sparse_score = doc.get("sparse_score", 0) * (1 - dense_weight)
            if key in fused:
                fused[key]["final_score"] += sparse_score
                fused[key]["sparse_score"] = doc.get("sparse_score", 0)
            else:
                fused[key] = {
                    **doc,
                    "final_score": sparse_score,
                }

        results = sorted(fused.values(), key=lambda x: x["final_score"], reverse=True)
        return results

    def _rerank(self, query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
        """BGE-reranker 精排"""
        if self.reranker is None or not docs:
            return docs[:top_k]

        pairs = [[query, d["content"][:500]] for d in docs]
        scores = self.reranker.predict(pairs)

        for i, doc in enumerate(docs):
            doc["rerank_score"] = float(scores[i])

        docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return docs[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.7,
        use_reranker: bool = True,
    ) -> List[Dict]:
        """
        混合检索主入口
        1. 并行执行稠密和稀疏检索
        2. 加权融合结果
        3. Reranker 精排
        """
        dense_docs = self._dense_search(query, top_k=10)
        sparse_docs = self._sparse_search(query, top_k=10)
        fused = self._fuse_results(dense_docs, sparse_docs, dense_weight)

        if use_reranker:
            fused = self._rerank(query, fused, top_k)
        else:
            fused = fused[:top_k]

        return fused
