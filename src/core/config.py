"""
MediChat 配置管理 —— 环境变量 + YAML 双模式
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    # 路径配置
    project_root: str = field(default_factory=lambda: os.environ.get("MEDICHAT_ROOT", "/hy-tmp/medichat"))
    model_path: str = "models/models/Qwen--Qwen2-7B-Instruct/snapshots/master"
    lora_adapter_path: str = "models/lora-adapter"
    vector_model_path: str = "models/models/BAAI--bge-large-zh-v1.5/snapshots/master"
    reranker_model_path: str = "models/models/BAAI--bge-reranker-v2-m3/snapshots/master"
    bert_model_path: str = "safety/classifier/saved_model"

    # 向量数据库
    milvus_uri: str = "data/milvus/milvus.db"
    collection_name: str = "medichat_knowledge"

    # 检索参数
    retrieval_top_k: int = 10
    dense_weight: float = 0.7       # 稠密向量权重
    sparse_weight: float = 0.3      # BM25 权重
    rerank_top_k: int = 5

    # 生成参数
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # 安全
    safety_enabled: bool = True
    post_filter_enabled: bool = True

    # 服务
    api_port: int = 8000
    ui_port: int = 7860

    # 日志
    log_level: str = "INFO"

    def resolve(self, relative_path: str) -> str:
        """将相对路径解析为绝对路径"""
        if os.path.isabs(relative_path):
            return relative_path
        return str(Path(self.project_root) / relative_path)

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量加载配置"""
        s = cls()
        for field_name in s.__dataclass_fields__:
            env_key = f"MEDICHAT_{field_name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                # 类型转换
                field_type = type(getattr(s, field_name))
                if field_type == bool:
                    setattr(s, field_name, env_val.lower() in ("true", "1", "yes"))
                elif field_type == int:
                    setattr(s, field_name, int(env_val))
                elif field_type == float:
                    setattr(s, field_name, float(env_val))
                else:
                    setattr(s, field_name, env_val)
        return s


# 全局单例
settings = Settings.from_env()
