import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from tqdm import tqdm

CHUNKS_FILE = "/hy-tmp/medichat/data/all_chunks.jsonl"
COLLECTION_NAME = "medichat_knowledge"

# 加载片段
chunks = []
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line.strip()))
print(f"加载 {len(chunks)} 个片段")

# 加载向量模型
print("加载 bge-large-zh-v1.5...")
model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
print("向量模型就绪")

# 连接 Milvus Lite
connections.connect(uri="/hy-tmp/medichat/data/milvus.db")

if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=32, is_primary=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
]
schema = CollectionSchema(fields, "MediChat 医疗知识库")
collection = Collection(COLLECTION_NAME, schema)

# 向量化写入
total = len(chunks)
for i in tqdm(range(0, total, 64)):
    batch = chunks[i:i+64]
    contents = [c["content"] for c in batch]
    embeddings = model.encode(contents, batch_size=64)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    entities = [
        [c["id"] for c in batch],
        [c["content"][:4000] for c in batch],
        embeddings.tolist(),
        [c["source"] for c in batch],
        [c["title"][:250] for c in batch],
        [c["category"] for c in batch],
    ]
    collection.insert(entities)

collection.flush()
collection.create_index("embedding", {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
collection.load()
print(f"\n完成！{total} 条记录已写入")
