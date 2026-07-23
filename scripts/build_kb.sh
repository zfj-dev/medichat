#!/bin/bash
set -e
cd "$(dirname "$0")/.."

if [ -f "data/milvus_backup.tar.gz" ]; then
    echo "=== 解压预构建向量库 ==="
    tar -xzf data/milvus_backup.tar.gz -C data/
    echo "✅ 向量库就绪"
else
    echo "=== 从头构建知识库 ==="
    python3.11 vectorize.py
    echo "✅ 知识库构建完成"
fi
