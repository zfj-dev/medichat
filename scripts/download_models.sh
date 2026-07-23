#!/bin/bash
set -e
echo "=== MediChat 模型下载 ==="
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

pip3.11 install modelscope -q

python3.11 -c "
from modelscope import snapshot_download
import os
models_dir = os.path.join('$PROJECT_ROOT', 'models', 'models')
os.makedirs(models_dir, exist_ok=True)
print('下载 Qwen2-7B-Instruct...')
snapshot_download('Qwen/Qwen2-7B-Instruct', cache_dir=models_dir)
print('下载 BGE-large-zh-v1.5...')
snapshot_download('BAAI/bge-large-zh-v1.5', cache_dir=models_dir)
print('全部下载完成！')
"
