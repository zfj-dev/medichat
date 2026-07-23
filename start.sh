#!/bin/bash
source /hy-tmp/medichat/venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p /hy-tmp/medichat/logs

echo "启动编排层..."
nohup python /hy-tmp/medichat/server/orchestrator_v2.py > /hy-tmp/medichat/logs/orchestrator.log 2>&1 &

echo "等待模型加载（约90秒，首次下载需更久）..."
sleep 90

echo "启动前端..."
nohup streamlit run /hy-tmp/medichat/server/streamlit_app.py --server.address 0.0.0.0 --server.port 7860 > /hy-tmp/medichat/logs/streamlit.log 2>&1 &

echo "MediChat 已启动"
echo "前端: http://localhost:7860"
