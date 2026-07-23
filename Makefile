.PHONY: help install test run clean docker-build docker-up

help:
	@echo "MediChat 开发命令"
	@echo ""
	@echo "  make install      安装依赖"
	@echo "  make test         运行测试"
	@echo "  make run          启动服务"
	@echo "  make docker-build 构建 Docker 镜像"
	@echo "  make docker-up    一键启动 (Docker Compose)"
	@echo "  make clean        清理临时文件"

install:
	pip install -e .

test:
	pytest tests/ -v

run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t medichat:latest -f docker/Dockerfile .

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
