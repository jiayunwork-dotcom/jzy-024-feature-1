#!/usr/bin/env bash
# 一条命令构建并启动单容器。
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p ./data/plants
docker compose up --build -d

echo
echo "服务已启动： http://127.0.0.1:8000"
echo "接口文档：   http://127.0.0.1:8000/docs"
echo "对象档目录： ./data/plants （挂载进容器 /data/plants）"
