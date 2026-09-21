FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PLANT_DIR=/data/plants

WORKDIR /srv

# 先装依赖，利用镜像层缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# 测试镜像：docker build --target test -t open-loop-margin:test .
#           docker run --rm open-loop-margin:test
FROM base AS test
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY tests ./tests
COPY pytest.ini ./
CMD ["python", "-m", "pytest"]

# 运行镜像（默认目标，docker compose up 走这里）
FROM base AS runtime

# 对象档本地文件持久化卷；不另起数据库进程
RUN mkdir -p /data/plants
VOLUME ["/data/plants"]

EXPOSE 8000

# 单容器一条命令启动（docker compose up 也走这里）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
