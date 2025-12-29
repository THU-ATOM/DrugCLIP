FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 禁用 Python 输出缓冲，确保日志实时输出
ENV PYTHONUNBUFFERED=1

# 安装 Python 和 pip（保持这层稳定）
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
 && rm -rf /var/lib/apt/lists/*

# ✅ 单独 COPY requirements.txt，提高缓存命中率
COPY requirements.txt /tmp/requirements.txt

# ✅ 先安装 Python 依赖（包含 torch），使用国内镜像加速
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --timeout 300

# 安装 Uni-Core（此时 torch 已经安装）
RUN git clone https://github.com/dptech-corp/Uni-Core.git /tmp/Uni-core \
    && cd /tmp/Uni-core \
    && python3 setup.py install \
    && rm -rf /tmp/Uni-core

# 修复 lmdb 权限问题：给 __pycache__ 目录设置写权限
RUN chmod -R 777 /usr/local/lib/python3.10/dist-packages/ || true

# ✅ 再复制项目代码（此 COPY 不会影响上面依赖安装缓存）
#COPY . /app

# ✅ 设置工作目录
WORKDIR /app

# 说明用途的注释（非执行指令）
# 挂载点说明：
# - /data       -> 工作目录（rundir）
# - /checkpoint -> 模型权重目录（checkpointdir）


