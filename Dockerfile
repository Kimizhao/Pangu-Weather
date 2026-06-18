# Pangu-Weather 离线推理 Docker 镜像
# 基于验证成功的配置: onnxruntime-gpu 1.18.0 + CUDA 11.8 + cuDNN 8.9
#
# 构建方法: docker build -t pangu-weather-inference .
# 运行方法: 见 docker-compose.yml 或 README_DOCKER.md
#
# 镜像仅包含推理框架和脚本，模型和数据通过volume挂载

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖和 Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && python3 -m pip install --upgrade pip --no-cache-dir

# 安装 Python 依赖（匹配验证成功的版本）
RUN pip install --no-cache-dir \
    onnxruntime-gpu==1.18.0 \
    "numpy<2" \
    onnx

# 验证 onnxruntime CUDA 支持
RUN python -c "import onnxruntime as ort; print('ORT版本:', ort.__version__); print('可用Provider:', ort.get_available_providers())"

# 创建工作目录
WORKDIR /app

# 复制推理脚本
COPY inference_docker.py /app/
COPY inference_iterative_docker.py /app/
COPY inference_gpu.py /app/
COPY inference_iterative.py /app/
COPY inference_cpu.py /app/
COPY pseudocode.py /app/
COPY README.md /app/

# 创建数据目录（模型和数据通过volume挂载）
RUN mkdir -p /app/input_data /app/output_data /app/constant_masks /app/models

# 设置环境变量
ENV PANGU_BASE_DIR=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cudnn/lib64:${LD_LIBRARY_PATH}

# 默认入口: 运行24小时推理
ENTRYPOINT ["python", "/app/inference_docker.py"]
