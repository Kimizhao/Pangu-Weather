# Pangu-Weather Docker 离线推理指南

## 镜像说明
- 基础镜像: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
- onnxruntime-gpu 1.18.0 + numpy 1.26.4
- 镜像不含模型和数据文件，通过volume挂载

## 离线部署步骤

### 1. 构建镜像（联网环境）
```bash
docker build -t pangu-weather-inference .
```

### 2. 保存镜像为tar文件（用于离线传输）
```bash
docker save pangu-weather-inference -o pangu-weather-inference.tar
```

### 3. 离线环境加载镜像
```bash
docker load -i pangu-weather-inference.tar
```

### 4. 准备模型和数据目录
将以下文件放入对应目录：
- `pangu_weather_24.onnx` (1.2GB) - 根目录或models/
- `pangu_weather_6.onnx` (1.2GB)
- `pangu_weather_3.onnx` (1.2GB)  
- `pangu_weather_1.onnx` (1.2GB)
- `input_data/input_upper.npy` (258MB)
- `input_data/input_surface.npy` (16MB)

### 5. 运行推理
```bash
# 24小时单步推理（GPU）
docker compose run pangu-24h

# 6小时单步推理（GPU）
docker compose run pangu-6h

# 7天迭代推理（GPU）
docker compose run pangu-iterative

# CPU推理（无GPU环境）
docker compose --profile cpu run pangu-cpu

# 直接docker run（不使用compose）
docker run --gpus all \
  -v $(pwd):/workspace:ro \
  -v $(pwd)/output_data:/app/output_data \
  -e PANGU_MODEL_DIR=/workspace \
  pangu-weather-inference

# 进入容器交互
docker compose run pangu-24h bash
```

### 6. 查看结果
推理结果保存在 `output_data/` 目录：
- `output_upper.npy` - 高空场预报 (5,13,721,1440)
- `output_surface.npy` - 地面场预报 (4,721,1440)
- `output_upper_iter_step*.npy` - 迭代预报各步

## 环境变量
| 变量 | 默认值 | 说明 |
|------|--------|------|
| CUDA_VISIBLE_DEVICES | 0 | GPU编号 |
| PANGU_MODEL_DIR | /workspace | 模型文件目录 |
| PANGU_BASE_DIR | /app | 工作目录 |
| ORT_INTRA_THREADS | 4 | 推理线程数 |

## 注意事项
- 需要 nvidia-container-toolkit 才能使用GPU
- 镜像约4.75GB，加上挂载的模型文件总共约9GB
- RTX 4090单步推理约6秒，CPU推理约220秒
