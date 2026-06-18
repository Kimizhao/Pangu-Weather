# Pangu-Weather

盘古气象预报模型 ONNX 推理项目。

## 脚本清单

| 脚本 | 用途 | 环境 |
|------|------|------|
| `inference_cpu.py` | 24h CPU推理 | 任意 |
| `inference_gpu.py` | 24h GPU推理（原始版） | CUDA 11.6 |
| `inference_gpu_4090.py` | 24h GPU推理（4090适配，硬编码路径） | 本地4090 |
| `inference_docker.py` | 24h GPU推理（Docker适配，环境变量配置路径） | Docker |
| `inference_iterative.py` | 7天迭代推理（原始版） | CUDA 11.6 |
| `inference_iterative_4090.py` | 7天迭代推理（4090适配，逐步加载模型） | 本地4090 |
| `inference_iterative_docker.py` | 7天迭代推理（Docker适配） | Docker |
| `inference_multi_lead_4090.py` | 4时效单步推理+迭代（4090，逐步释放显存） | 本地4090 |
| `pseudocode.py` | 3D神经网络伪代码 | 参考用 |

## CUDA兼容性红线

**onnxruntime-gpu==1.18.0** 是唯一在RTX 4090上验证通过的版本。更高版本(1.23.2)的cuDNN 9前端在sm_89上崩溃。

本地运行必须设置 LD_LIBRARY_PATH 包含 nvidia-*-cu11 pip包路径。Docker环境用 `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` 基础镜像即可。

## Docker离线部署

镜像不含模型和数据，通过volume挂载。详见 `README_DOCKER.md`。

```bash
# GPU推理
docker run --rm --gpus all -v $(pwd):/workspace:ro -v $(pwd)/output_data:/app/output_data -e PANGU_MODEL_DIR=/workspace pangu-weather-inference
```

## 数据格式

- 输入高空场: `(5, 13, 721, 1440)` float32 — Z/Q/T/U/V × 13气压层 × 0.25°网格
- 输入地面场: `(4, 721, 1440)` float32 — MSLP/U10/V10/T2M × 0.25°网格
- 模型: 4个ONNX文件(~1.1GB each) — pangu_weather_{1,3,6,24}.onnx
