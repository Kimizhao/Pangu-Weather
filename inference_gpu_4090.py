import os
import numpy as np
import onnxruntime as ort
import time

# 设置LD_LIBRARY_PATH使onnxruntime找到CUDA 12库（nvidia pip包）
nvidia_lib_paths = [
    '/home/tx/miniconda3/envs/pangu/lib/python3.10/site-packages/nvidia/cublas/lib',
    '/home/tx/miniconda3/envs/pangu/lib/python3.10/site-packages/nvidia/cuda_runtime/lib',
    '/home/tx/miniconda3/envs/pangu/lib/python3.10/site-packages/nvidia/cudnn/lib',
    '/home/tx/miniconda3/envs/pangu/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib',
]
existing_ld = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = ':'.join(nvidia_lib_paths) + ':' + existing_ld

# 指定使用GPU 1（RTX 4090）
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 输入输出目录（使用绝对路径）
BASE_DIR = '/mnt/data/workspace/Pangu-Weather'
input_data_dir = os.path.join(BASE_DIR, 'input_data')
output_data_dir = os.path.join(BASE_DIR, 'output_data')

# 设置 onnxruntime 会话选项（适配4090）
options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
# 4090有24GB显存，适当增加线程数（原始为1）
options.intra_op_num_threads = 4
options.inter_op_num_threads = 4

# CUDA Provider 配置 - 适配4090
cuda_provider_options = {
    'arena_extend_strategy': 'kSameAsRequested',
}

# 初始化推理会话（跳过 onnx.load 冗余操作，直接用 InferenceSession）
print("正在加载24小时模型 pangu_weather_24.onnx ...")
start_time = time.time()
ort_session_24 = ort.InferenceSession(
    os.path.join(BASE_DIR, 'pangu_weather_24.onnx'),
    sess_options=options,
    providers=[('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider']
)
load_time = time.time() - start_time
print(f"模型加载完成，耗时 {load_time:.2f} 秒")
print(f"实际使用的Provider: {ort_session_24.get_providers()}")

# 加载输入数据
print("正在加载输入数据...")
input_upper = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)

print(f"输入高空场形状: {input_upper.shape}")  # 应为 (5, 13, 721, 1440)
print(f"输入地面场形状: {input_surface.shape}")  # 应为 (4, 721, 1440)
print(f"数据类型: {input_upper.dtype}")  # 应为 float32

# 运行24小时预报推理
print("开始24小时预报推理（使用RTX 4090 GPU）...")
inference_start = time.time()
output, output_surface = ort_session_24.run(
    None,
    {'input': input_upper, 'input_surface': input_surface}
)
inference_time = time.time() - inference_start
print(f"推理完成，耗时 {inference_time:.2f} 秒")

# 保存结果
os.makedirs(output_data_dir, exist_ok=True)
np.save(os.path.join(output_data_dir, 'output_upper'), output)
np.save(os.path.join(output_data_dir, 'output_surface'), output_surface)

print(f"输出高空场形状: {output.shape}")
print(f"输出地面场形状: {output_surface.shape}")

# 输出数值范围验证
print(f"高空场输出范围: min={output.min():.2f}, max={output.max():.2f}")
print(f"地面场输出范围: min={output_surface.min():.2f}, max={output_surface.max():.2f}")

# 各变量范围检查
var_names_upper = ['Z(位势)', 'Q(比湿)', 'T(温度)', 'U(纬向风)', 'V(经向风)']
var_names_surface = ['MSLP(地面气压)', 'U10(10m纬向风)', 'V10(10m经向风)', 'T2M(2m温度)']
for i, name in enumerate(var_names_upper):
    print(f"  {name}: min={output[i].min():.4f}, max={output[i].max():.4f}, mean={output[i].mean():.4f}")
for i, name in enumerate(var_names_surface):
    print(f"  {name}: min={output_surface[i].min():.4f}, max={output_surface[i].max():.4f}, mean={output_surface[i].mean():.4f}")

print(f"\n推理结果已保存到 {output_data_dir}/ 目录")
print(f"总耗时: 模型加载 {load_time:.2f}s + 推理 {inference_time:.2f}s = {load_time + inference_time:.2f}s")
