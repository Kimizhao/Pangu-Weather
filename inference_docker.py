import os
import numpy as np
import onnxruntime as ort
import time

# 自动检测目录（支持Docker容器和本地环境）
BASE_DIR = os.environ.get('PANGU_BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get('PANGU_MODEL_DIR', os.path.join(BASE_DIR, 'models') if os.path.isdir(os.path.join(BASE_DIR, 'models')) else BASE_DIR)
# 输入数据目录：优先在MODEL_DIR下找，否则在BASE_DIR下找
input_data_dir = os.path.join(MODEL_DIR, 'input_data') if os.path.isdir(os.path.join(MODEL_DIR, 'input_data')) else os.path.join(BASE_DIR, 'input_data')
output_data_dir = os.path.join(BASE_DIR, 'output_data')

# GPU选择（默认使用所有可见GPU，可通过环境变量指定）
cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if not cuda_device:
    # 默认使用第一个GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Session options
options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
# 线程数可通过环境变量调整
options.intra_op_num_threads = int(os.environ.get('ORT_INTRA_THREADS', '4'))
options.inter_op_num_threads = int(os.environ.get('ORT_INTER_THREADS', '4'))

cuda_provider_options = {'arena_extend_strategy': 'kSameAsRequested'}

# 加载24小时模型
print("="*60)
print("Pangu-Weather 24小时预报推理")
print("="*60)
print(f"工作目录: {BASE_DIR}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")

model_path = os.path.join(MODEL_DIR, 'pangu_weather_24.onnx')
print(f"正在加载模型: {model_path}")
start_time = time.time()
ort_session_24 = ort.InferenceSession(
    model_path,
    sess_options=options,
    providers=[('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider']
)
load_time = time.time() - start_time
print(f"模型加载耗时: {load_time:.2f}s")
print(f"使用Provider: {ort_session_24.get_providers()}")

# 加载输入数据
input_upper = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)
print(f"输入高空场: shape={input_upper.shape}, dtype={input_upper.dtype}")
print(f"输入地面场: shape={input_surface.shape}, dtype={input_surface.dtype}")

# 运行推理
print("开始推理...")
inf_start = time.time()
output, output_surface = ort_session_24.run(
    None, {'input': input_upper, 'input_surface': input_surface}
)
inf_time = time.time() - inf_start
print(f"推理完成，耗时 {inf_time:.2f}s")

# 保存结果
os.makedirs(output_data_dir, exist_ok=True)
np.save(os.path.join(output_data_dir, 'output_upper'), output)
np.save(os.path.join(output_data_dir, 'output_surface'), output_surface)

print(f"输出高空场: shape={output.shape}")
print(f"输出地面场: shape={output_surface.shape}")

# 变量验证
var_names_upper = ['Z(位势)', 'Q(比湿)', 'T(温度)', 'U(纬向风)', 'V(经向风)']
var_names_surface = ['MSLP(地面气压)', 'U10(10m纬向风)', 'V10(10m经向风)', 'T2M(2m温度)']
for i, name in enumerate(var_names_upper):
    print(f"  {name}: min={output[i].min():.4f}, max={output[i].max():.4f}, mean={output[i].mean():.4f}")
for i, name in enumerate(var_names_surface):
    print(f"  {name}: min={output_surface[i].min():.4f}, max={output_surface[i].max():.4f}, mean={output_surface[i].mean():.4f}")

print(f"\n结果已保存: {output_data_dir}/")
print(f"总耗时: 加载 {load_time:.2f}s + 推理 {inf_time:.2f}s = {load_time + inf_time:.2f}s")
