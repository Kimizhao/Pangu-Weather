import os
import numpy as np
import onnxruntime as ort
import time
import gc

# 自动检测目录
BASE_DIR = os.environ.get('PANGU_BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get('PANGU_MODEL_DIR', os.path.join(BASE_DIR, 'models') if os.path.isdir(os.path.join(BASE_DIR, 'models')) else BASE_DIR)
input_data_dir = os.path.join(MODEL_DIR, 'input_data')
output_data_dir = os.path.join(BASE_DIR, 'output_data')

# GPU选择
cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

var_names_upper = ['Z(位势)', 'Q(比湿)', 'T(温度)', 'U(纬向风)', 'V(经向风)']
var_names_surface = ['MSLP(地面气压)', 'U10(10m纬向风)', 'V10(10m经向风)', 'T2M(2m温度)']

# 加载输入数据
input_upper = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)
print(f"输入高空场: shape={input_upper.shape}")
print(f"输入地面场: shape={input_surface.shape}")

def create_session(model_file):
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 4
    cuda_opts = {'arena_extend_strategy': 'kSameAsRequested'}
    sess = ort.InferenceSession(
        os.path.join(MODEL_DIR, model_file),
        sess_options=options,
        providers=[('CUDAExecutionProvider', cuda_opts), 'CPUExecutionProvider']
    )
    return sess

# 迭代推理：7天预报（24h+6h，逐步加载模型以节省显存）
print("=" * 60)
print("迭代推理: 7天预报（24h + 6h模型组合）")
print("=" * 60)

current_upper = input_upper.copy()
current_surface = input_surface.copy()
input_24_upper = input_upper.copy()
input_24_surface = input_surface.copy()

total_start = time.time()

for i in range(28):
    step_num = i + 1
    use_24h = (step_num % 4 == 0)
    model_file = 'pangu_weather_24.onnx' if use_24h else 'pangu_weather_6.onnx'
    model_tag = "24h" if use_24h else "6h"

    sess = create_session(model_file)

    if use_24h:
        output, output_surface = sess.run(None, {'input': input_24_upper, 'input_surface': input_24_surface})
        input_24_upper = output.copy()
        input_24_surface = output_surface.copy()
    else:
        output, output_surface = sess.run(None, {'input': current_upper, 'input_surface': current_surface})

    current_upper = output.copy()
    current_surface = output_surface.copy()

    del sess; gc.collect()

    os.makedirs(output_data_dir, exist_ok=True)
    np.save(os.path.join(output_data_dir, f'output_upper_iter_step{step_num}'), output)
    np.save(os.path.join(output_data_dir, f'output_surface_iter_step{step_num}'), output_surface)

    elapsed = time.time() - total_start
    print(f"步骤 {step_num}/28: {model_tag} | 已用{elapsed:.1f}s")

total_time = time.time() - total_start
print(f"\n迭代推理完成，总耗时: {total_time:.2f}s")
print(f"最终输出高空场: {current_upper.shape}")
for i, name in enumerate(var_names_upper):
    print(f"  {name}: min={current_upper[i].min():.4f}, max={current_upper[i].max():.4f}")
for i, name in enumerate(var_names_surface):
    print(f"  {name}: min={current_surface[i].min():.4f}, max={current_surface[i].max():.4f}")
print(f"\n28步结果保存在 {output_data_dir}/")
