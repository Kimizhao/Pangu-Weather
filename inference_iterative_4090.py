import os
import numpy as np
import onnxruntime as ort
import time
import gc

# 设置LD_LIBRARY_PATH
nvidia_lib_paths = [
    '/home/tx/miniconda3/envs/pangu/lib/python3.10/site-packages/nvidia/cublas/lib',
    '/home/tx/miniconda3/envs/pangu/lib/python3.10/site-packages/nvidia/cuda_runtime/lib',
    '/home/tx/miniconda3/envs/pangu/lib/python3.10/site-packages/nvidia/cudnn/lib',
    '/home/tx/miniconda3/envs/pangu/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib',
    '/home/tx/miniconda3/envs/pangu/lib/python3.10/site-packages/nvidia/cufft/lib',
    '/home/tx/miniconda3/envs/pangu/lib/python3.10/site-packages/nvidia/cusolver/lib',
    '/home/tx/miniconda3/envs/pangu/lib/python3.10/site-packages/nvidia/cusparse/lib',
    '/home/tx/miniconda3/envs/pangu/lib/python3.10/site-packages/nvidia/curand/lib',
    '/home/tx/miniconda3/envs/pangu/lib/python3.10/site-packages/nvidia/nvjitlink/lib',
]
existing_ld = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = ':'.join(nvidia_lib_paths) + ':' + existing_ld
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

BASE_DIR = '/mnt/data/workspace/Pangu-Weather'
input_data_dir = os.path.join(BASE_DIR, 'input_data')
output_data_dir = os.path.join(BASE_DIR, 'output_data')

# 加载输入数据
input_upper = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)
print(f"输入高空场: shape={input_upper.shape}")
print(f"输入地面场: shape={input_surface.shape}")

var_names_upper = ['Z(位势)', 'Q(比湿)', 'T(温度)', 'U(纬向风)', 'V(经向风)']
var_names_surface = ['MSLP(地面气压)', 'U10(10m纬向风)', 'V10(10m经向风)', 'T2M(2m温度)']

def create_session(model_file):
    """创建onnxruntime会话"""
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 4
    cuda_opts = {'arena_extend_strategy': 'kSameAsRequested'}
    sess = ort.InferenceSession(
        os.path.join(BASE_DIR, model_file),
        sess_options=options,
        providers=[('CUDAExecutionProvider', cuda_opts), 'CPUExecutionProvider']
    )
    return sess

# ====== 迭代推理：7天预报（24h+6h，逐步加载模型） ======
print(f"\n{'='*60}")
print("迭代推理: 7天预报（24h + 6h模型，逐步加载以节省显存）")
print(f"{'='*60}")

current_upper = input_upper.copy()
current_surface = input_surface.copy()
input_24_upper = input_upper.copy()
input_24_surface = input_surface.copy()

total_start = time.time()
total_load_time = 0
total_inf_time = 0

for i in range(28):
    step_num = i + 1
    use_24h = (step_num % 4 == 0)
    model_file = 'pangu_weather_24.onnx' if use_24h else 'pangu_weather_6.onnx'
    model_tag = "24h" if use_24h else "6h"

    # 加载模型
    t0 = time.time()
    sess = create_session(model_file)
    load_time = time.time() - t0
    total_load_time += load_time

    # 运行推理
    if use_24h:
        t0 = time.time()
        output, output_surface = sess.run(None, {'input': input_24_upper, 'input_surface': input_24_surface})
        inf_time = time.time() - t0
        input_24_upper = output.copy()
        input_24_surface = output_surface.copy()
    else:
        t0 = time.time()
        output, output_surface = sess.run(None, {'input': current_upper, 'input_surface': current_surface})
        inf_time = time.time() - t0

    total_inf_time += inf_time

    current_upper = output.copy()
    current_surface = output_surface.copy()

    # 释放模型显存
    del sess
    gc.collect()

    # 保存结果
    np.save(os.path.join(output_data_dir, f'output_upper_iter_step{step_num}'), output)
    np.save(os.path.join(output_data_dir, f'output_surface_iter_step{step_num}'), output_surface)

    elapsed = time.time() - total_start
    print(f"步骤 {step_num}/28: {model_tag}模型 | 加载{load_time:.2f}s + 推理{inf_time:.2f}s | 已用{elapsed:.1f}s")

total_time = time.time() - total_start
print(f"\n{'='*60}")
print("迭代推理完成")
print(f"{'='*60}")
print(f"总耗时: {total_time:.2f}s (加载{total_load_time:.2f}s + 推理{total_inf_time:.2f}s)")
print(f"最终输出高空场: {current_upper.shape}")
print(f"最终输出地面场: {current_surface.shape}")
for i, name in enumerate(var_names_upper):
    print(f"  {name}: min={current_upper[i].min():.4f}, max={current_upper[i].max():.4f}, mean={current_upper[i].mean():.4f}")
for i, name in enumerate(var_names_surface):
    print(f"  {name}: min={current_surface[i].min():.4f}, max={current_surface[i].max():.4f}, mean={current_surface[i].mean():.4f}")
print(f"\n28步结果保存在 {output_data_dir}/")
