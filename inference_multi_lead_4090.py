import os
import numpy as np
import onnxruntime as ort
import time

# 设置LD_LIBRARY_PATH使onnxruntime找到CUDA库
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

# 指定使用GPU 1（RTX 4090）
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

BASE_DIR = '/mnt/data/workspace/Pangu-Weather'
input_data_dir = os.path.join(BASE_DIR, 'input_data')
output_data_dir = os.path.join(BASE_DIR, 'output_data')

# 加载输入数据
input_upper = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)
print(f"输入高空场: shape={input_upper.shape}, dtype={input_upper.dtype}")
print(f"输入地面场: shape={input_surface.shape}, dtype={input_surface.dtype}")

# Session options
options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
options.intra_op_num_threads = 4
options.inter_op_num_threads = 4

cuda_provider_options = {'arena_extend_strategy': 'kSameAsRequested'}

var_names_upper = ['Z(位势)', 'Q(比湿)', 'T(温度)', 'U(纬向风)', 'V(经向风)']
var_names_surface = ['MSLP(地面气压)', 'U10(10m纬向风)', 'V10(10m经向风)', 'T2M(2m温度)']

def run_single_inference(model_name, model_file, input_upper, input_surface):
    """运行单个模型推理，完成后释放GPU显存"""
    print(f"\n{'='*60}")
    print(f"单步推理: {model_name}")
    print(f"{'='*60}")

    t0 = time.time()
    sess = ort.InferenceSession(
        os.path.join(BASE_DIR, model_file),
        sess_options=options,
        providers=[('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider']
    )
    load_time = time.time() - t0
    print(f"模型加载耗时: {load_time:.2f}s, Provider: {sess.get_providers()}")

    t0 = time.time()
    output, output_surface = sess.run(None, {'input': input_upper, 'input_surface': input_surface})
    inf_time = time.time() - t0
    print(f"推理耗时: {inf_time:.2f}s")
    print(f"输出高空场: {output.shape}, 地面场: {output_surface.shape}")

    # 输出数值验证
    for i, name in enumerate(var_names_upper):
        print(f"  {name}: min={output[i].min():.4f}, max={output[i].max():.4f}, mean={output[i].mean():.4f}")
    for i, name in enumerate(var_names_surface):
        print(f"  {name}: min={output_surface[i].min():.4f}, max={output_surface[i].max():.4f}, mean={output_surface[i].mean():.4f}")

    # 释放GPU显存 - 删除session对象
    del sess
    import gc; gc.collect()

    return output, output_surface, load_time, inf_time

# ====== 单步推理：各时效模型 ======
models = [
    ('6小时预报 (pangu_weather_6.onnx)', 'pangu_weather_6.onnx', '6h'),
    ('3小时预报 (pangu_weather_3.onnx)', 'pangu_weather_3.onnx', '3h'),
    ('1小时预报 (pangu_weather_1.onnx)', 'pangu_weather_1.onnx', '1h'),
    ('24小时预报 (pangu_weather_24.onnx)', 'pangu_weather_24.onnx', '24h'),
]

results_summary = []
for desc, file, tag in models:
    output, output_surface, load_t, inf_t = run_single_inference(desc, file, input_upper, input_surface)
    np.save(os.path.join(output_data_dir, f'output_upper_{tag}'), output)
    np.save(os.path.join(output_data_dir, f'output_surface_{tag}'), output_surface)
    print(f"{tag}预报已保存到 output_data/")
    results_summary.append((desc, load_t, inf_t))

# ====== 迭代推理：7天预报（24h+6h组合） ======
print(f"\n{'='*60}")
print("迭代推理: 7天预报（24h + 6h模型组合）")
print(f"{'='*60}")

t0 = time.time()
sess_24 = ort.InferenceSession(
    os.path.join(BASE_DIR, 'pangu_weather_24.onnx'),
    sess_options=options,
    providers=[('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider']
)
load_24 = time.time() - t0
print(f"24h模型加载: {load_24:.2f}s")

t0 = time.time()
sess_6 = ort.InferenceSession(
    os.path.join(BASE_DIR, 'pangu_weather_6.onnx'),
    sess_options=options,
    providers=[('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider']
)
load_6 = time.time() - t0
print(f"6h模型加载: {load_6:.2f}s")
print(f"两个模型已加载, Provider: {sess_24.get_providers()}")

current_upper = input_upper.copy()
current_surface = input_surface.copy()
input_24_upper = input_upper.copy()
input_24_surface = input_surface.copy()

total_start = time.time()

for i in range(28):
    step_num = i + 1
    t0 = time.time()

    if step_num % 4 == 0:
        output, output_surface = sess_24.run(
            None, {'input': input_24_upper, 'input_surface': input_24_surface}
        )
        input_24_upper = output.copy()
        input_24_surface = output_surface.copy()
        model_used = "24h"
    else:
        output, output_surface = sess_6.run(
            None, {'input': current_upper, 'input_surface': current_surface}
        )
        model_used = "6h"

    dt = time.time() - t0
    current_upper = output.copy()
    current_surface = output_surface.copy()

    np.save(os.path.join(output_data_dir, f'output_upper_iter_step{step_num}'), output)
    np.save(os.path.join(output_data_dir, f'output_surface_iter_step{step_num}'), output_surface)
    print(f"步骤 {step_num}/28: {model_used}模型 → {dt:.2f}s")

total_time = time.time() - total_start
print(f"\n7天迭代预报总耗时: {total_time:.2f}s")
print(f"最终输出高空场: {current_upper.shape}")
print(f"最终输出地面场: {current_surface.shape}")
for i, name in enumerate(var_names_upper):
    print(f"  {name}: min={current_upper[i].min():.4f}, max={current_upper[i].max():.4f}, mean={current_upper[i].mean():.4f}")
for i, name in enumerate(var_names_surface):
    print(f"  {name}: min={current_surface[i].min():.4f}, max={current_surface[i].max():.4f}, mean={current_surface[i].mean():.4f}")

# 释放迭代模型显存
del sess_24, sess_6
import gc; gc.collect()

# ====== 总结 ======
print(f"\n{'='*60}")
print("所有推理完成 - 性能总结")
print(f"{'='*60}")
print(f"{'模型':<30} {'加载(s)':<10} {'推理(s)':<10}")
print("-"*50)
for desc, load_t, inf_t in results_summary:
    print(f"{desc:<30} {load_t:<10.2f} {inf_t:<10.2f}")
print(f"迭代推理(28步=7天): {total_time:.2f}s")
print(f"\n所有结果保存在 {output_data_dir}/")
