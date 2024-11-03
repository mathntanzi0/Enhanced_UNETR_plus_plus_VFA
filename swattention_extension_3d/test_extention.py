import torch
import importlib.util
import time

so_file_path = '/mnt/lustre/users/sntanzi/swattention_extension_3d/swattention_compiled.so'

# Load the compiled extension
spec = importlib.util.spec_from_file_location('swattention', so_file_path)
swattention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(swattention)

# Test the compiled function
B, N, D, H, W, C, kernel_size, cuda_threads = 1, 1, 24, 24, 24, 64, 3, 128

depth = D
height = H
width = W
L = depth * height * width

queries = torch.rand((B, N, L, 64), device="cuda", dtype=torch.float32)
keys = torch.rand((B, N, L, 64), device="cuda", dtype=torch.float32)
rpb = torch.rand((N, kernel_size * kernel_size * kernel_size), device="cuda", dtype=torch.float32)

start_time = time.time()

# Testing qk_rpb_forward
output_qk_rpb = swattention.qk_rpb_forward(queries, keys, rpb, height, width, depth, kernel_size, cuda_threads)
print("qk_rpb_forward output:", output_qk_rpb)

d_attn_weight = torch.rand_like(output_qk_rpb)
# Testing qk_rpb_backward
output_qk_rpb_bw = swattention.qk_rpb_backward(d_attn_weight, queries, keys, height, width, depth, kernel_size, cuda_threads)
print("qk_rpb_backward output:", output_qk_rpb_bw)

# Testing av_forward
values = torch.rand((B, N, L, C), device="cuda", dtype=torch.float32)
output_av = swattention.av_forward(output_qk_rpb, values, height, width, depth, kernel_size, cuda_threads)
print("av_forward output:", output_av)

# Testing av_backward
d_output = torch.rand_like(output_av)
output_av_bw = swattention.av_backward(d_output, output_qk_rpb, values, height, width, depth, kernel_size, cuda_threads)
print("av_backward output:", output_av_bw)

execution_time = time.time() - start_time
print(f"execution_time: {execution_time:.2f} seconds")