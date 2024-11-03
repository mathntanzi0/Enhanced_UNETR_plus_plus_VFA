import torch
from torch.utils.cpp_extension import load_inline
import time
import os
import shutil

os.environ['CUDA_HOME'] = '/apps/chpc/cuda/12.4'

# Function to read the content of the files
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Read the C++ and CUDA files
cpp_code = read_file('swattention.cpp')
cuda_code_av_fw = read_file('av_fw_kernel.cu')
cuda_code_av_bw = read_file('av_bw_kernel.cu')
cuda_code_qk_rpb_fw = read_file('qk_rpb_fw_kernel.cu')
cuda_code_qk_rpb_bw = read_file('qk_rpb_bw_kernel.cu')

# Combine all CUDA code into one string
cuda_code = f"""
{cuda_code_av_fw}
{cuda_code_av_bw}
{cuda_code_qk_rpb_fw}
{cuda_code_qk_rpb_bw}
"""

# Track compilation time
start_time = time.time()

# Use load_inline to compile and load the extension
swattention = load_inline(
    name='swattention',
    cpp_sources=[cpp_code],
    cuda_sources=[cuda_code],
    extra_cflags=['-O2'],  # Optional: optimization flags for the C++ compiler
    extra_cuda_cflags=['-O2'],  # Optional: optimization flags for the CUDA compiler
)

# Save the compiled shared object to a known location
extension_path = f'{swattention.__file__}'
shutil.copy(extension_path, 'swattention_compiled.so')

# Calculate the compilation time
compilation_time = time.time() - start_time
print(f"Compilation completed in {compilation_time:.2f} seconds.")



# # Test the compiled function
# B, N, D, H, W, C, kernel_size, cuda_threads = 1, 1, 24, 24, 24, 64, 3, 128

# depth = D
# height = H
# width = W
# L = depth * height * width

# queries = torch.rand((B, N, L, 64), device="cuda", dtype=torch.float32)
# keys = torch.rand((B, N, L, 64), device="cuda", dtype=torch.float32)
# rpb = torch.rand((N, kernel_size * kernel_size * kernel_size), device="cuda", dtype=torch.float32)

# start_time = time.time()

# # Testing qk_rpb_forward
# output_qk_rpb = swattention.qk_rpb_forward(queries, keys, rpb, height, width, depth, kernel_size, cuda_threads)
# print("qk_rpb_forward output:", output_qk_rpb)

# d_attn_weight = torch.rand_like(output_qk_rpb)
# # Testing qk_rpb_backward
# output_qk_rpb_bw = swattention.qk_rpb_backward(d_attn_weight, queries, keys, height, width, depth, kernel_size, cuda_threads)
# print("qk_rpb_backward output:", output_qk_rpb_bw)

# # Testing av_forward
# values = torch.rand((B, N, L, C), device="cuda", dtype=torch.float32)
# output_av = swattention.av_forward(output_qk_rpb, values, height, width, depth, kernel_size, cuda_threads)
# print("av_forward output:", output_av)

# # Testing av_backward
# d_output = torch.rand_like(output_av)
# output_av_bw = swattention.av_backward(d_output, output_qk_rpb, values, height, width, depth, kernel_size, cuda_threads)
# print("av_backward output:", output_av_bw)

# execution_time = time.time() - start_time
# print(f"execution_time: {execution_time:.2f} seconds")