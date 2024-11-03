#include <torch/extension.h>
#include <cmath>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/AccumulateType.h>

template <typename scalar_t>
__global__ void rpb_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_attn_weight, // B, H, L, span
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> d_rpb,
    int height,
    int width,
    int depth,
    int kernel_size,
    int d_rpb_numel
){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < d_attn_weight.size(1)) {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < d_attn_weight.size(2)) {
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < d_attn_weight.size(3)) {
                const int i = y / (width * depth);
                const int j = (y / depth) % width;
                const int k = y % depth;
                const int ki = z / (kernel_size * kernel_size);
                const int kj = (z / kernel_size) % kernel_size;
                const int kk = z % kernel_size;
                const int ni = i + ki - (kernel_size - 1) / 2;
                const int nj = j + kj - (kernel_size - 1) / 2;
                const int nk = k + kk - (kernel_size - 1) / 2;
                
                scalar_t updt = scalar_t(0);
                if (((ni >= 0) && (ni < height)) && ((nj >= 0) && (nj < width)) && ((nk >= 0) && (nk < depth))) {
                    #pragma unroll
                    for (int b = 0; b < d_attn_weight.size(0); ++b)
                        updt += d_attn_weight[b][x][y][z];
                }
                const int index = x * d_rpb.size(1) + z;
                at::native::fastAtomicAdd(d_rpb.data(), index, d_rpb_numel, static_cast<scalar_t>(updt), true);
            }
        }
    }
}

template <typename scalar_t>
__global__ void qk_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_attn_weight,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> keys,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_queries,
    int height,
    int width,
    int depth,
    int kernel_size
){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < (keys.size(0) * keys.size(1))) {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < keys.size(2)) {
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < keys.size(3)) {
                const int b = x / keys.size(1);
                const int h = x - b * keys.size(1);
                const int i = y / (width * depth);
                const int j = (y / depth) % width;
                const int k = y % depth;
                const int start_i = i - (kernel_size - 1) / 2;
                const int start_j = j - (kernel_size - 1) / 2;
                const int start_k = k - (kernel_size - 1) / 2;
                
                scalar_t updt = scalar_t(0);
                int k_offset = 0;

                #pragma unroll
                for (int current_i = start_i; current_i < (start_i + kernel_size); ++current_i) {
                    #pragma unroll
                    for (int current_j = start_j; current_j < (start_j + kernel_size); ++current_j) {
                        #pragma unroll
                        for (int current_k = start_k; current_k < (start_k + kernel_size); ++current_k) {
                            if (((current_i >= 0) && (current_i < height)) && ((current_j >= 0) && (current_j < width)) && ((current_k >= 0) && (current_k < depth))) {
                                const int current_offset = current_i * (width * depth) + current_j * depth + current_k;
                                updt += d_attn_weight[b][h][y][k_offset] * keys[b][h][current_offset][z];
                            }
                            ++k_offset;
                        }
                    }
                }
                d_queries[b][h][y][z] = updt;
            }
        }
    }
}

template <typename scalar_t>
__global__ void qk_inverse_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_attn_weight,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> queries,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_keys,
    int height,
    int width,
    int depth,
    int kernel_size
){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < (d_keys.size(0) * d_keys.size(1))) {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < d_keys.size(2)) {
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < d_keys.size(3)) {
                const int b = x / d_keys.size(1);
                const int h = x - b * d_keys.size(1);
                const int i = y / (width * depth);
                const int j = (y / depth) % width;
                const int k = y % depth;
                const int q_start_i = i - kernel_size / 2;
                const int q_end_i = i + 1 + (kernel_size - 1) / 2;
                const int q_start_j = j - kernel_size / 2;
                const int q_end_j = j + 1 + (kernel_size - 1) / 2;
                const int q_start_k = k - kernel_size / 2;
                const int q_end_k = k + 1 + (kernel_size - 1) / 2;
                
                scalar_t updt = scalar_t(0);
                int k_offset = kernel_size * kernel_size * kernel_size;

                #pragma unroll
                for (int current_i = q_start_i; current_i < q_end_i; ++current_i) {
                    #pragma unroll
                    for (int current_j = q_start_j; current_j < q_end_j; ++current_j) {
                        #pragma unroll
                        for (int current_k = q_start_k; current_k < q_end_k; ++current_k) {
                            --k_offset;
                            if (((current_i >= 0) && (current_i < height)) && ((current_j >= 0) && (current_j < width)) && ((current_k >= 0) && (current_k < depth))) {
                                const int current_offset = current_i * (width * depth) + current_j * depth + current_k;
                                updt += d_attn_weight[b][h][current_offset][k_offset] * queries[b][h][current_offset][z];
                            }
                        }
                    }
                }
                d_keys[b][h][y][z] = updt;
            }
        }
    }
}

std::vector<torch::Tensor> qk_rpb_bw_cu(
    const torch::Tensor d_attn_weight,
    const torch::Tensor queries,
    const torch::Tensor keys,
    int height,
    int width,
    int depth,
    int kernel_size,
    int cuda_threads
){
    TORCH_CHECK((cuda_threads > 0) && (cuda_threads <= 1024), "The value of CUDA_NUM_THREADS should be between 1 and 1024");
    TORCH_CHECK(queries.size(0) == keys.size(0), "Batch size mismatch");
    TORCH_CHECK(queries.size(1) == keys.size(1), "Number of attention heads mismatch");
    TORCH_CHECK(queries.size(2) == (height * width * depth), "Sequence length mismatch");

    auto d_queries = torch::zeros_like(queries);
    auto d_keys = torch::zeros_like(keys);
    auto d_rpb = torch::zeros({d_attn_weight.size(1), d_attn_weight.size(3)}, d_attn_weight.options());

    const int block_x = std::min<int>(cuda_threads, queries.size(0) * queries.size(1));
    const int block_y = std::min<int>(cuda_threads / block_x, queries.size(2));
    const int block_z = std::min<int>(cuda_threads / (block_x * block_y), queries.size(3));

    dim3 threads(block_x, block_y, block_z);

    const int grid_x = (queries.size(0) * queries.size(1) + block_x - 1) / block_x;
    const int grid_y = (queries.size(2) + block_y - 1) / block_y;
    const int grid_z = (queries.size(3) + block_z - 1) / block_z;

    dim3 blocks(grid_x, grid_y, grid_z);

    AT_DISPATCH_FLOATING_TYPES(queries.scalar_type(), "rpb_bw_kernel", ([&] {
        rpb_bw_kernel<scalar_t><<<blocks, threads>>>(
            d_attn_weight.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            d_rpb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            height, width, depth, kernel_size, d_rpb.numel()
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(queries.scalar_type(), "qk_bw_kernel", ([&] {
        qk_bw_kernel<scalar_t><<<blocks, threads>>>(
            d_attn_weight.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            keys.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            d_queries.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            height, width, depth, kernel_size
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(queries.scalar_type(), "qk_inverse_bw_kernel", ([&] {
        qk_inverse_bw_kernel<scalar_t><<<blocks, threads>>>(
            d_attn_weight.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            queries.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            d_keys.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            height, width, depth, kernel_size
        );
    }));

    return {d_queries, d_keys, d_rpb};
}