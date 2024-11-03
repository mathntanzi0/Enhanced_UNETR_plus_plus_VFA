# Set the path to GCC 9.2.0 binaries
export PATH=/apps/compilers/gcc/9.2.0/bin:$PATH

# Set the library path for GCC 9.2.0
export LD_LIBRARY_PATH=/apps/compilers/gcc/9.2.0/lib:/apps/compilers/gcc/9.2.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/9.2.0/gmp-6.2.1/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/9.2.0/mpfr-4.1.0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/9.2.0/mpc-1.2.1/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/9.2.0/isl-0.24/lib:$LD_LIBRARY_PATH

# Set the C++ compiler and C compiler for GCC 9.2.0
export CXX=/apps/compilers/gcc/9.2.0/bin/g++
export CC=/apps/compilers/gcc/9.2.0/bin/gcc

# Set CUDA environment variables
export CUDA_HOME=/apps/chpc/cuda/12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Source Conda
source /home/sntanzi/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate unetr_pp

# Set the CUDA architecture list for PyTorch
export TORCH_CUDA_ARCH_LIST="7.0"

# Navigate to the directory containing the CUDA extension
cd /mnt/lustre/users/sntanzi/swattention_extension_3d

# Run
python setup.py