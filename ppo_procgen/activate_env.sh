module load StdEnv/2020  gcc/9.3.0
module load cuda/11.8.0
module load opencv/4.8.0
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

source ~/jaxenv/bin/activate