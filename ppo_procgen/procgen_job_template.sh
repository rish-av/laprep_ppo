#!/bin/bash

#SBATCH --account=def-ebrahimi
#SBATCH --time=6:00:00
#SBATCH --gpus=1
#SBATCH --mem=20G
#SBATCH --output=out/%x_%A.out
#SBATCH --error=out/%x_%A.err
#SBATCH --mail-user=mail.rishav9@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --job-name=dqn_procgen

module load StdEnv/2020  gcc/9.3.0
module load cuda/11.8.0
module load opencv/4.8.0
module load httpproxy
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
source ~/jaxenv/bin/activate
python "${SCRIPT}" --env_name "${ENV_NAME}" --seed "${SEED}" --train_steps 35000000
