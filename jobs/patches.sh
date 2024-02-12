#!/bin/bash

#SBATCH --job-name=train_patches
#SBATCH --partition=defq
#SBATCH --gpus=A30:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=80:00:00
#SBATCH --error=patches.err
#SBATCH --output=patches.out

export XLA_FLAGS="--xla_gpu_cuda_data_dir=/scistor/informatica/emi232/anaconda3/envs/dl2022/"
module load shared
module load 2022
source activate dl2022

cd ../PROT

pip install click
pip install fair-esm
pip install .

PROT train -c "configs/patches.yml"

echo "done"

