#!/bin/bash

#SBATCH --job-name=predict
#SBATCH --partition=defq
#SBATCH --gpus=A30:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=24:00:00
#SBATCH --error=predict.err
#SBATCH --output=predict.out

export XLA_FLAGS="--xla_gpu_cuda_data_dir=/scistor/informatica/emi232/anaconda3/envs/dl2022/"
module load shared
module load 2022
source activate dl2022

cd ../patchprot

pip install click
pip install fair-esm
pip install .

PROT predict -c /scistor/informatica/dgi460/multitask/patchprot/configs/esm2_multitask.yml -d /scistor/informatica/dgi460/multitask/patchprot/saved/ESM2_multitask/0112-214827/checkpoints/model_best.pth -p "SecondaryFeatures" -i /scistor/informatica/dgi460/multitask/predictions/test.txt

echo "done"
