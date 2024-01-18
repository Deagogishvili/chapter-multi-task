#!/bin/bash
#SBATCH --job-name=train_multitask
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --error=multitask.err
#SBATCH --output=multitask.out

source activate dl

cd ../patchprot

PROT train -c "configs/esm2_multitask.yml"

Echo "done"
