#!/bin/bash
#SBATCH --job-name=train_original
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --error=original.err
#SBATCH --output=original.out

source activate dl

cd ../patchprot

PROT train -c "configs/esm2_original.yml"

Echo "done"
