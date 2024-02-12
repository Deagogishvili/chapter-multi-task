#!/bin/bash

#SBATCH --job-name=aggregation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --error=process.err
#SBATCH --output=process.out

#module load 2023
#module load Anaconda3

source activate dl

echo "preparing"

python model.py 

echo "done"
