#!/bin/bash

#SBATCH --job-name=train_patchprot
#SBATCH --partition=gpua100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=80:00:00
#SBATCH --error=patchprot.err
#SBATCH --output=patchprot.out

module load cuda/12.2
module load python/3.9

source patchprot_env/bin/activate

cd ../PROT

pip install click
pip install fair-esm
pip install .

PROT train -c "configs/patchprot.yml"

echo "done"

