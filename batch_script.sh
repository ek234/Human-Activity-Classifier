#!/bin/zsh
#SBATCH -A research
#SBATCH -n 15
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file.txt

module load cuda/10.1
module load cudnn/7-cuda-10.1

micromamba activate smaiP
./run_all.sh
