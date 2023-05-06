#!/bin/zsh
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file.txt

./run_svm.sh
