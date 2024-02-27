#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=11:40:00
#SBATCH --cpus-per-task=4

source /etc/profile
module add cuda/12.0
module add anaconda3/2023.09

source activate /storage/hpc/00/zhangz65/acaconda3/envs/satlas
python fine_tuning.py
