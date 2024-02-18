#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=23:59:59
#SBATCH --mem-per-cpu 16G
#SBATCH --gres=gpu:1

#SBATCH --job-name train_dual_model_sync
#SBATCH --output slurm/train_bg_pfc_combined.out
python -u main.py --env 'hanoi-v0' --testenv 'hanoitest-v0' --algo 'Dreamerv2' --exp-name 'default_hp' --train --eval 
