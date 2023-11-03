#!/bin/bash
#
#SBATCH --job-name=cross-pow
#SBATCH --account=cosmo_ai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:30:00
source /home/dongx/.bashrc
conda activate /home/dongx/anaconda3/envs/env_pytorch
cd /lcrc/project/cosmo_ai/dongx/plot-folder/PM-plot/cross-pow/
export PYTHONPATH="$PWD/Unet"
python cross-pow-nbdy-testpowinstead.py