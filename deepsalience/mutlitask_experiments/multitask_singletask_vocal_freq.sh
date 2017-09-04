#!/bin/bash
#
#SBATCH --job-name=mtask_base
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=14:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=multitask_singletask_vocal_freq_%j.out

module purge
module load cuda/8.0.44
module load cudnn/8.0v5.1
module load ffmpeg/intel/3.2.2

source ~/.bashrc
unset XDG_RUNTIME_DIR

cd ~/repos/multif0/deepsalience/mutlitask_experiments

python multitask_singletask_vocal_freq.py
