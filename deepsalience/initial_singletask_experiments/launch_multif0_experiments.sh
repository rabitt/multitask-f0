for n in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17; do
echo "#!/bin/bash
#
#SBATCH --job-name=exper$n
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=50GB
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=exper$n_%j.out

module purge
module load cuda/8.0.44
module load cudnn/8.0v5.1
module load ffmpeg/intel/3.2.2

source ~/.bashrc
unset XDG_RUNTIME_DIR

cd ~/repos/multif0/deepsalience

python multif0_exper${n}.py
" >> exp_multif0_${n}.s

sbatch exp_multif0_${n}.s
done

for n in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17; do
echo "#!/bin/bash
#
#SBATCH --job-name=exper${n}_b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=50GB
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=exper${n}_b_%j.out

module purge
module load cuda/8.0.44
module load cudnn/8.0v5.1
module load ffmpeg/intel/3.2.2

source ~/.bashrc
unset XDG_RUNTIME_DIR

cd ~/repos/multif0/deepsalience

python multif0_exper${n}_batchin.py
" >> exp_multif0_${n}_batchin.s

sbatch exp_multif0_${n}_batchin.s
done