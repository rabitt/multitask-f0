for n in 1 2 3 4 5 6; do
echo "#!/bin/bash
#
#SBATCH --job-name=nharm${n}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=50GB
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=nharm${n}_%j.out

module purge
module load cuda/8.0.44
module load cudnn/8.0v5.1
module load ffmpeg/intel/3.2.2

source ~/.bashrc
unset XDG_RUNTIME_DIR

cd ~/repos/multif0/deepsalience

python multif0_nharms_experiment.py $n
" >> exp_nharm${n}.s

sbatch exp_nharm${n}.s
done
