for file in mel2_exper*.py; do
echo "#!/bin/bash
#
#SBATCH --job-name=${file/.py/}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=50GB
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${file/.py/}_%j.out

module purge
module load cuda/8.0.44
module load cudnn/8.0v5.1
module load ffmpeg/intel/3.2.2

source ~/.bashrc
unset XDG_RUNTIME_DIR

cd ~/repos/multif0/deepsalience

python $file
" >> exp_${file/.py/.s}

sbatch exp_${file/.py/.s}
done
