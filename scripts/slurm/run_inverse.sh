#!/bin/bash
#SBATCH --job-name=inverse_0000
#SBATCH -c 4                               # 4 cores per task
#SBATCH -t 00-04:00:00
#SBATCH -o logs/output_%j.log
#SBATCH -e logs/error_%j.log
#SBATCH -p kempner_requeue

#SBATCH --account=kempner_ydu_lab
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1


# Load necessary modules
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh

module load gcc/14.2.0-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01 
export HF_HOME=/n/netscratch/ydu_lab/Lab/alex/.cache/huggingface
export TORCH_HOME=/n/netscratch/ydu_lab/Lab/alex/.cache
CUDA_VISIBLE_DEVICES=0

conda activate wan

echo "Starting Python script..."

PYTHONPATH=. python scripts/inverse_video.py \
    --video-dir samples/0000/ \
    --expected-frames 49 \
    --output results/decoded_0000.txt

# PYTHONPATH=. python scripts/inverse_video.py \
#     --video-dir samples/0000/ \
#     --expected-frames 49 \
#     --output results/decoded_0000.txt
