#!/bin/bash
#SBATCH --job-name=bbox
#SBATCH -c 4                               # 4 cores per task
#SBATCH -t 00-00:30:00
#SBATCH -o logs/output_%j.log
#SBATCH -e logs/error_%j.log
#SBATCH -p seas_gpu

#SBATCH --account=ydu_lab
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

PYTHONPATH=. python scripts/inference.py configs/opensora-v1-3/inference/t2v.py \
    --num-frames 49 --resolution 360p --aspect-ratio 9:16 \
    --prompt-path scripts/txt/prompts.txt \
    --ckpt-path "/n/netscratch/ydu_lab/Lab/alex/OpenSora-STDiT-v4-360p" \
    --layernorm-kernel False \
    --prc True \
    --sample-name prc \
    --index-as-dir True

# python scripts/clip_video.py ./samples/samples/sample_waterfall_prc_0000.mp4 ./samples/samples/clipped_27.mp4 -f 27
# #
# PYTHONPATH=. python scripts/inverse_video.py \
#     ./samples/samples/sample_waterfall_prc_0000.mp4 \
#     ./samples/samples/clipped_47.mp4 \
#     ./samples/samples/clipped_44.mp4 \
#     ./samples/samples/clipped_42.mp4 \
#     ./samples/samples/clipped_39.mp4 \
#     ./samples/samples/clipped_37.mp4 \
#     ./samples/samples/clipped_34.mp4 \
#     ./samples/samples/clipped_32.mp4 \
#     ./samples/samples/clipped_29.mp4 \
#     ./samples/samples/clipped_27.mp4 \
#     ./samples/samples/clipped_25.mp4 \
#     ./samples/samples/clipped_22.mp4 \
#     ./samples/samples/clipped_20.mp4 \
#     ./samples/samples/clipped_17.mp4 \
#     ./samples/samples/clipped_15.mp4 \
#     ./samples/samples/clipped_12.mp4 \
#     --expected-frames 49 \
#     --output decoded_results.txt
