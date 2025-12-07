#!/bin/bash
#SBATCH --job-name=simulate_atk
#SBATCH -c 4                               # 4 cores per task
#SBATCH -t 00-00:30:00
#SBATCH -o logs/output_%j.log
#SBATCH -e logs/error_%j.log
#SBATCH -p seas_compute

#SBATCH --account=ydu_lab
#SBATCH --mem=32GB


# Load necessary modules
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh

module load gcc/14.2.0-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01 
export HF_HOME=/n/netscratch/ydu_lab/Lab/alex/.cache/huggingface
export TORCH_HOME=/n/netscratch/ydu_lab/Lab/alex/.cache
CUDA_VISIBLE_DEVICES=0

conda activate wan

echo "Starting attack simulation..."

# Video indices to process
VIDEOS=(0000)

# Frame counts to clip to (with approximate percentages)
# 39=80%, 34=70%, 29=60%, 25=50%, 20=40%, 15=30%, 10=20%
# FRAMES=(37 36 35 34 33 32 20 19 18 17 16 15 14 13 12 11 10)
FRAMES=(44 39 34 29 25 20 15 10)

# Which frames to save (front=first n frames, back=last n frames)
DIRECTIONS=(front back)

# For clipping attacks
for vid in "${VIDEOS[@]}"; do
    echo "Processing video $vid... CLIPPING ATTACKS"
    for frames in "${FRAMES[@]}"; do
        for dir in "${DIRECTIONS[@]}"; do
            ./scripts/clip_frames.sh \
                "samples/${vid}/prc_${vid}.mp4" \
                "samples/${vid}/prc_${vid}_${frames}frames_${dir}.mp4" \
                "$frames" \
                "$dir"
        done
    done
done

PROBABILITIES=(0.1 0.2 0.25 0.3 0.35 0.4 0.5 0.6 0.65 0.7 0.75 0.8 0.9)

# For random frame drop attacks
for vid in "${VIDEOS[@]}"; do
    echo "Processing video $vid... RANDOM FRAME DROP ATTACKS"
    for p in "${PROBABILITIES[@]}"; do
        ./scripts/random_frame_drop.sh \
            "samples/${vid}/prc_${vid}.mp4" \
            "samples/${vid}/prc_${vid}_random_frame_drop_${p}.mp4" \
            "$p"
    done
done

# For subsampling attacks (keep every Nth frame)
SUBSAMPLE_RATES=(2 3 4 5)
for vid in "${VIDEOS[@]}"; do
    echo "Processing video $vid... SUBSAMPLING ATTACKS"
    for n in "${SUBSAMPLE_RATES[@]}"; do
        ./scripts/subsample_frames.sh \
            "samples/${vid}/prc_${vid}.mp4" \
            "samples/${vid}/prc_${vid}_subsample_${n}x.mp4" \
            "$n"
    done
done

# # For frame interpolation attacks (RIFE)
# RIFE_DIR="../Practical-RIFE"
# RIFE_MODEL="/n/netscratch/ydu_lab/Lab/alex/RIFE"
# for vid in "${VIDEOS[@]}"; do
#     echo "Processing video $vid... FRAME INTERPOLATION ATTACK 2x"
#     python3 "${RIFE_DIR}/inference_video.py" \
#         --video "samples/${vid}/prc_${vid}.mp4" \
#         --output "samples/${vid}/prc_${vid}_rife_2x.mp4" \
#         --model "${RIFE_MODEL}" \
#         --multi 2 \
#         --fps 48
#     echo "Processing video $vid... FRAME INTERPOLATION ATTACK 4x"
#     python3 "${RIFE_DIR}/inference_video.py" \
#         --video "samples/${vid}/prc_${vid}.mp4" \
#         --output "samples/${vid}/prc_${vid}_rife_4x.mp4" \
#         --model "${RIFE_MODEL}" \
#         --multi 4 \
#         --fps 96
# done

echo "Done!"