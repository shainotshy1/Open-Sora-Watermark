#!/bin/bash

#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=fSora
#SBATCH --output='fSora.out'

eval "$(micromamba shell hook --shell=bash)"

micromamba activate sora2

VIDEO_DIR=/anvil/scratch/x-sdickman/Open-Sora-Watermark/samples/vbench_samples
OUTPUT_DIR=/anvil/scratch/x-sdickman/Open-Sora-Watermark/vbench_output
# CKPT_DIR=$2
# LOG_BASE=$CKPT_DIR
# mkdir -p $LOG_BASE
# echo "Logging to $LOG_BASE"

# GPUS=(0 1 2 3 4 5 6 7)
IDX=5 # Choose from (0 1 2 3 4 5 6 7)   
START_INDEX_LIST=(0 2 6 7 8 9 10 13)
END_INDEX_LIST=(2 6 7 8 9 10 13 16)
# TASK_ID_LIST=(calc_vbench_a calc_vbench_b calc_vbench_c calc_vbench_d calc_vbench_e calc_vbench_f calc_vbench_g calc_vbench_h) # for log records only

python eval/vbench/calc_vbench.py $VIDEO_DIR $OUTPUT_DIR --start ${START_INDEX_LIST[$IDX]} --end ${END_INDEX_LIST[$IDX]} #> ${LOG_BASE}/${TASK_ID_LIST[i]}.log 2>&1 &