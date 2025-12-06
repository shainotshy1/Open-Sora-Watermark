#!/bin/bash

#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --job-name=5aSora
#SBATCH --output='5aSora.out'

eval "$(micromamba shell hook --shell=bash)"

micromamba activate sora2

CKPT=hpcai-tech/OpenSora-VAE-v1.3
NUM_FRAMES=49
MODEL_NAME=OpenSoraVAE_V1_3
RES=360p
ASP_RATIO=9:16

NUM_SAMPLING_STEPS=30

if [[ $CKPT == *"ema"* ]]; then
    parentdir=$(dirname $CKPT)
    CKPT_BASE=$(basename $parentdir)_ema
else
    CKPT_BASE=$(basename $CKPT)
fi
# # LOG_BASE=$(dirname $CKPT)/eval
# LOG_BASE=./sample/eval

# mkdir -p ${LOG_BASE}

# echo "Logging to $LOG_BASE"

IDX=0 # Choose from (0 1 2 3 4 5 6 7)
TASK_ID_LIST=(4a 4b 4c 4d 4e 4f 4g 4h) # for log records only
START_INDEX_LIST=(0 120 240 360 480 600 720 840)
END_INDEX_LIST=(120 240 360 480 600 720 840 2000)

script=/anvil/scratch/x-sdickman/Open-Sora-Watermark/eval/sample.sh
bash $script ${START_INDEX_LIST[$IDX]} ${END_INDEX_LIST[$IDX]} # Using hardcoded script in sample.sh
# bash $script $CKPT ${NUM_FRAMES} ${MODEL_NAME} -4 ${START_INDEX_LIST[$IDX]} ${END_INDEX_LIST[$IDX]} ${RES} ${ASP_RATIO} ${NUM_SAMPLING_STEPS} #> ${LOG_BASE}/${TASK_ID_LIST[IDX]}.log 2>&1 &