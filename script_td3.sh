#!/bin/bash

#source activate pyRL  # Activate Conda environment if needed

PHASE=$1   # train/test
GPU_IDS=$2 # GPU ID
NUM_WORKERS=$3 # Number of data loader workers
EXP_TAG=$4 # Experiment tag

LOG_DIR="./logs/${EXP_TAG}"
mkdir -p ${LOG_DIR}  # Create log directory if it doesn't exist

LOG="${LOG_DIR}/${PHASE}_${EXP_TAG}_$(date +'%Y-%m-%d_%H-%M').log"
exec &> >(tee -a "$LOG")  # Save log output

echo "Running TD3 $PHASE with GPU $GPU_IDS..."
CUDA_VISIBLE_DEVICES=$GPU_IDS python main_td3.py --phase ${PHASE} --num_workers ${NUM_WORKERS}

echo "Done!"
