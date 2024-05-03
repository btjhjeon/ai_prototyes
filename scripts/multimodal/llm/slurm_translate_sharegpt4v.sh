#!/bin/bash

#SBATCH -J translate
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/%j.%x.info.log
#SBATCH --error=logs/%j.%x.error.log

PYTHONPATH=$PYTHONPATH:$PWD python -u -m scripts.multimodal.llm.translate_sharegpt4v \
    -d=/t1data/project/multimodal/dataset/SFT/2024.04.22/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap91k.json \
    -k=conversations.value \
    -o=/t1data/project/multimodal/dataset/SFT/2024.04.22/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap91k_ko.json
