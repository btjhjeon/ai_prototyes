#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%j.%x.info.log
#SBATCH --error=logs/%j.%x.error.log

module load Python/3.9.6-GCCcore-11.2.0
source ~/venv/pt_nightly/bin/activate

run_cmd="PWD=$PWD TOKENIZERS_PARALLELISM=false python -m scripts.language.qna.data_aug_with_gpt"

echo $run_cmd

srun -l \
    --output=$PWD/logs/%j.%x.log \
    bash -c "$run_cmd"
set +x
