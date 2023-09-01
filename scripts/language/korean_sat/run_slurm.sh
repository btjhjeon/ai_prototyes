#!/bin/bash
#SBATCH --output=logs/%j.%x.info.log
#SBATCH --error=logs/%j.%x.error.log

module load Python/3.9.6-GCCcore-11.2.0
source ~/venv/pt_nightly/bin/activate

run_cmd="PWD=$PWD python -m scripts.language.korean_sat.main $@"

echo $run_cmd

srun -l \
    --output=$PWD/logs/%j.%x.log \
    bash -c "$run_cmd"
set +x
