mkdir -p logs/
mkdir -p outputs/

data_path=$PWD/2023.json

for num_shot in -1 0 1
do

# SKT LLM 7B (2022.12.15) -> 보고용은 (230503)으로 표현
gres="gpu:1"
model_name=gpt3_7B_2022.12.15
job_name="${model_name}_${num_shot}"
args=" \
    --data_path=$data_path \
    --model=hf \
    --model_path=/t1data/project/checkpoints/$model_name/hf \
    --num_shot=$num_shot \
    --bf16"

sbatch --job-name "$job_name" --gres $gres --partition batch \
    $PWD/scripts/language/korean_sat/run_slurm.sh $args

# SKT LLM 39B (20230503) -> 보고용은 (230503)으로 표현
gres="gpu:4"
model_name=gpt3_39B_20230503
job_name="${model_name}_${num_shot}"
args=" \
    --data_path=$data_path \
    --model=hf \
    --model_path=/t1data/project/checkpoints/$model_name/hf \
    --num_shot=$num_shot \
    --bf16"

sbatch --job-name "$job_name" --gres $gres --partition batch \
    $PWD/scripts/language/korean_sat/run_slurm.sh $args

# SKT LLM 7B (230807)
gres="gpu:1"
model_name=gpt3_7B_230807
job_name="${model_name}_${num_shot}"
args=" \
    --data_path=$data_path \
    --model=hf \
    --model_path=/t1data/checkpoints/$model_name/hf \
    --num_shot=$num_shot \
    --bf16"

sbatch --job-name "$job_name" --gres $gres --partition batch \
    $PWD/scripts/language/korean_sat/run_slurm.sh $args

# SKT LLM 39B (r230831) -> 보고용은 (230831)로 표현
gres="gpu:4"
model_name=gpt3_39B_r230831
job_name="${model_name}_${num_shot}"
args=" \
    --data_path=$data_path \
    --model=hf \
    --model_path=/t1data/checkpoints/$model_name/hf \
    --num_shot=$num_shot \
    --bf16"

sbatch --job-name "$job_name" --gres $gres --partition batch \
    $PWD/scripts/language/korean_sat/run_slurm.sh $args

# Llama-2 70B (FM)
gres="gpu:4"
model_name=llama-2-70b-hf-muzi
job_name="${model_name}_${num_shot}"
args=" \
    --data_path=$data_path \
    --model=hf \
    --model_path=/sas_data/project/LLMP/model/llama-2/llama-2-70b-hf-muzi \
    --model_max_length=4096 \
    --num_shot=$num_shot \
    --fp16"

sbatch --job-name "$job_name" --gres $gres --partition batch \
    $PWD/scripts/language/korean_sat/run_slurm.sh $args

# GPT-4
gres="gpu:1"
model_name=gpt-4
job_name="${model_name}_${num_shot}"
args=" \
    --data_path=$data_path \
    --model=openai \
    --model_name=gpt-4 \
    --num_shot=$num_shot"

sbatch --job-name "$job_name" --gres $gres --partition batch \
    $PWD/scripts/language/korean_sat/run_slurm.sh $args

# ChatGPT
gres="gpu:1"
model_name=gpt-3.5-turbo
job_name="${model_name}_${num_shot}"
args=" \
    --data_path=$data_path \
    --model=openai \
    --model_name=gpt-3.5-turbo \
    --num_shot=$num_shot"

sbatch --job-name "$job_name" --gres $gres --partition batch \
    $PWD/scripts/language/korean_sat/run_slurm.sh $args

# Claude 2
gres="gpu:1"
model_name=claude-2
job_name="${model_name}_${num_shot}"
args=" \
    --data_path=$data_path \
    --model=anthropic \
    --model_name=claude-2 \
    --num_shot=$num_shot"

sbatch --job-name "$job_name" --gres $gres --partition batch \
    $PWD/scripts/language/korean_sat/run_slurm.sh $args

done
