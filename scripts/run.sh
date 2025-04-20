#!/bin/bash
set -x

cuda=$1
model=$2
dataset=$3
algo=$4
exp_id=$5

port=$(shuf -i25000-30000 -n1)

# --finetuning_type mylora \

if [[ "${model}" == "llama-chat" ]]; then
    model_name_or_path=/disk/jianggangwei/Llama-2-7b_chat_hf
    lora_module="q_proj,v_proj"
    batch_size=4
    gradient_accumulation_steps=2
    template="vanilla"
fi

data_cmd="--data_dir ./data/train --task_name ${dataset}"
# lr=0
epoch=10
lr=1e-04

if [[ "${dataset:2:5}" == "511" ]]; then
    batch_size=2
    gradient_accumulation_steps=4
fi

project_dir="."
output_dir=${project_dir}/results/${exp_id}
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed --include localhost:${cuda} --master_port $port src/run.py \
   --do_train \
   --model_name_or_path ${model_name_or_path} \
   --output_dir ${output_dir} --overwrite_output_dir \
   --per_device_train_batch_size ${batch_size} \
   --per_device_eval_batch_size 2 \
   --algo ${algo} \
   --lora_target ${lora_module} \
   --bf16 \
   --gradient_accumulation_steps ${gradient_accumulation_steps} \
   --learning_rate ${lr} \
   --num_train_epochs ${epoch} \
   --deepspeed configs/stage2_llama.config \
   --run_name ${exp_id} \
   --max_source_length 512 \
   --max_target_length 128 \
   --generation_max_length 128 \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_ratio 0.1 \
   --logging_strategy steps \
   --logging_steps 20 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0 \
   ${data_cmd}  $6 \
   > ${log_dir}/train.log 2>&1
# sleep 5
# --warmup_ratio 0.1