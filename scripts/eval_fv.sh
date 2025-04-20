which python

#!/bin/bash
cuda_id=$1
model=$2
dataset=$3
save_path=$4
cmd=$5

datasets=("${dataset}")
# datasets=('antonym')


if [[ "${model}" == "llama-chat" ]]; then
    model_name='/disk/jianggangwei/Llama-2-7b_chat_hf'
fi


for d_name in "${datasets[@]}"
do
    echo "Running Script for: ${d_name}"
    CUDA_VISIBLE_DEVICES=${cuda_id} python src/fvector/evaluate_function_vector.py \
        --dataset_name=${d_name} \
        --save_path_root=$save_path \
        --model_name=${model_name} \
        --max_eval_size=100 ${cmd}
done