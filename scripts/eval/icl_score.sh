exp_name=$1
adapter_path=$2
cuda=$3

model=llama-chat

generaldata=coqa,hellaswag,obqa,nq,lambada,alpaca,ob_count

icldata=icl_coqa,icl_hellaswag,icl_obqa,icl_nq,icl_lambada,icl_alpaca,icl_ob_count


bash scripts/eval_model.sh ${cuda} ${model} \
    ${icldata}  ./results/${exp_name} \
    "--adapter_name_or_path ${adapter_path} --exp_name icl_f5 --prefixes_type N --separators_type N  --max_eval_size 190  --generate_str"


wait

