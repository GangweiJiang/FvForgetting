exp_name=$1
adapter_path=$2
cuda=$3
model=llama-chat

# seq0
data1=ni618 
data2=ni1290 
data3=ni589 
data4=ni511
data5=ni1357


concatdata=${data1}_test,${data2}_test,${data3}_test,${data4}_test,${data5}_test


bash scripts/eval_model.sh ${cuda} ${model} \
    ${concatdata}  ./results/${exp_name} \
    "--adapter_name_or_path ${adapter_path} --exp_name train_f5 --prefixes_type N --separators_type N  --max_eval_size 110  --generate_str"