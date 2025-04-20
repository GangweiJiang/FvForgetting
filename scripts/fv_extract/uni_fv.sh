data=$1
model=llama-chat


if [[ "${data}" == "seq0" ]]; then
    data1=ni618 
    data2=ni1290 
    data3=ni589 
    data4=ni511
    data5=ni1357

elif [[ "${data}" == "seq1" ]]; then
    data1=ni195  
    data2=ni1343
    data3=ni1310
    data4=ni1292
    data5=ni363
elif [[ "${data}" == "seq2" ]]; then
    data1=ni360 
    data2=ni363 
    data3=ni1290  
    data4=ni339
    data5=ni1510 
fi


bash scripts/eval_fv.sh 0 ${model} ${data1}_icl ./results/function_vector \
    "--universal_set --exp_name uni  --prefixes_type N --separators_type N  --max_eval_size 100  --gen --no_eval" &

bash scripts/eval_fv.sh 1 ${model} ${data2}_icl ./results/function_vector \
    "--universal_set --exp_name uni  --prefixes_type N --separators_type N  --max_eval_size 100  --gen --no_eval" &

bash scripts/eval_fv.sh 2 ${model} ${data3}_icl ./results/function_vector \
    "--universal_set --exp_name uni  --prefixes_type N --separators_type N  --max_eval_size 100  --gen --no_eval" &

bash scripts/eval_fv.sh 3 ${model} ${data4}_icl ./results/function_vector \
    "--universal_set --exp_name uni  --prefixes_type N --separators_type N  --max_eval_size 100  --gen --no_eval" &

bash scripts/eval_fv.sh 4 ${model} ${data5}_icl ./results/function_vector \
    "--universal_set --exp_name uni  --prefixes_type N --separators_type N  --max_eval_size 100  --gen --no_eval" &