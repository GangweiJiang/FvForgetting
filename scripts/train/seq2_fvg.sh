set -x
cuda0=0
cuda1=1
cuda2=2
cuda3=3
cuda=${cuda0},${cuda1},${cuda2},${cuda3}
cuda=0,1,2,3

model=llama-chat
seed=$1
algo=naive

data1=ni360 
data2=ni363 
data3=ni1290  
data4=ni339
data5=ni1510 

exp_name=ni_seq2_fvg_${seed}
alpha1=1.0
alpha2=0.1
alpha3=1.0
edit_layer=9

adapter_path1="./results/${exp_name}/1${data1}"
wait
adapter_path2=${adapter_path1}",./results/${exp_name}/2${data2}"
wait
adapter_path3=${adapter_path2}",./results/${exp_name}/3${data3}"
wait
adapter_path4=${adapter_path3}",./results/${exp_name}/4${data4}"
wait
adapter_path5=${adapter_path4}",./results/${exp_name}/5${data5}"
wait

bash scripts/run.sh ${cuda} ${model} \
    ${data1} ${algo} \
    ${exp_name}/1${data1} \
    "--pr_alpha ${alpha3} --pr_loss_type ind --local_model llama --fv_pr --seed ${seed} --fv_kl --edit_layer ${edit_layer} --kl_alpha1 ${alpha1} --kl_alpha2 ${alpha2} --func_path ./results/function_vector/${data1}_icl/uni_function_vector.pt --max_train_samples 1000 --num_train_epochs 10 " 
wait
for i in $(seq 2 5);
do

    data_name="data$i"
    ind=$((i-1))
    adapter_name="adapter_path$((i-1))"

    bash scripts/run.sh ${cuda} ${model} \
        ${!data_name} ${algo} \
        ${exp_name}/${i}${!data_name} \
        "--pr_alpha ${alpha3} --pr_loss_type ind --local_model llama --fv_pr --create_new_adapter True --adapter_name_or_path ${!adapter_name} --edit_layer ${edit_layer} --seed ${seed} --fv_kl --kl_alpha1 ${alpha1} --kl_alpha2 ${alpha2} --func_path ./results/function_vector/${!data_name}_icl/uni_function_vector.pt  --max_train_samples 1000 --num_train_epochs 10 " 
    wait
done

bash scripts/eval/general_score.sh ${exp_name} ${adapter_path5} ${cuda0} &
bash scripts/eval/icl_score.sh ${exp_name} ${adapter_path5} ${cuda1} &
bash scripts/eval/seq2_score.sh ${exp_name} ${adapter_path5} ${cuda2} &
wait
