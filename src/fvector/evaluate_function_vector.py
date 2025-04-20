import os, json
import torch, numpy as np
import argparse
import sys

sys.path.append("/disk/jianggangwei/fv_guided_traning")

# Include prompt creation helper functions
from src.fvector.utils.prompt_utils import *
from src.fvector.utils.intervention_utils import *
from src.fvector.utils.model_utils import *
from src.fvector.utils.eval_utils import *
from src.fvector.utils.extract_utils import *
from src.fvector.compute_indirect_effect import compute_indirect_effect

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--n_top_heads', help='Number of attenion head outputs used to compute function vector', required=False, type=int, default=10)
    parser.add_argument('--edit_layer', help='Layer for intervention. If -1, sweep over all layers', type=int, required=False, default=-1) # 
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--exp_name', help='Name of model to be loaded', type=str, required=False, default='base')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='./data/fv')
    parser.add_argument('--save_path_root', help='File path to save to', type=str, required=False, default='./results/function_vector/')
    parser.add_argument('--ie_path_root', help='File path to load indirect effects from', type=str, required=False, default=None)
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=43)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mean_activations_path', help='Path to file containing mean_head_activations for the specified task', required=False, type=str, default=None)
    parser.add_argument('--adapter_name_or_path', help='Path to file containing mean_head_activations for the specified task', required=False, type=str, default=None)
    parser.add_argument('--indirect_effect_path', help='Path to file containing indirect_effect scores for the specified task', required=False, type=str, default=None)    
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)    
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type=int, required=False, default=10)
    parser.add_argument('--n_trials', help="Number of in-context prompts to average over for indirect_effect", type=int, required=False, default=20)
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', type=json.loads, required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    parser.add_argument("--prefixes_type", help="Metric to use when evaluating generated strings", type=str, required=False, default="Q")
    parser.add_argument("--separators_type", help="Metric to use when evaluating generated strings", type=str, required=False, default="Q")
    parser.add_argument('--separators', help='Prompt template separators to be used', type=json.loads, required=False, default={"input":"\n", "output":"\n\n", "instructions":""})    
    parser.add_argument('--compute_baseline', help='Whether to compute the model baseline 0-shot -> n-shot performance', type=bool, required=False, default=True)
    parser.add_argument('--generate_str', help='Whether to generate long-form completions for the task', action='store_true', required=False)
    parser.add_argument("--metric", help="Metric to use when evaluating generated strings", type=str, required=False, default="f1_score")
    parser.add_argument("--universal_set", help="Flag for whether to evaluate using the univeral set of heads", action="store_true", required=False)
    parser.add_argument('--max_eval_size', help='Number of seeds', type=int, required=False, default=-1)
    parser.add_argument('--adapter_type', help='Number of seeds', type=str, required=False, default="lora")
    parser.add_argument('--act_type', help='Number of seeds', type=str, required=False, default="head")
    parser.add_argument('--effect_type', help='Number of seeds', type=str, required=False, default="each")
    parser.add_argument('--gen', help='Whether to generate long-form completions for the task', action='store_true', required=False)
    parser.add_argument('--no_eval', help='Whether to generate long-form completions for the task', action='store_true', required=False)
    parser.add_argument('--no_fv', help='Whether to generate long-form completions for the task', action='store_true', required=False)
    parser.add_argument('--no_save', help='Whether to generate long-form completions for the task', action='store_true', required=False)
    
    args = parser.parse_args()  

    dataset_name = args.dataset_name
    model_name = args.model_name
    exp_name = args.exp_name
    root_data_dir = args.root_data_dir
    save_path_root = f"{args.save_path_root}/{dataset_name}"
    ie_path_root = f"{args.ie_path_root}/{dataset_name}" if args.ie_path_root else save_path_root
    seed = args.seed
    device = args.device
    mean_activations_path = args.mean_activations_path
    indirect_effect_path = args.indirect_effect_path
    n_top_heads = args.n_top_heads
    eval_edit_layer = args.edit_layer

    test_split = float(args.test_split)
    n_shots = args.n_shots
    n_trials = args.n_trials
    
    if args.prefixes_type == "Q":
        prefixes = {"input":"Q:", "output":"A:", "instructions":""}
    elif args.prefixes_type == "N":
        prefixes = {"input":"", "output":"", "instructions":""}
    elif args.prefixes_type == "I":
        prefixes = {"input":"Input:", "output":"Output:", "instructions":""}
    else:
        prefixes = {"input":"Question:", "output":"Answer:", "instructions":""}
    if args.separators_type == "N":
        separators = {"input":"", "output":"\n\n", "instructions":""}
    else:
        separators = {"input":"\n", "output":"\n\n", "instructions":""}

    
    print(prefixes)
    compute_baseline = args.compute_baseline
    generate_str = args.generate_str
    metric = args.metric
    universal_set = args.universal_set

    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, args, device=device)
    if args.edit_layer == -1: # sweep over all layers if edit_layer=-1
        eval_edit_layer = [0, model_config['n_layers']]

    # Load the dataset
    print("Loading Dataset")
    set_seed(seed)
    dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)
    dataset['test'].raw_data = dataset['test'].raw_data[-args.max_eval_size:]
    dataset['valid'].raw_data = dataset['valid'].raw_data[:200]
    
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    print('train', len(dataset['train']))
    print('valid', len(dataset['valid']))
    print('test', len(dataset['test']))

    # 1. Compute Model 10-shot Baseline & 2. Filter test set to cases where model gets it correct

    default_activations_path = f'{save_path_root}/{args.exp_name}_mean_head_activations.pt'
    # default_activations_path = f'{save_path_root}/mean_head_activations.pt'

    set_seed(seed)
    # Load or Re-Compute mean_head_activations
    if mean_activations_path is not None and os.path.exists(mean_activations_path):
        mean_activations = torch.load(mean_activations_path)
    elif mean_activations_path is None and os.path.exists(default_activations_path):
        mean_activations = torch.load(default_activations_path)        
    else:
        print("Computing Mean Activations")


        print(f"Filtering Dataset via {n_shots}-shot Eval")
        fs_samples_file_name = f'{save_path_root}/fs_samples_layer_sweep.json'
        print(fs_samples_file_name)
        
        set_seed(seed+42)
        # need to add evaluation on multi-token output sentence

        filter_set_validation = None

        if '100' in args.act_type:
            filter_set_validation = np.arange(min(100,len(dataset['valid'])))
        elif os.path.exists(fs_samples_file_name):
            with open(fs_samples_file_name, 'r') as results_file:
                samples = json.load(results_file)
                cnt = 0
                filter_set_validation = []
                for j in tqdm(range(len(dataset['valid'])), total=len(dataset['valid'])):
                    if dataset['valid'][j]["input"] == samples[cnt]['input']:
                        filter_set_validation.append(j)
                        cnt += 1
                        if cnt == len(samples):
                            break
                print(len(filter_set_validation))
        else:
            if "_icl" in args.dataset_name:
                fs_results_validation = n_shot_eval_no_intervention(dataset=dataset, n_shots=0, model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=True, test_split='valid', prefixes=prefixes, separators=separators)
            else:
                fs_results_validation = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=True, test_split='valid', prefixes=prefixes, separators=separators)
            
            if args.gen:
                filter_cand_validation = np.where(np.array(fs_results_validation['clean_rank_list']) < 10)[0]
                nll_top_k = np.argsort(np.array(fs_results_validation['clean_nll_list']))[:100]
                filter_set_validation = []
                for i in nll_top_k:
                    # print(fs_results_validation['clean_rank_list'][i])
                    if i in filter_cand_validation:
                        filter_set_validation.append(i)
                if len(filter_set_validation)<50:
                    for i in nll_top_k[:50]:
                        if i not in filter_set_validation:
                            filter_set_validation.append(i)

                print("filter_set_validation size:", len(filter_set_validation))
            else:
                filter_set_validation = np.where(np.array(fs_results_validation['clean_rank_list']) == 0)[0]
                print("filter_set_validation size:", len(filter_set_validation))

            with open(fs_samples_file_name, 'w') as results_file:
                samples = []
                for j in tqdm(range(len(dataset['valid'])), total=len(dataset['valid'])):
                    if j not in filter_set_validation:
                        continue
                    word_pairs_test = dataset['valid'][j]
                    samples.append({"input": word_pairs_test["input"], "output": str(word_pairs_test["output"])})
                json.dump(samples, results_file, indent=2)

        
        set_seed(seed)
        if "layer" in args.act_type:
            mean_activations = get_mean_layer_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer, n_icl_examples=0,
                                                    prefixes=prefixes, separators=separators, filter_set=filter_set_validation)
        else:
            if "_icl" in args.dataset_name or 'zero' in args.act_type:
                mean_activations = get_mean_head_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer, n_icl_examples=0,
                                                        prefixes=prefixes, separators=separators, filter_set=filter_set_validation)
            else:
                mean_activations = get_mean_head_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer, n_icl_examples=n_shots,
                                                        prefixes=prefixes, separators=separators, filter_set=filter_set_validation)
        if not args.no_save:
            torch.save(mean_activations, default_activations_path)

    if args.no_fv:
        exit()

    # Load or Re-Compute indirect_effect values
    if indirect_effect_path is not None and os.path.exists(indirect_effect_path):
        print("Load Indirect Effects from", indirect_effect_path)
        indirect_effect = torch.load(indirect_effect_path)
    elif indirect_effect_path is None and os.path.exists(f'{ie_path_root}/{exp_name}_indirect_effect.pt'):
        indirect_effect_path = f'{ie_path_root}/{exp_name}_indirect_effect.pt'
        print("Load Indirect Effects from", indirect_effect_path)
        indirect_effect = torch.load(indirect_effect_path) 
    elif not universal_set:     # Only compute indirect effects if we need to
        print("Computing Indirect Effects")
        set_seed(seed)

        filter_set_validation=None
        
        indirect_effect = compute_indirect_effect(dataset, mean_activations, model=model, model_config=model_config, tokenizer=tokenizer, n_shots=0,
                                                  n_trials=n_trials, last_token_only=True, prefixes=prefixes, separators=separators, filter_set=filter_set_validation, shuffle=False, itype=args.effect_type)
        
        args.indirect_effect_path = f'{ie_path_root}/{exp_name}_indirect_effect.pt'
        torch.save(indirect_effect, args.indirect_effect_path)


    print("Compute Function Vector")
    # Compute Function Vector
    if universal_set:
        fv, top_heads = compute_universal_function_vector(mean_activations, model, model_config=model_config, n_top_heads=n_top_heads)   
    else:
        fv, top_heads = compute_function_vector(mean_activations, indirect_effect, model, model_config=model_config, n_top_heads=n_top_heads, gen=args.gen)   
    
    default_fv_path = f'{save_path_root}/{exp_name}_function_vector.pt'
    print("Save function vector at", default_fv_path)
    if not args.no_save:
        torch.save(fv, default_fv_path)
    
    if args.no_eval:
        exit()

    if isinstance(eval_edit_layer, int):
        print(f"Running ZS Eval with edit_layer={eval_edit_layer}")
        set_seed(seed)
        if generate_str:
            pred_filepath = f"{save_path_root}/preds/{model_config['name_or_path'].replace('/', '_')}_ZS_intervention_layer{eval_edit_layer}.txt"
            zs_results = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=eval_edit_layer, n_shots=0,
                                     model=model, model_config=model_config, tokenizer=tokenizer, 
                                     generate_str=generate_str, metric=metric, pred_filepath=pred_filepath, prefixes=prefixes, separators=separators)
        else:
            zs_results = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=eval_edit_layer, n_shots=0,
                                    model=model, model_config=model_config, tokenizer=tokenizer, prefixes=prefixes, separators=separators)
        zs_results_file_suffix = f'_editlayer_{eval_edit_layer}.json'   

    else:
        print(f"Running sweep over layers {eval_edit_layer}")
        zs_results = {}
        set_seed(seed)
        
        if generate_str:
            zs_results[-2] = n_shot_eval_no_intervention(dataset=dataset, n_shots=0, model=model, 
                                            model_config=model_config, tokenizer=tokenizer, 
                                            compute_ppl=False, test_split='test', 
                                            prefixes=prefixes, separators=separators, 
                                            generate_str=generate_str, data_name=dataset_name)
            print("performance on 0shot", sum(zs_results[-2]['score'])/len(zs_results[-2]['score']))
        else:
            zs_results[-2] = n_shot_eval_no_intervention(dataset=dataset, n_shots=0, model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=True, test_split='test', prefixes=prefixes, separators=separators)
            print("performance on 0shot", zs_results[-2]['clean_topk'], zs_results[-2]['clean_nll'])

        for edit_layer in [0,3,6,8,10,12,15,18,21,24]:
            set_seed(seed)
            if generate_str:
                zs_results[edit_layer] = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=edit_layer, n_shots=0, 
                                                    model=model, model_config=model_config, tokenizer=tokenizer, 
                                                    generate_str=generate_str, metric=metric, prefixes=prefixes, separators=separators, data_name=dataset_name)
            else:
                zs_results[edit_layer] = n_shot_eval(dataset=dataset, fv_vector=fv, edit_layer=edit_layer, n_shots=0, prefixes=prefixes, separators=separators,
                                                    model=model, model_config=model_config, tokenizer=tokenizer,)
                print(edit_layer, zs_results[edit_layer]['intervention_topk'], zs_results[edit_layer]['intervention_nll'])
      
        zs_results_file_suffix = f'_sweep_{args.exp_name}.json'

    # Save results to files
    zs_results_file_name = make_valid_path_name(f'{save_path_root}/zs_results' + zs_results_file_suffix)
    args.zs_results_file_name = zs_results_file_name
    with open(zs_results_file_name, 'w') as results_file:
        json.dump(zs_results, results_file, indent=2)
    