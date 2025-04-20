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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'False'

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--exp_name', help='Name of model to be loaded', type=str, required=False, default='base')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='./data/eval')
    parser.add_argument('--save_path_root', help='File path to save to', type=str, required=False, default='./results')
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=43)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--adapter_name_or_path', help='Path to file containing mean_head_activations for the specified task', required=False, type=str, default=None)
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)    
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type=int, required=False, default=10)
    parser.add_argument('--n_trials', help="Number of in-context prompts to average over for indirect_effect", type=int, required=False, default=20)
    parser.add_argument("--prefixes_type", help="Metric to use when evaluating generated strings", type=str, required=False, default="Q")
    parser.add_argument("--separators_type", help="Metric to use when evaluating generated strings", type=str, required=False, default="Q")
    parser.add_argument('--generate_str', help='Whether to generate long-form completions for the task', action='store_true', required=False)
    parser.add_argument("--universal_set", help="Flag for whether to evaluate using the univeral set of heads", action="store_true", required=False)
    parser.add_argument('--max_eval_size', help='Number of seeds', type=int, required=False, default=-1)
    parser.add_argument('--adapter_type', help='Number of seeds', type=str, required=False, default="lora")
    
    args = parser.parse_args()  

    dataset_name = args.dataset_name
    model_name = args.model_name
    exp_name = args.exp_name
    root_data_dir = args.root_data_dir
    save_path_root = f"{args.save_path_root}"
    seed = args.seed
    device = args.device

    test_split = float(args.test_split)
    test_split = 1
    n_shots = args.n_shots
    n_trials = args.n_trials

    # prefixes = args.prefixes 
    
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

    
    # separators = args.separators
    generate_str = args.generate_str


    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, args, device=device)
    model.eval()

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    dataset_name_list = args.dataset_name.split(",")
    score = [0,0,0]
    zs_results = {}
    # Load the dataset
    for dataset_name in dataset_name_list:
        print("Loading Dataset", dataset_name)
        set_seed(seed)
        dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)
        dataset['test'].raw_data = dataset['test'].raw_data[-args.max_eval_size:]
        
        print('test', len(dataset['test']))

        set_seed(seed)
        zs_results[dataset_name] = n_shot_eval_no_intervention(dataset=dataset, n_shots=0, model=model, 
                                                model_config=model_config, tokenizer=tokenizer, 
                                                compute_ppl=False, test_split='test', 
                                                prefixes=prefixes, separators=separators, 
                                                generate_str=generate_str, data_name=dataset_name)
        
        if generate_str:
            print("performance on 0shot", dataset_name, zs_results[dataset_name]['score'])
            score[0] += sum(zs_results[dataset_name]['score'])/len(zs_results[dataset_name]['score'])
        else:
            print("performance on 0shot",dataset_name, zs_results[dataset_name]['clean_topk'])
            score[0] += zs_results[dataset_name]['clean_topk'][0][1]                                                        
            score[1] += zs_results[dataset_name]['clean_topk'][1][1]
            score[2] += zs_results[dataset_name]['clean_topk'][2][1]
        

    print([i/len(dataset_name_list) for i in score])
    zs_results["0avg"] = [i/len(dataset_name_list) for i in score]

    zs_results_file_suffix = f'_{args.exp_name}.json'
    zs_results_file_name = make_valid_path_name(f'{save_path_root}/results' + zs_results_file_suffix)
    args.zs_results_file_name = zs_results_file_name
    with open(zs_results_file_name, 'w') as results_file:
        json.dump(zs_results, results_file, indent=2)

    