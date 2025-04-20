import os, re, json
from tqdm import tqdm
import torch, numpy as np
import argparse
from src.baukit import TraceDict

# Include prompt creation helper functions
from src.fvector.utils.prompt_utils import *
from src.fvector.utils.intervention_utils import *
from src.fvector.utils.model_utils import *
from src.fvector.utils.extract_utils import *

def get_answer_id(query, answer, tokenizer):
    """
    Parameters:
    query (str): query as a string
    answer (str): expected answer as a string
    tokenizer: huggingface tokenizer
    
    Returns: 
    answer_ids (list): A list of the contextualized tokens of the answer
    """
    source = tokenizer(query, truncation=False, padding=False).input_ids
    target = tokenizer(query + answer, truncation=False, padding=False).input_ids
    # print(query)
    # print(query + answer)
    # print(len(source) ,len(target))
    assert len(source) < len(target) < tokenizer.model_max_length
    answer_ids = target[len(source): ]

    if tokenizer.decode(answer_ids[0])=='' or tokenizer.decode(answer_ids[0])==' ' or answer_ids[0] ==29871: # avoid spacing issues
        print(query)
        print(answer)
        print(target[len(source): ])
        print("target output is blank, please check the input output and tokenizer")
        exit()

    return answer_ids



def compute_top_k_accuracy(target_token_ranks, k=10) -> float:
    """
    Evaluation to compute topk accuracy.

    Parameters:
    target_token_ranks: the distribution of output token ranks
    k: how many tokens we're looking at (top K)

    Return:
    The accuracy of the token in the top k of tokens
    """

    target_token_ranks = np.array(target_token_ranks)
    return (target_token_ranks < k).sum(axis=0) / len(target_token_ranks) 

def compute_best_token_rank(prob_dist, target_ids) -> int:
    """
    Computes the best rank given a list of potential targets (target_ids) for a given probability distribution (prob_dist)
    """
    related_token_ranks = [compute_individual_token_rank(prob_dist, x) for x in target_ids]
    return min(related_token_ranks)

def activation_replacement_per_class_intervention(prompt_data, avg_activations, dummy_labels, model, model_config, tokenizer, last_token_only=True):
    """
    Experiment to determine top intervention locations through avg activation replacement. 
    Performs a systematic sweep over attention heads (layer, head) to track their causal influence on probs of key tokens.

    Parameters: 
    prompt_data: dict containing ICL prompt examples, and template information
    avg_activations: avg activation of each attention head in the model taken across n_trials ICL prompts
    dummy_labels: labels and indices for a baseline prompt with the same number of example pairs
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    last_token_only: If True, only computes indirect effect for heads at the final token position. If False, computes indirect_effect for heads for all token classes

    Returns:   
    indirect_effect_storage: torch tensor containing the indirect_effect of each head for each token class.
    """
    device = model.device

    # Get sentence and token labels
    query, target = prompt_data['query_target']['input'], prompt_data['query_target']['output']
    query = query[0] if isinstance(query, list) else query

    target = target[0] if isinstance(target, list) else target
    
    sentence = [create_prompt(prompt_data)]

    target_token_id = get_answer_id(sentence[0], target, tokenizer)

    target = [target]

    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    original_pred_idx = len(inputs.input_ids.squeeze()) - 1

    target_completion = "".join(sentence + target)
    nll_inputs = tokenizer(target_completion, return_tensors='pt').to(device)
    nll_targets = nll_inputs.input_ids.clone()
    target_len = len(nll_targets.squeeze()) - len(inputs.input_ids.squeeze()) 
    nll_targets[:,:-target_len] = -100  # This is the accepted value to skip indices when computing loss (see nn.CrossEntropyLoss default)
    


    indirect_effect_storage = torch.zeros(model_config['n_layers'], model_config['n_heads'],2)

    # Clean Run of Baseline:
    clean_outputs = model(**nll_inputs, labels=nll_targets)
    clean_nll = clean_outputs.loss.item()
    clean_output = clean_outputs.logits[:,original_pred_idx,:]
    # print(clean_output.shape)
    clean_probs = torch.softmax(clean_output, dim=-1)

    # For every layer, head, token combination perform the replacement & track the change in meaningful tokens
    # for layer in range(model_config['n_layers']):
    for layer in range(1):
        head_hook_layer = [model_config['attn_hook_names'][l] for l in range(model_config['n_layers'])]
        
        # for head_n in range(model_config['n_heads']):
        for head_n in range(1):
           
            # intervention_locations = [(layer, head_n, token_n) for token_n in class_token_inds]
            pos = [(14, 1, 0.0391), (11, 2, 0.0225), (9, 25, 0.02), (12, 15, 0.0196), (12, 28, 0.0191), (13, 7, 0.0171), (11, 18, 0.0152), (12, 18, 0.0113), (16, 10, 0.007), (14, 16, 0.007),]
            intervention_locations = [(layer, head_n, original_pred_idx) for (layer, head_n, _) in pos[:10]]
            intervention_fn = replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
                                                        model=model, model_config=model_config,
                                                        batched_input=False)
            with TraceDict(model, layers=head_hook_layer, edit_output=intervention_fn) as td:                
                output = model(**nll_inputs, labels=nll_targets) 
                 # batch_size x n_tokens x vocab_size, only want last token prediction
            
            intervention_nll = output.loss.item()
            intervention_output = output.logits[:,original_pred_idx,:]
            # TRACK probs of tokens of interest
            # print(intervention_output.shape)
            intervention_probs = torch.softmax(intervention_output, dim=-1) # convert to probability distribution
            # print(target_token_id)
            # print(torch.LongTensor([target_token_id[0]]))
            # print((intervention_probs-clean_probs).index_select(1, torch.LongTensor(target_token_id[0]).to(device).to(device).squeeze()))
            print("clean", clean_nll)
            print("intervention", intervention_nll)
            
            intervention_rank = compute_individual_token_rank(intervention_probs, torch.LongTensor([target_token_id[0]]).to(device).squeeze())
            clean_rank = compute_individual_token_rank(clean_probs, torch.LongTensor([target_token_id[0]]).to(device).squeeze()) 
            # print(clean_rank, intervention_rank)
            # clean_rank_list.append(clean_rank)
            # intervention_rank_list.append(intervention_rank)
        # print([(K, compute_top_k_accuracy(clean_rank_list, K)) for K in range(1,4)])
                
        # print([(K, compute_top_k_accuracy(intervention_rank_list, K)) for K in range(1,4)])
    return clean_rank, intervention_rank, clean_nll, intervention_nll


def activation_replacement_each(prompt_data, avg_activations, dummy_labels, model, model_config, tokenizer, last_token_only=True):
    """
    Experiment to determine top intervention locations through avg activation replacement. 
    Performs a systematic sweep over attention heads (layer, head) to track their causal influence on probs of key tokens.

    Parameters: 
    prompt_data: dict containing ICL prompt examples, and template information
    avg_activations: avg activation of each attention head in the model taken across n_trials ICL prompts
    dummy_labels: labels and indices for a baseline prompt with the same number of example pairs
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    last_token_only: If True, only computes indirect effect for heads at the final token position. If False, computes indirect_effect for heads for all token classes

    Returns:   
    indirect_effect_storage: torch tensor containing the indirect_effect of each head for each token class.
    """
    device = model.device

    # Get sentence and token labels
    query, target = prompt_data['query_target']['input'], prompt_data['query_target']['output']
    query = query[0] if isinstance(query, list) else query

    target = target[0] if isinstance(target, list) else target
    
    sentence = [create_prompt(prompt_data)]


    target_token_id = get_answer_id(sentence[0], target, tokenizer)

    target = [target]

    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    original_pred_idx = len(inputs.input_ids.squeeze()) - 1

    target_completion = "".join(sentence + target)
    nll_inputs = tokenizer(target_completion, return_tensors='pt').to(device)
    nll_targets = nll_inputs.input_ids.clone()
    target_len = len(nll_targets.squeeze()) - len(inputs.input_ids.squeeze()) 
    nll_targets[:,:-target_len] = -100  # This is the accepted value to skip indices when computing loss (see nn.CrossEntropyLoss default)
    

    indirect_effect_storage = torch.zeros(model_config['n_layers'], model_config['n_heads'],2)

    # Clean Run of Baseline:
    clean_outputs = model(**nll_inputs, labels=nll_targets)
    clean_nll = clean_outputs.loss.item()
    clean_output = clean_outputs.logits[:,original_pred_idx,:]
    # print(clean_output.shape)
    clean_probs = torch.softmax(clean_output, dim=-1)
    # print(clean_probs.shape)

    # For every layer, head, token combination perform the replacement & track the change in meaningful tokens
    for layer in range(model_config['n_layers']):
    # for layer in range(1):
        head_hook_layer = [model_config['attn_hook_names'][l] for l in range(model_config['n_layers'])]
        
        for head_n in range(model_config['n_heads']):
        # for head_n in range(1):

            intervention_locations = [(layer, head_n, original_pred_idx)]
            intervention_fn = replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
                                                        model=model, model_config=model_config,
                                                        batched_input=False)
            with TraceDict(model, layers=head_hook_layer, edit_output=intervention_fn) as td:                
                output = model(**nll_inputs, labels=nll_targets) # batch_size x n_tokens x vocab_size, only want last token prediction
            
            intervention_nll = output.loss.item()
            intervention_output = output.logits[:,original_pred_idx,:]
            # TRACK probs of tokens of interest
            # print(intervention_output.shape)
            intervention_probs = torch.softmax(intervention_output, dim=-1) # convert to probability distribution
            # print(target_token_id)
            # print(torch.LongTensor([target_token_id[0]]))
            # print((intervention_probs-clean_probs).index_select(1, torch.LongTensor(target_token_id[0]).to(device).to(device).squeeze()))
            indirect_effect_storage[layer,head_n, 0] = (intervention_probs-clean_probs).index_select(1, torch.LongTensor([target_token_id[0]]).to(device).squeeze()).squeeze()
            indirect_effect_storage[layer,head_n, 1] = (intervention_nll-clean_nll)
              
    return indirect_effect_storage


def compute_indirect_effect(dataset, mean_activations, model, model_config, tokenizer, n_shots=10, n_trials=25, last_token_only=True, prefixes=None, separators=None, filter_set=None, shuffle=True, itype='each'):
    """
    Computes Indirect Effect of each head in the model

    Parameters:
    dataset: ICL dataset
    mean_activations:
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_shots: Number of shots in each in-context prompt
    n_trials: Number of in-context prompts to average over
    last_token_only: If True, only computes Indirect Effect for heads at the final token position. If False, computes Indirect Effect for heads for all token classes


    Returns:
    indirect_effect: torch tensor of the indirect effect for each attention head in the model, size n_trials * n_layers * n_heads
    """
    n_test_examples = 1
    split = 'test'
    # split = 'valid'
    mean_activations = mean_activations.to(model.device)

    if prefixes is not None and separators is not None:
        dummy_gt_labels = get_dummy_token_labels(n_shots, tokenizer=tokenizer, prefixes=prefixes, separators=separators)
    else:
        dummy_gt_labels = get_dummy_token_labels(n_shots, tokenizer=tokenizer)
    # print("dummy_gt_labels ",dummy_gt_labels)
    is_llama = 'llama' in model_config['name_or_path'].lower() or 'mistral' in model_config['name_or_path'].lower() 
    prepend_bos = not is_llama

    if last_token_only:
        indirect_effect = torch.zeros(n_trials,model_config['n_layers'], model_config['n_heads'],2)
    else:
        indirect_effect = torch.zeros(n_trials,model_config['n_layers'], model_config['n_heads'],10) # have 10 classes of tokens

    if filter_set is None:
        filter_set = np.arange(len(dataset[split]))
    
    clean_rank_list = []
    intervention_rank_list = []
    clean_ppl_list = []
    intervention_ppl_list = []
    for i in tqdm(range(n_trials), total=n_trials):
        if itype=="self":
            word_pairs = dataset['train'][np.random.choice(len(dataset['train']),5, replace=False)]
            word_pairs_test = dataset[split][i]
            if prefixes is not None and separators is not None:
                prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, shuffle_labels=False, 
                                                            prepend_bos_token=prepend_bos, prefixes=prefixes, separators=separators)
            else:
                prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, 
                                                           shuffle_labels=False, prepend_bos_token=prepend_bos)


            query = prompt_data['query_target']['input']
            query = query[0] if isinstance(query, list) else query
            sentence = [create_prompt(prompt_data)]

            inputs = tokenizer(sentence, return_tensors='pt').to(model.device)
            original_pred_idx = len(inputs.input_ids.squeeze()) - 1
            
            MAX_NEW_TOKENS = 5
            output = model.generate(inputs.input_ids, top_p=0.9, temperature=0.1,
                                max_new_tokens=MAX_NEW_TOKENS,
                                pad_token_id=tokenizer.eos_token_id,)
            output_str = tokenizer.decode(output.squeeze()[inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        

        word_pairs = dataset['train'][np.random.choice(len(dataset['train']),n_shots, replace=False)]
        # word_pairs_test = dataset[split][np.random.choice(filter_set,n_test_examples, replace=False)]
        word_pairs_test = dataset[split][i]
        if itype=="self":
            word_pairs_test["output"] = output_str
            print(word_pairs_test)
        if prefixes is not None and separators is not None:
            prompt_data_random = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, shuffle_labels=shuffle, 
                                                           prepend_bos_token=prepend_bos, prefixes=prefixes, separators=separators)
        else:
            prompt_data_random = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, 
                                                           shuffle_labels=shuffle, prepend_bos_token=prepend_bos)
        # print(prompt_data_random)

        if itype=="all":
            a,b,c,d = activation_replacement_per_class_intervention(prompt_data=prompt_data_random, 
                                                                    avg_activations = mean_activations, 
                                                                    dummy_labels=dummy_gt_labels, 
                                                                    model=model, model_config=model_config, tokenizer=tokenizer, 
                                                                    last_token_only=last_token_only)
            clean_rank_list.append(a)
            intervention_rank_list.append(b)            
            clean_ppl_list.append(c)
            intervention_ppl_list.append(d)
        elif itype=="each":
            ind_effects = activation_replacement_each(prompt_data=prompt_data_random, 
                                                                    avg_activations = mean_activations, 
                                                                    dummy_labels=dummy_gt_labels, 
                                                                    model=model, model_config=model_config, tokenizer=tokenizer, 
                                                                    last_token_only=last_token_only)

            indirect_effect[i] = ind_effects.squeeze()
        elif itype=="self":
            ind_effects = activation_replacement_each(prompt_data=prompt_data_random, 
                                                                    avg_activations = mean_activations, 
                                                                    dummy_labels=dummy_gt_labels, 
                                                                    model=model, model_config=model_config, tokenizer=tokenizer, 
                                                                    last_token_only=last_token_only)

            indirect_effect[i] = ind_effects.squeeze()
    if itype=="all":   
        print(np.mean(clean_ppl_list))
        print(np.mean(intervention_ppl_list))

        print([(K, compute_top_k_accuracy(clean_rank_list, K)) for K in range(1,4)])
        print([(K, compute_top_k_accuracy(intervention_rank_list, K)) for K in range(1,4)])
        exit()
    return indirect_effect


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=True)
    parser.add_argument('--model_name', help='Name of model to be loaded', type=str, required=False, default='EleutherAI/gpt-j-6b')
    parser.add_argument('--root_data_dir', help='Root directory of data files', type=str, required=False, default='../dataset_files')
    parser.add_argument('--save_path_root', help='File path to save indirect effect to', type=str, required=False, default='../results')
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type =int, required=False, default=10)
    parser.add_argument('--n_trials', help="Number of in-context prompts to average over", type=int, required=False, default=25)
    parser.add_argument('--test_split', help="Percentage corresponding to test set split size", required=False, default=0.3)
    parser.add_argument('--device', help='Device to run on',type=str, required=False, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mean_activations_path', help='Path to mean activations file used for intervention', required=False, type=str, default=None)
    parser.add_argument('--last_token_only', help='Whether to compute indirect effect for heads at only the final token position, or for all token classes', required=False, type=bool, default=True)
    parser.add_argument('--prefixes', help='Prompt template prefixes to be used', type=json.loads, required=False, default={"input":"Q:", "output":"A:", "instructions":""})
    parser.add_argument('--separators', help='Prompt template separators to be used', type=json.loads, required=False, default={"input":"\n", "output":"\n\n", "instructions":""})    
        
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name
    root_data_dir = args.root_data_dir
    save_path_root = f"{args.save_path_root}/{dataset_name}"
    seed = args.seed
    n_shots = args.n_shots
    n_trials = args.n_trials
    test_split = args.test_split
    device = args.device
    mean_activations_path = args.mean_activations_path
    last_token_only = args.last_token_only
    prefixes = args.prefixes
    separators = args.separators


    # Load Model & Tokenizer
    torch.set_grad_enabled(False)
    print("Loading Model")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device)

    set_seed(seed)

    # Load the dataset
    print("Loading Dataset")
    dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)
    

    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    # Load or Re-Compute Mean Activations
    if mean_activations_path is not None and os.path.exists(mean_activations_path):
        mean_activations = torch.load(mean_activations_path)
    elif mean_activations_path is None and os.path.exists(f'{save_path_root}/{dataset_name}_mean_head_activations.pt'):
        mean_activations_path = f'{save_path_root}/{dataset_name}_mean_head_activations.pt'
        mean_activations = torch.load(mean_activations_path)        
    else:
        print("Computing Mean Activations")
        mean_activations = get_mean_head_activations(dataset, model=model, model_config=model_config, tokenizer=tokenizer, 
                                                     n_icl_examples=n_shots, N_TRIALS=n_trials, prefixes=prefixes, separators=separators)
        torch.save(mean_activations, f'{save_path_root}/{dataset_name}_mean_head_activations.pt')

    print("Computing Indirect Effect")
    indirect_effect = compute_indirect_effect(dataset, mean_activations, model=model, model_config=model_config, tokenizer=tokenizer, 
                                              n_shots=n_shots, n_trials=n_trials, last_token_only=last_token_only, prefixes=prefixes, separators=separators)

    # Write args to file
    args.save_path_root = save_path_root
    args.mean_activations_path = mean_activations_path
    with open(f'{save_path_root}/indirect_effect_args.txt', 'w') as arg_file:
        json.dump(args.__dict__, arg_file, indent=2)

    torch.save(indirect_effect, f'{save_path_root}/{dataset_name}_indirect_effect.pt')

    