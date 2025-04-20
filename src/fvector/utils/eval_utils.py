import torch
import string
import itertools
import re
import numpy as np
from tqdm import tqdm
from .prompt_utils import *
from .model_utils import *
from .intervention_utils import *
from rouge_score import rouge_scorer


def rougeL_score(prediction, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure

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

def compute_individual_token_rank(prob_dist, target_id) -> int:
    """
    Individual computation of token ranks across a single distribution.

    Parameters:
    prob_dist: the distribution of scores for a single output
    target_id: the target id we care about

    Return:
    A single value representing the token rank for that single token
    """
    if isinstance(target_id, list):
        target_id = target_id[0]

    return torch.where(torch.argsort(prob_dist.squeeze(), descending=True) == target_id)[0].item()


def compute_best_token_rank(prob_dist, target_ids) -> int:
    """
    Computes the best rank given a list of potential targets (target_ids) for a given probability distribution (prob_dist)
    """
    related_token_ranks = [compute_individual_token_rank(prob_dist, x) for x in target_ids]
    return min(related_token_ranks)

def compute_top_k_elements(x, K=10) -> list:
    """
    Computes the top k elements of a torch tensor (x), and returns them as a list of index tuples
    """
    h_shape = x.shape
    topk_vals, topk_inds  = torch.topk(x.view(-1), k=K, largest=True)
    top_lh = list(zip(*np.unravel_index(topk_inds, h_shape), [round(x.item(),4) for x in topk_vals]))
    top_elements = top_lh[:K]
    return top_elements

def decode_to_vocab(prob_dist, tokenizer, k=5) -> list:
    """
    Decodes and returns the top K words of a probability distribution

    Parameters:
    prob_dist: torch tensor of model logits (distribution over the vocabulary)
    tokenizer: huggingface model tokenizer
    k: number of vocabulary words to include

    Returns:
    list of top K decoded vocabulary words in the probability distribution as strings, along with their probabilities (float)
    """
    get_topk = lambda  x,K=1: torch.topk(torch.softmax(x, dim=-1), dim=-1, k=K)
    if not isinstance(prob_dist, torch.Tensor):
        prob_dist = torch.Tensor(prob_dist)

    return [(tokenizer.decode(x),round(y.item(), 5)) for x,y in zip(get_topk(prob_dist,k).indices[0],get_topk(prob_dist,k).values[0])]

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

    # print("answer",answer)
    # print(len(source) ,len(target))
    # assert len(source) < len(target) < tokenizer.model_max_length
    if len(source) < len(target):
        answer_ids = target[len(source): ]
    else:
        answer_ids = tokenizer(answer, truncation=False, padding=False).input_ids

    if tokenizer.decode(answer_ids[0])=='' or tokenizer.decode(answer_ids[0])==' ' or answer_ids[0] ==29871: # avoid spacing issues
        print(query)
        print(answer)
        print(target[len(source): ])
        print("target output is blank, please check the input output and tokenizer")
        exit()

    return answer_ids

def fv_to_vocab(function_vector, model, model_config, tokenizer, n_tokens=10):
    """
    Decodes a provided function vector into the model's vocabulary embedding space.

    Parameters:
    function_vector: torch vector extracted from ICL contexts that represents a particular function
    model: huggingface model
    model_config: dict with model information - n_layers, n_heads, etc.
    tokenizer: huggingface tokenizer
    n_tokens: number of top tokens to include in the decoding

    Returns:
    decoded_tokens: list of tuples of the form [(token, probability), ...]
    """

    if 'gpt-j' in model_config['name_or_path']:
        decoder = torch.nn.Sequential(model.transformer.ln_f, model.lm_head, torch.nn.Softmax(dim=-1))
    elif 'llama' in model_config['name_or_path'].lower() or 'mistral' in model_config['name_or_path'].lower():
        decoder = torch.nn.Sequential(model.model.norm, model.lm_head, torch.nn.Softmax(dim=-1))
    else:
        raise ValueError("Model not yet supported")
    
    d_out = decoder(function_vector.reshape(1,1,model_config['resid_dim']).to(model.device))

    vals, inds = torch.topk(d_out, k=n_tokens,largest=True)
    decoded_tokens = [(tokenizer.decode(x),round(y.item(), 4)) for x,y in zip(inds.squeeze(), vals.squeeze())]
    return decoded_tokens

def compute_dataset_baseline(dataset, model, model_config, tokenizer, n_shots=10, seed=42, generate_str=False, metric=None, prefixes=None, separators=None) -> dict:
    """
    Computes the ICL performance of the model on the provided dataset for a varying number of shots.

    Parameters:
    dataset: ICL dataset
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_shots: The upper bound of ICL examples to be used when evaluating the ICL performance of the model
    seed: seed for determining dataset split
    generate_str: whether to generate a string of tokens or predict a single token
    metric: metric to use for longer generations (F1, exact match, etc.), or None for single token prediction accuracy is used

    Returns:
    results_dict: dictionary containing the ICL performance results as the number of shots in ICL prompts varies.
    """
    results_dict = {}
    for N in range(n_shots+1):
        set_seed(seed)
        results_dict[N] = n_shot_eval_no_intervention(dataset, n_shots=N, model=model, model_config=model_config, tokenizer=tokenizer,
                                                      generate_str=generate_str, metric=metric, prefixes=prefixes, separators=separators)
    return results_dict

def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    """Return true if prediction is (case insensitive) prefix of the target."""
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)

# Evaluate a sentence
def sentence_eval(sentence, target, model, tokenizer, compute_nll=True, generate_str=False, pred_file=None, metric_fn=None, verbose=False, data_name=None):
    """
    Evaluate a single sentence completion for a model, comparing to the given target.

    Parameters:
    sentence: sentence to have the model process and predict
    target: expected response of the model
    model: huggingface model
    tokenizer: huggingface tokenizer
    compute_nll: whether to compute the negative log likelihood of a teacher-forced answer prompt (used for computing PPL)
    generate_str: whether to generate a string of tokens or predict a single token
    pred_file: filepath to save intermediate generations for debugging
    metric: metric to use for longer generations (F1, exact match, etc.)

    Returns:
    model output on the provided sentence
    """
    # Clean Run, No Intervention:
    device = model.device
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    if verbose:
        print(inputs)
    original_pred_idx = len(inputs.input_ids.squeeze()) - 1

    if compute_nll:
        target_completion = "".join(sentence + target)
        # print(target_completion)
        nll_inputs = tokenizer(target_completion, return_tensors='pt').to(device)
        nll_targets = nll_inputs.input_ids.clone()
        target_len = len(nll_targets.squeeze()) - len(inputs.input_ids.squeeze()) 
        nll_targets[:,:-target_len] = -100  # This is the accepted value to skip indices when computing loss in nn.CrossEntropyLoss
        # print( inputs.input_ids.shape)
        output = model(**nll_inputs, labels=nll_targets)

        clean_nll = output.loss.item()
        clean_output = output.logits[:,original_pred_idx,:]
    elif generate_str:

        if data_name in ["mmlu_stem_mc","piqa_mc","arc_c_mc","cmmlu_stem_mc","icl_arc_c","icl_hellaswag","icl_mmlu_stem",
                    "mmlu_stem_icl","hellaswag_icl","arc_c","arc_c_icl",'coqa', 'icl_coqa',
                    "word_sort","ob_count","ob_count_icl", "icl_ob_count","boolq",
                    'ni339_test', 'ni363_test', 'ni1292_test', 'ni1310_test', 'ni1343_test', 'ni195_test', 
                    'ni339_test_icl', 'ni363_test_icl', 'ni1292_test_icl', 'ni1310_test_icl', 'ni1343_test_icl', 'ni195_test_icl',
                    'ni472_test', 'ni061_test', 'ni611_test', 'ni214_test', 'ni141_test', 'ni198_test', 
                    'ni220_test', 'ni231_test', 'ni163_test', 'ni141_test', 'ni273_test', 'ni224_test',
                    'ni472_test_icl', 'ni061_test_icl', 'ni611_test_icl', 'ni214_test_icl', 'ni141_test_icl', 'ni198_test_icl', 
                    'ni220_test_icl', 'ni231_test_icl', 'ni163_test_icl', 'ni141_test_icl', 'ni273_test_icl', 'ni224_test_icl',
                    'lambada', 'icl_lambada', 'summedits', 'wic', 'icl_wic', 'obqa', 'icl_obqa', 'scienceQA', 'cstance', 'fomc', 'numglue']:
            MAX_NEW_TOKENS = 6
        elif data_name in ['ni1510_test_icl', 'ni1510_test_icl', 'ni292_test_icl', 'ni292_test', 'nq', 'icl_nq']:
            MAX_NEW_TOKENS = 12
        elif data_name in ["hellaswag", 'mmlu_stem', 'ni002_test_icl', 'ni002_test']:
            MAX_NEW_TOKENS = 20
        elif data_name in [ 'ni618_test', 'ni618_test_icl', 'ni589_test', 'ni589_test_icl', 
                            'ni1355_test', 'ni024_test', 'ni488_test', 'ni1355_test_icl', 'ni024_test_icl', 'ni488_test_icl', 'py150']:
            MAX_NEW_TOKENS = 30
        elif 'gsm8k' in data_name or data_name in ['ni130_test_icl', 'ni130_test']:
            MAX_NEW_TOKENS = 512
        else:
            MAX_NEW_TOKENS = 64

            # "num_beams": 1,
            # "temperature": 1.0,
            # "repetition_penalty": 1.0,
            # "eos_token_id": 2,
            # "pad_token_id": 1,
        # top_p=0.9, temperature=0.1,
        output = model.generate(inputs.input_ids, top_p=0.9, temperature=0.1,
                                max_new_tokens=MAX_NEW_TOKENS,
                                pad_token_id=tokenizer.eos_token_id,)
        # print(len(output.squeeze())-MAX_NEW_TOKENS, inputs.input_ids.shape[1])
        output_str = tokenizer.decode(output.squeeze()[inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # print(tokenizer.decode(output.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True))
        # print(output_str)
        parsed_str, score = parse_generation(output_str, target, metric_fn, data_name)
        # print(parsed_str, target, score)
        if pred_file:
            pred_file.write(f"{parsed_str.strip()}\n")
    else:
        clean_output = model(**inputs).logits[:,-1,:]
        # next_token = torch.argmax(clean_output, dim=-1)

        # # 将这个token的索引转换回字符
        # next_token_text = tokenizer.decode(next_token.item())
        # print(next_token_text)
    

    if compute_nll:
        return clean_output, clean_nll
    elif generate_str:
        return score
    else:
        return clean_output 


def n_shot_eval(dataset, fv_vector, edit_layer: int, n_shots: int, model, model_config, tokenizer, shuffle_labels:bool=False,
                filter_set=None, prefixes=None, separators=None, generate_str=False, pred_filepath=None, data_name=None,
                metric="rougeL"):
    """
    Evaluate a model and FV intervention on the model using the provided ICL dataset.

    Parameters:
    dataset: ICL dataset
    function_vector: torch vector that triggers execution of a task when added to a particular layer
    edit_layer: layer index 
    n_shots: the number of ICL examples in each in-context prompt
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    shuffle_labels: Whether to shuffle the ICL labels or not
    filter_set: whether to only include samples the model gets correct via ICL
    prefixes: dict of ICL template prefixes for each ICL component (input, output, instructions)
    separators: dict of ICL template separators for each ICL component (input, output, instructions)
    generate_str: whether to generate a string of tokens or predict a single token
    pred_filepath: filepath to save intermediate generations for debugging
    metric: metric to use for longer generations (F1, exact match, etc.)

    Returns:
    results: dict of topk accuracy on the test dataset, for both the model's n-shot, and n-shot + FV intervention, as well as the token rank of each prediction
    """
    compute_clean = (edit_layer == 0)
    clean_rank_list = []
    intervention_rank_list = []
    clean_nll_list = []
    intervention_nll_list = []

    if generate_str:
        clean_score_list = []
        intervention_score_list = []

    is_llama = 'llama' in model_config['name_or_path'].lower() or 'mistral' in model_config['name_or_path'].lower() 
    if filter_set is None:
        filter_set = np.arange(len(dataset['test']))

    if pred_filepath:
        pred_file = open(pred_filepath, 'w')
    else:
        pred_file = None        

    print(len(filter_set))
    for j in tqdm(range(len(dataset['test'])), total=len(dataset['test'])):
        if j not in filter_set:
            continue
    # for j in tqdm(filter_set):
        if n_shots == 0:
            word_pairs = {'input':[], 'output':[]}
        else:
            word_pairs = dataset['train'][np.random.choice(len(dataset['train']),n_shots, replace=False)]
        word_pairs_test = dataset['test'][j]

        prepend_bos = not is_llama
        if prefixes is not None and separators is not None:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair = word_pairs_test, prepend_bos_token=prepend_bos, 
                                                    shuffle_labels=shuffle_labels, prefixes=prefixes, separators=separators)
        else:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair = word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=shuffle_labels)
            
        # Get relevant parts of the Prompt
        query, target = prompt_data['query_target']['input'], prompt_data['query_target']['output']
        query = query[0] if isinstance(query, list) else query
        if generate_str:
            target = [target] if not isinstance(target, list) else target
        else:
            target = target[0] if isinstance(target, list) else target

        sentence = [create_prompt(prompt_data)]

        # Figure out tokens of interest
        target_token_id = 0
        if not generate_str:
            target_token_id = get_answer_id(sentence[0], target, tokenizer)


        if generate_str:
            if metric == "f1_score":
                metric_fn = f1_score
            elif metric == "exact_match_score":
                metric_fn = exact_match_score
            elif metric == "first_word_score":
                metric_fn = first_word_score
            elif metric == "rougeL":
                metric_fn = rougeL_score
            else:
                raise ValueError(f"Unknown metric: {metric}. Recognized metrics: [\"f1_score\", \"exact_match_score\"]")
            clean_output, intervention_output = function_vector_intervention(sentence, target = target, edit_layer = edit_layer,
                                                                            function_vector = fv_vector,
                                                                            model=model, model_config=model_config, tokenizer=tokenizer, 
                                                                            compute_nll=False, generate_str=generate_str, compute_clean=compute_clean, dataname=data_name) 
            # clean_parsed_str, clean_score = parse_generation(clean_output, target, metric_fn, data_name)
            intervention_parsed_str, intervention_score = parse_generation(intervention_output, target, metric_fn, data_name)
            
            # clean_score_list.append(clean_score)
            intervention_score_list.append(intervention_score)

            # if pred_file:
            #     pred_file.write(f"{clean_parsed_str.strip()}\t|||\t{intervention_parsed_str}\n")

        else:
            clean_output, intervention_output,clean_nll, intervention_nll = function_vector_intervention(sentence, target = [target], edit_layer = edit_layer, 
                                                                              function_vector = fv_vector,
                                                                              model=model, model_config=model_config, tokenizer=tokenizer, 
                                                                              compute_nll=True, compute_clean=compute_clean) 
        
            if clean_output is not None:
                clean_rank = compute_individual_token_rank(clean_output, target_token_id)
                clean_rank_list.append(clean_rank)
                clean_nll_list.append(clean_nll)

            intervention_rank = compute_individual_token_rank(intervention_output, target_token_id)        
            intervention_rank_list.append(intervention_rank)
            intervention_nll_list.append(intervention_nll)

    if generate_str:
        results = {"clean_score": clean_score_list,
                   "intervention_score": intervention_score_list}
        print(np.mean(intervention_score_list))
    else:      
        results = {"clean_topk": [(K, compute_top_k_accuracy(clean_rank_list, K)) for K in range(1,4)],
                   "clean_nll": np.mean(clean_nll_list),
                   "clean_rank_list": clean_rank_list,
                   
                   "intervention_topk": [(K, compute_top_k_accuracy(intervention_rank_list, K)) for K in range(1,4)],
                   "intervention_nll": np.mean(intervention_nll_list),
                   "intervention_rank_list":intervention_rank_list}
    
    if pred_filepath:
        pred_file.close()
    
    return results


# Evaluate few-shot dataset w/o intervention
def n_shot_eval_no_intervention(dataset, n_shots, model, model_config, tokenizer, compute_ppl=True, generate_str=False,
                                filter_set=None, shuffle_labels=False, prefixes=None, separators=None, pred_filepath=None,
                                metric="rougeL", test_split='test', data_name=None):
    """
    Evaluate a model (without any interventions) on the provided ICL dataset.

    Parameters:
    dataset: ICL dataset
    n_shots: the number of ICL examples in each in-context prompt
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    compute_ppl: whether to compute perplexity of teacher-forced correct completion for base model & intervened model
    generate_str: whether to generate a string of tokens or predict a single token
    shuffle_labels: Whether to shuffle the ICL labels or not
    prefixes: dict of ICL template prefixes for each ICL component (input, output, instructions)
    separators: dict of ICL template separators for each ICL component (input, output, instructions)
    pred_filepath: filepath to save intermediate generations for debugging
    metric: metric to use for longer generations (F1, exact match, etc.)
    test_split: the dataset test split to use as the "test" dataset, typically set to 'test' or 'valid'

    Returns:
    results: dict of topk (k=1,2,3) accuracy on the test_split dataset, for both the model's n-shot
    """
    clean_rank_list = []

    if compute_ppl:
        clean_nll_list = []

    if generate_str:
        score_list = []

    if filter_set is None:
        filter_set = np.arange(len(dataset[test_split]))
    is_llama = 'llama' in model_config['name_or_path'].lower() or 'mistral' in model_config['name_or_path'].lower() 
    prepend_bos = not is_llama

    if pred_filepath:
        pred_file = open(pred_filepath, 'w')
    else:
        pred_file = None

    print(len(filter_set))
    for j in tqdm(range(len(dataset[test_split])), total=len(dataset[test_split])):
        if j not in filter_set:
            continue
        if n_shots == 0:
            word_pairs = {'input':[], 'output':[]}
        else:
            word_pairs = dataset['train'][np.random.choice(len(dataset['train']),n_shots, replace=False)]

        word_pairs_test = dataset[test_split][j]
        if prefixes is not None and separators is not None:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair = word_pairs_test, prepend_bos_token=prepend_bos, 
                                                    shuffle_labels=shuffle_labels, prefixes=prefixes, separators=separators)
        else:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair = word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=shuffle_labels)
        
        # Get relevant parts of the Prompt
        query, target = prompt_data['query_target']['input'], prompt_data['query_target']['output']
        query = query[0] if isinstance(query, list) else query
        if generate_str:
            target = [target] if not isinstance(target, list) else target
        else:
            target = target[0] if isinstance(target, list) else target
        
        sentence = [create_prompt(prompt_data)]

        # Figure out tokens of interest
        target_token_id = 0 
        if not generate_str:
            target_token_id = get_answer_id(sentence[0], target, tokenizer)
        
        if j == 0:
            print(sentence[0])
            print(target, target_token_id)
        
        if compute_ppl:
            # print(1)
            clean_output, clean_nll = sentence_eval(sentence, target = [target],
                                                    model=model, tokenizer=tokenizer, 
                                                    compute_nll=compute_ppl,data_name=data_name)
            clean_nll_list.append(clean_nll)
            if j == 0 :

                next_token = torch.argmax(clean_output, dim=-1)

                # 将这个token的索引转换回字符
                print(next_token)
                next_token_text = tokenizer.decode(next_token.item())
                print(next_token_text)
        elif generate_str:
            # print(2)
            # print(metric)
            if metric == "f1_score":
                metric_fn = f1_score
            elif metric == "exact_match_score":
                metric_fn = exact_match_score
            elif metric == "first_word_score":
                metric_fn = first_word_score
            elif metric == "rougeL":
                metric_fn = rougeL_score
            else:
                raise ValueError(f"Unknown metric: {metric}. Recognized metrics: [\"f1_score\", \"exact_match_score\"]")
            score = sentence_eval(sentence, target=target, model=model,
                                  tokenizer=tokenizer, compute_nll=False,
                                  generate_str=True, pred_file=pred_file,
                                  metric_fn=metric_fn, verbose=(j == 0), data_name=data_name)
            score_list.append(score)
        else:
            # print(3)
            clean_output = sentence_eval(sentence, target = [target],
                                         model=model, tokenizer=tokenizer, compute_nll=False, data_name=data_name)

        if not generate_str:
            clean_rank = compute_individual_token_rank(clean_output, target_token_id)
            # if clean_rank < 1:
            #     next_token = torch.argmax(clean_output, dim=-1)
            #     # 将这个token的索引转换回字符
            #     next_token_text = tokenizer.decode(next_token.item())
            #     print(next_token, next_token_text)
            # print(torch.argsort(clean_output.squeeze(), descending=True)[:3])
            clean_rank_list.append(clean_rank)


    if generate_str:
        results = {"score": score_list}
    else:
        results = {"clean_topk": [(K, compute_top_k_accuracy(clean_rank_list, K)) for K in range(1,4)],
                   "clean_rank_list": clean_rank_list}
                   
    if compute_ppl:
        # results['clean_ppl'] = np.exp(clean_nll_list).mean()
        results['clean_nll_list'] = clean_nll_list
        results['clean_nll'] = np.mean(clean_nll_list)

    if pred_filepath:
        pred_file.close()
    
    return results


# Logic from huggingface `evaluate` library
def normalize_answer(s):
    """Lowercase text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Harmonic mean of pred overlap with gold and gold overlap with pred."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """Only correct if the prediction matches the entire answer."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def first_word_score(prediction, ground_truth):
    """Only correct if the predicted first word matches the answer's first word."""
    prediction = normalize_answer(prediction).split()
    ground_truth = normalize_answer(ground_truth).split()
    if len(prediction) > 0 and len(ground_truth) > 0:
        return prediction[0] == ground_truth[0]
    else:
        return len(prediction) == len(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Pick maximum score across possible answers."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def first_option_postprocess(text: str, options: str) -> str:
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        f'答案是?\s?([{options}])',
        f'答案是?\s?：([{options}])',
        f'答案是?\s?:([{options}])',
        f'答案应该?是\s?([{options}])',
        f'答案应该?选\s?([{options}])',
        f'答案为\s?([{options}])',
        f'答案选\s?([{options}])',
        f'选择?\s?([{options}])',
        f'只有选?项?\s?([{options}])\s?是?对',
        f'只有选?项?\s?([{options}])\s?是?错',
        f'只有选?项?\s?([{options}])\s?不?正确',
        f'只有选?项?\s?([{options}])\s?错误',
        f'说法不?对选?项?的?是\s?([{options}])',
        f'说法不?正确选?项?的?是\s?([{options}])',
        f'说法错误选?项?的?是\s?([{options}])',
        f'([{options}])\s?是正确的',
        f'([{options}])\s?是正确答案',
        f'选项\s?([{options}])\s?正确',
        f'所以答\s?([{options}])',
        f'1.\s?([{options}])[.。$]?$',
        f'所以\s?([{options}][.。$]?$)',
        f'所有\s?([{options}][.。$]?$)',
        f'[\s，：:,]([{options}])[。，,\.]?$',
        f'[\s，,：:][故即]([{options}])[。\.]?$',
        f'[\s，,：:]因此([{options}])[。\.]?$',
        f'[是为。]\s?([{options}])[。\.]?$',
        f'因此\s?([{options}])[。\.]?$',
        f'显然\s?([{options}])[。\.]?$',
        f'1.\s?(.*?)$',
        f'答案是\s?(\S+)(?:。|$)',
        f'答案应该是\s?(\S+)(?:。|$)',
        f'答案为\s?(\S+)(?:。|$)',
        f'(\s|^)[{options}][\s。，,：:\.$]',
        f'[Tt]he answer is ([{options}])',
        f'[Tt]he answer is option ([{options}])',
        f'[Tt]he correct answer is ([{options}])',
        f'[Tt]he correct answer is option ([{options}])',
        f'[Tt]he answer to the question is ([{options}])',
        f'Answer: ([{options}])',
        f'([{options}]):',
        f'(^|\s)[{options}](\s|$)',
        f'[{options}]',
    ]
    # flake8: noqa
    # yapf: enable

    # regexes = [re.compile(pattern) for pattern in patterns]
    # for regex in regexes:
    for pattern in patterns:
        regex = re.compile(pattern)
        match = regex.search(text)
        if match:
            outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ''


def parse_generation(output_str, target, metric_fn, task=None):
    """Parse a generated string for the target, and score using the specified metric"""
    # ans_regex = re.compile("([\w. ]+)[\nQ]*")
    # parsed_str = ans_regex.findall(output_str)
    if task in ["mmlu_stem_mc","piqa_mc","arc_c_mc","cmmlu_stem_mc","icl_arc_c","icl_hellaswag","icl_mmlu_stem",
                "mmlu_stem","mmlu_stem_icl","hellaswag","hellaswag_icl","arc_c","arc_c_icl", 
                'obqa', 'icl_obqa', 'scienceQA', 'cstance', 'fomc']:
        output_str = first_option_postprocess(output_str, "ABCD")
    elif task in ['coqa', 'icl_coqa',]:
        output_str = first_option_postprocess(output_str, "ABCDE")
    elif task in ["word_sort","ob_count","ob_count_icl", "icl_ob_count"]:
        output_str = bbh_freeform_postprocess(output_str)
    elif task in ["boolq", "summedits", "wic", "icl_wic"]:
        output_str = self_first_option_postprocess(output_str, ['yes','no'])
    elif task in ["gsm8k","gsm8k_icl", "icl_gsm8k", 'numgluecm']:
        output_str = gsm8k_postprocess(output_str)
    elif task in ["mathgsm8k"]:
        output_str = mathgsm8k_postprocess(output_str)
    elif task in ["alpaca", "icl_alpaca"]:
        output_str = alpaca_postprocess(output_str)
    elif task in ["lambada", "icl_lambada"]:
        output_str = lambada_postprocess(output_str)
    elif task in ["nq", "icl_nq"]:
        output_str = nq_postprocess(output_str)
    elif task in ["meetingbank", 'py150']:
        output_str = n_postprocess(output_str)
    elif 'ni' in task:
        output_str = ni_postprocess(output_str)

    # print(output_str)
    if len(output_str) > 0:
        score = metric_max_over_ground_truths(metric_fn, output_str, target)
    else:
        score = 0.0
    
    return output_str, score

def make_valid_path_name(path: str):
    """
    Returns an updated path name if given name already exists
    """
    file_name, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = file_name + "_(" + str(counter) + ")" + extension
        counter += 1

    return path

def portability_eval(dataset, fv_vector, edit_layer:int, model, model_config, tokenizer, n_eval_templates:int=20, seed:int=42):
    """
    Evaluates the portability of a function vector when used in prompts with other template forms (different from Q:{}\nA:{}\n\n).

    Parameters:
    dataset: ICL dataset
    fv_vector: torch vector extracted from an LM that triggers a task to be executed by the model
    edit_layer: layer at which to add the function vector
    model: huggingface model
    model_config: dict containing model config parameters (n_layers, n_heads, model name, etc.)
    tokenizer: huggingface tokenizer
    n_eval_templates: number of different templates to use for evaluation
    seed: seed for dataset splitting

    Returns:
    fs_res_dict: dict containing results of few-shot performance on different prompt templates
    zs_res_dict: dict containing results on zero-shot prompt templates
    fs_shuffled_res_dict: dict containing results on few-shot shuffled prompt templates
    templates: list of templates used for evaluation, 
    """
    # Pre-define portability template parts
    all_prefixes = [{'input': 'A:', 'output': 'B:', 'instructions': ''},
                    {'input': 'input:', 'output': 'output:', 'instructions': ''},
                    {'input': 'Input:', 'output': 'Output:', 'instructions': ''},
                    {'input': 'In:', 'output': 'Out:', 'instructions': ''},
                    {'input': 'question:', 'output': 'answer:', 'instructions': ''},
                    {'input': 'Question:', 'output': 'Answer:', 'instructions': ''},
                    {'input': '', 'output': ' ->', 'instructions': ''},
                    {'input': '', 'output': ' :', 'instructions': ''},
                    {'input': 'text:', 'output': 'label:', 'instructions': ''},
                    {'input': 'x:', 'output': 'f(x):', 'instructions': ''},
                    {'input': 'x:', 'output': 'y:', 'instructions': ''},
                    {'input': 'X:', 'output': 'Y:', 'instructions': ''}]

    all_separators=[{'input': ' ', 'output': '', 'instructions': ''},
                    {'input': ' ', 'output': '\n', 'instructions': ''},
                    {'input': ' ', 'output': '\n\n', 'instructions': ''},
                    {'input': '\n', 'output': '\n', 'instructions': ''},
                    {'input': '\n', 'output': '\n\n', 'instructions': ''},
                    {'input': '\n\n', 'output': '\n\n', 'instructions': ''},
                    {'input': ' ', 'output': '|', 'instructions': ''},
                    {'input': '\n', 'output': '|', 'instructions': ''},
                    {'input': '|', 'output': '\n', 'instructions': ''},
                    {'input': '|', 'output': '\n\n', 'instructions': ''}]

    # Choose a random subset of n_eval_templates combinations
    all_combinations = list(itertools.product(all_prefixes, all_separators))
    set_seed(seed)
    random_combos = [list(x) for x in np.array(all_combinations)[np.random.choice(np.arange(len(all_combinations)), n_eval_templates, replace=False)]]

    zs_res_dict = {}
    fs_res_dict = {}
    fs_shuffled_res_dict = {}
    templates = []
    for i,(p,s) in enumerate(random_combos):

        template_repr = p['input'] + '{}' + s['input'] + p['output'] + '{}' + s['output']
        templates.append(template_repr)

        set_seed(seed)
        # FS Eval + Filtering
        fs_results = n_shot_eval_no_intervention(dataset=dataset, n_shots=10, model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=False, prefixes=p, separators=s)
        filter_set = np.where(np.array(fs_results['clean_rank_list']) == 0)[0]
        fs_res_dict[i] = fs_results

        # ZS Eval
        zs_res_dict[i] = n_shot_eval(dataset, fv_vector, edit_layer, 0, model, model_config, tokenizer, filter_set=filter_set, prefixes=p, separators=s)

        # ZS Eval
        fs_shuffled_res_dict[i] = n_shot_eval(dataset, fv_vector, edit_layer, 10, model, model_config, tokenizer, filter_set=filter_set, prefixes=p, separators=s, shuffle_labels=True)
    
    return fs_res_dict, zs_res_dict,fs_shuffled_res_dict,  templates



def general_postprocess(text: str) -> str:
    # Cut off the first newline, period, or comma
    truncated_text = re.split(r'[\n,]', text, 1)[0]

    # Remove punctuation
    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    # Remove article
    no_articles = re.sub(r'\b(a|an|the)\b',
                         '',
                         no_punctuation,
                         flags=re.IGNORECASE)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()

    return cleaned_text

def self_first_option_postprocess(text: str, options: list) -> str:
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    options = [o.lower() for o in options]
    options_str = "|".join(options)
    patterns = [
        f'答案是?\s?({options_str})',
        f'答案是?\s?：({options_str})',
        f'答案是?\s?:({options_str})',
        f'答案应该?是\s?({options_str})',
        f'答案应该?选\s?({options_str})',
        f'答案为\s?({options_str})',
        f'答案选\s?({options_str})',
        f'选择?\s?({options_str})',
        f'只有选?项?\s?({options_str})\s?是?对',
        f'只有选?项?\s?({options_str})\s?是?错',
        f'只有选?项?\s?({options_str})\s?不?正确',
        f'只有选?项?\s?({options_str})\s?错误',
        f'说法不?对选?项?的?是\s?({options_str})',
        f'说法不?正确选?项?的?是\s?({options_str})',
        f'说法错误选?项?的?是\s?({options_str})',
        f'({options_str})\s?是正确的',
        f'({options_str})\s?是正确答案',
        f'选项\s?({options_str})\s?正确',
        f'所以答\s?({options_str})',
        f'1.\s?({options_str})[.。$]?$',
        f'所以\s?({options_str}[.。$]?$)',
        f'所有\s?({options_str}[.。$]?$)',
        f'[\s，：:,]({options_str})[。，,\.]?$',
        f'[\s，,：:][故即]({options_str})[。\.]?$',
        f'[\s，,：:]因此({options_str})[。\.]?$',
        f'[是为。]\s?({options_str})[。\.]?$',
        f'因此\s?({options_str})[。\.]?$',
        f'显然\s?({options_str})[。\.]?$',
        f'1.\s?(.*?)$',
        f'答案是\s?(\S+)(?:。|$)',
        f'答案应该是\s?(\S+)(?:。|$)',
        f'答案为\s?(\S+)(?:。|$)',
        f'(\s|^){options_str}[\s。，,：:\.$]',
        f'[Tt]he answer is ({options_str})',
        f'[Tt]he answer is option ({options_str})',
        f'[Tt]he correct answer is ({options_str})',
        f'[Tt]he correct answer is option ({options_str})',
        f'[Tt]he answer to the question is ({options_str})',
        f'[Aa]nswer: ({options_str})',
        f'({options_str}):',
        f'(^|\s){options_str}(\s|$)',
        f'({options_str})',
    ]
    # flake8: noqa
    # yapf: enable

    # regexes = [re.compile(pattern) for pattern in patterns]
    # for regex in regexes:
    text = text.lower()
    for pattern in patterns:
        regex = re.compile(pattern)
        match = regex.search(text)
        if match:
            outputs = match.group(0)
            for idx,i in enumerate(options):
                if i in outputs:
                    return i
    return ''

def first_option_postprocess(text: str, options: str) -> str:
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        f'答案是?\s?([{options}])',
        f'答案是?\s?：([{options}])',
        f'答案是?\s?:([{options}])',
        f'答案应该?是\s?([{options}])',
        f'答案应该?选\s?([{options}])',
        f'答案为\s?([{options}])',
        f'答案选\s?([{options}])',
        f'选择?\s?([{options}])',
        f'只有选?项?\s?([{options}])\s?是?对',
        f'只有选?项?\s?([{options}])\s?是?错',
        f'只有选?项?\s?([{options}])\s?不?正确',
        f'只有选?项?\s?([{options}])\s?错误',
        f'说法不?对选?项?的?是\s?([{options}])',
        f'说法不?正确选?项?的?是\s?([{options}])',
        f'说法错误选?项?的?是\s?([{options}])',
        f'([{options}])\s?是正确的',
        f'([{options}])\s?是正确答案',
        f'选项\s?([{options}])\s?正确',
        f'所以答\s?([{options}])',
        f'1.\s?([{options}])[.。$]?$',
        f'所以\s?([{options}][.。$]?$)',
        f'所有\s?([{options}][.。$]?$)',
        f'[\s，：:,]([{options}])[。，,\.]?$',
        f'[\s，,：:][故即]([{options}])[。\.]?$',
        f'[\s，,：:]因此([{options}])[。\.]?$',
        f'[是为。]\s?([{options}])[。\.]?$',
        f'因此\s?([{options}])[。\.]?$',
        f'显然\s?([{options}])[。\.]?$',
        f'1.\s?(.*?)$',
        f'答案是\s?(\S+)(?:。|$)',
        f'答案应该是\s?(\S+)(?:。|$)',
        f'答案为\s?(\S+)(?:。|$)',
        f'(\s|^)[{options}][\s。，,：:\.$]',
        f'[Tt]he answer is ([{options}])',
        f'[Tt]he answer is option ([{options}])',
        f'[Tt]he correct answer is ([{options}])',
        f'[Tt]he correct answer is option ([{options}])',
        f'[Tt]he answer to the question is ([{options}])',
        f'Answer: ([{options}])',
        f'([{options}]):',
        f'(^|\s)[{options}](\s|$)',
        f'[{options}]',
    ]
    # flake8: noqa
    # yapf: enable

    # regexes = [re.compile(pattern) for pattern in patterns]
    # for regex in regexes:

    for pattern in patterns:
        regex = re.compile(pattern)
        match = regex.search(text)
        if match:
            outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ''

def gsm8k_postprocess(text: str) -> str:
    # print(text)
    text = text.split('\n\n')[0]
    text = text.split(' ')[::-1]
    flag = False
    ret = ''
    for i in range(len(text)):
        s = text[i]
        for i in range(len(s)):
            if s[i].isdigit():
                flag = True
                ret = s
                break
        if flag:
            break
    ret1 = ''
    for i in range(len(ret)):
        # deal with potential float number
        if ret[i].isdigit() or ret[i] == '.':
            ret1 += ret[i]
    return ret1.strip('.')



def mathgsm8k_postprocess(text: str) -> str:
    # print(text)
    # text = text.split('\n')[-1]
    text = text.split(' ')[::-1]
    flag = False
    ret = ''
    for i in range(len(text)):
        s = text[i]
        for i in range(len(s)):
            if s[i].isdigit():
                flag = True
                ret = s
                break
        if flag:
            break
    ret1 = ''
    for i in range(len(ret)):
        # deal with potential float number
        if ret[i].isdigit() or ret[i] == '.':
            ret1 += ret[i]
    return ret1.strip('.')


def n_postprocess(text: str) -> str:
    text = text.split('\n')[0]
    return text
def alpaca_postprocess(text: str) -> str:
    text = text.split('\n\n\n')[0]
    return text

def lambada_postprocess(text: str) -> str:
    ans = text.split(' ')
    try:
        ans = ans[1] if len(ans[0])==0 else ans[0]
        if ans.endswith('.') or ans.endswith('?'):
            ans = ans[:-1]
        return ans
    except Exception as E:
        return ""

def nq_postprocess(text: str) -> str:
    ans = text
    ans_line = ans.split('answer is ')
    if len(ans_line) == 1:
        ans_line = ans.split('answer is: ')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    ans = ans.split('\n')[0]
    if ans.endswith('.') or ans.endswith('?'):
        ans = ans[:-1]
    return ans

def ni_postprocess(text: str) -> str:
    text = text.split('Input')[0]
    text = text.split('#')[0]
    return text

def bbh_freeform_postprocess(text: str) -> str:
    ans = text
    # print(ans)
    ans_line = ans.split('answer is ')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    ans = ans.split('\n')[0]
    if ans.endswith('.'):
        ans = ans[:-1]
    if "</s>" in ans:
        ans = ans[:-4]
    return ans

# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

