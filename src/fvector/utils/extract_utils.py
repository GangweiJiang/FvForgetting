import os, re, json

import torch, numpy as np
import pandas as pd
from src.baukit import TraceDict

# Include prompt creation helper functions
from .prompt_utils import *
from .intervention_utils import *
from .model_utils import *
from .eval_utils import *


# Attention Activations
def gather_attn_activations(prompt_data, layers, dummy_labels, model, tokenizer):
    """
    Collects activations for an ICL prompt 

    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    layers: layer names to get activatons from
    dummy_labels: labels and indices for a baseline prompt with the same number of example pairs
    model: huggingface model
    tokenizer: huggingface tokenizer

    Returns:
    td: tracedict with stored activations
    idx_map: map of token indices to respective averaged token indices
    idx_avg: dict containing token indices of multi-token words
    """   
    
    # Get sentence and token labels
    query = prompt_data['query_target']['input']
    token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query)
    sentence = [prompt_string]

    inputs = tokenizer(sentence, return_tensors='pt').to(model.device)
    # print(prompt_string, len(inputs['input_ids'][0]))
    idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy_labels)

    # Access Activations 
    with TraceDict(model, layers=layers, retain_input=True, retain_output=False) as td:                
        model(**inputs) # batch_size x n_tokens x vocab_size, only want last token prediction

    return td, idx_map, idx_avg

def get_mean_head_activations(dataset, model, model_config, tokenizer, n_icl_examples = 10, N_TRIALS = 100, shuffle_labels=False, prefixes=None, separators=None, filter_set=None):
    """
    Computes the average activations for each attention head in the model, where multi-token phrases are condensed into a single slot through averaging.

    Parameters: 
    dataset: ICL dataset
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_icl_examples: Number of shots in each in-context prompt
    N_TRIALS: Number of in-context prompts to average over
    shuffle_labels: Whether to shuffle the ICL labels or not
    prefixes: ICL template prefixes
    separators: ICL template separators
    filter_set: whether to only include samples the model gets correct via ICL

    Returns:
    mean_activations: avg activation of each attention head in the model taken across n_trials ICL prompts
    """
    def split_activations_by_head(activations, model_config):
        new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
        activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
        return activations
    
    n_test_examples = 1
    if prefixes is not None and separators is not None:
        dummy_labels = get_dummy_token_labels(n_icl_examples, tokenizer=tokenizer, prefixes=prefixes, separators=separators)
    else:
        dummy_labels = get_dummy_token_labels(n_icl_examples, tokenizer=tokenizer)
    # print(dummy_labels)
    # activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['n_heads'], len(dummy_labels), model_config['resid_dim']//model_config['n_heads'])
    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['n_heads'], 1, model_config['resid_dim']//model_config['n_heads'])

    if filter_set is None:
        filter_set = np.arange(len(dataset['valid']))

    is_llama = 'llama' in model_config['name_or_path'].lower() or 'mistral' in model_config['name_or_path'].lower()
    prepend_bos = not is_llama

    for n in range(N_TRIALS):
        word_pairs = dataset['train'][np.random.choice(len(dataset['train']), n_icl_examples, replace=False)]
        word_pairs_test = dataset['valid'][np.random.choice(filter_set, n_test_examples, replace=False)]
        if prefixes is not None and separators is not None:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, 
                                                    shuffle_labels=shuffle_labels, prefixes=prefixes, separators=separators)
        else:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=shuffle_labels)

        activations_td,idx_map,idx_avg = gather_attn_activations(prompt_data=prompt_data, 
                                                            layers = model_config['attn_hook_names'], 
                                                            dummy_labels=dummy_labels, 
                                                            model=model, 
                                                            tokenizer=tokenizer)

        stack_initial = torch.vstack([split_activations_by_head(activations_td[layer].input, model_config) for layer in model_config['attn_hook_names']]).permute(0,2,1,3)
        # stack_filtered = stack_initial[:,:,list(idx_map.keys())]
        # if len(list(idx_map.keys())) != len(dummy_labels):
        #     activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['n_heads'], len(list(idx_map.keys())), model_config['resid_dim']//model_config['n_heads'])

        # for (i,j) in idx_avg.values():
        #     # print(j, stack_initial.shape[2])
        #     stack_filtered[:,:,idx_map[i]] = stack_initial[:,:,i:j+1].mean(axis=2) # Average activations of multi-token words across all its tokens
        stack_filtered = stack_initial[:,:,-1]
        # print(stack_filtered.shape)
        activation_storage[n] = torch.unsqueeze(stack_filtered, 2)  

    mean_activations = activation_storage.mean(dim=0)
    return mean_activations

# Layer Activations
def gather_layer_activations(prompt_data, layers, model, tokenizer):
    """
    Collects activations for an ICL prompt 

    Parameters:
    prompt_data: dict containing
    layers: layer names to get activatons from
    model: huggingface model
    tokenizer: huggingface tokenizer

    Returns:
    td: tracedict with stored activations
    """   
    
    # Get sentence and token labels
    query = prompt_data['query_target']['input']
    _, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query)
    sentence = [prompt_string]
    # print()
    inputs = tokenizer(sentence, return_tensors='pt').to(model.device)
    # print(prompt_string, len(inputs['input_ids'][0]))
    # del inputs['token_type_ids']
    # Access Activations 
    with TraceDict(model, layers=layers, retain_input=False, retain_output=True) as td:           
        model(**inputs) # batch_size x n_tokens x vocab_size, only want last token prediction

    return td

def get_mean_layer_activations(dataset, model, model_config, tokenizer, n_icl_examples = 10, N_TRIALS = 100, shuffle_labels=False, prefixes=None, separators=None, filter_set=None):
    """
    Computes the average activations for each layer in the model, at the final predictive token.

    Parameters: 
    dataset: ICL dataset
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    n_icl_examples: Number of shots in each in-context prompt
    N_TRIALS: Number of in-context prompts to average over
    shuffle_labels: Whether to shuffle the ICL labels or not
    prefixes: ICL template prefixes
    separators: ICL template separators
    filter_set: whether to only include samples the model gets correct via ICL

    Returns:
    mean_activations: avg activation of each layer hidden state of the model taken across n_trials ICL prompts
    """
    n_test_examples = 1
    activation_storage = torch.zeros(N_TRIALS, model_config['n_layers'], model_config['resid_dim'])

    if filter_set is None:
        filter_set = np.arange(len(dataset['valid']))

    is_llama = 'llama' in model_config['name_or_path'].lower() or 'mistral' in model_config['name_or_path'].lower()
    prepend_bos = not is_llama

    for n in range(N_TRIALS):
        word_pairs = dataset['train'][np.random.choice(len(dataset['train']),n_icl_examples, replace=False)]
        word_pairs_test = dataset['valid'][np.random.choice(filter_set,n_test_examples, replace=False)]
        # print(word_pairs)
        if prefixes is not None and separators is not None:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, 
                                                    shuffle_labels=shuffle_labels, prefixes=prefixes, separators=separators)
        else:
            prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos, shuffle_labels=shuffle_labels)
        activations_td = gather_layer_activations(prompt_data=prompt_data, 
                                                  layers = model_config['layer_hook_names'], 
                                                  model=model, 
                                                  tokenizer=tokenizer)
        
        stack_initial = torch.vstack([activations_td[layer].output[0] for layer in model_config['layer_hook_names']])
        stack_filtered = stack_initial[:,-1,:] #Last token 
        # print(stack_filtered.shape)
        activation_storage[n] = stack_filtered

    mean_activations = activation_storage.mean(dim=0)
    return mean_activations

# Attention Weights
def get_value_weighted_attention(sentence, model, model_config, tokenizer):
    """

    Parameters:
    sentence: sentence to extract attention patterns for
    model: huggingface model
    model_config: dict with model information - n_layers, n_heads, etc.
    tokenizer: huggingface tokenizer

    Returns:
    attentions: attention heatmaps
    value_weighted_attn: value-weighted attention heatmaps
    """    
    inputs = tokenizer(sentence, return_tensors='pt').to(model.device)
    output = model(**inputs, output_attentions=True) # batch_size x n_tokens x vocab_size, only want last token prediction
    attentions = torch.vstack(output.attentions) # (layers, heads, tokens, tokens)
    values = torch.vstack([output.past_key_values[i][1] for i in range(model_config['n_layers'])]) # (layers, heads, tokens, head_dim)
    value_weighted_attn = torch.einsum("abcd,abd->abcd", attentions, values.norm(dim=-1))
    return attentions, value_weighted_attn

def get_token_averaged_attention(dataset, model, model_config, tokenizer, n_shots=10, storage_max=100, filter_set=None):
    """

    Parameters:
    dataset: ICL dataset
    model: huggingface model
    model_config: dict with model information - n_layers, n_heads, etc.
    tokenizer: huggingface tokenizer
    n_shots: number of ICL example pairs to use for each prompt
    storage_max: max number of sentences to average attention pattern over
    filter_set: list of ints to filter to desired dataset instances

    Returns:
    attn_storage: attention heatmaps
    vw_attn_storage: value-weighted attention heatmaps
    token_labels: sample token labels for an n-shot prompt
    """
    if filter_set is not None:
        storage_size = min(len(filter_set), storage_max)
        storage_inds = [int(x) for x in filter_set[:storage_size]]
    else:
        storage_size = min(len(dataset['valid']), storage_max)
        storage_inds = [int(x) for x in np.arange(storage_size)]

    dummy_labels = get_dummy_token_labels(n_shots, tokenizer=tokenizer)  
    attn_storage = torch.zeros(storage_size, model_config['n_layers'], model_config['n_heads'], len(dummy_labels), len(dummy_labels))
    vw_attn_storage = torch.zeros(storage_size, model_config['n_layers'], model_config['n_heads'], len(dummy_labels), len(dummy_labels))

    for ind,s in enumerate(storage_inds):
        if n_shots == 0:
            word_pairs = {'input':[], 'output':[]}
        else:
            word_pairs = dataset['train'][np.random.choice(len(dataset['train']),n_shots, replace=False)]
        
        is_llama = 'llama' in model_config['name_or_path'].lower() or 'mistral' in model_config['name_or_path'].lower()
        prepend_bos = not is_llama

        word_pairs_test = dataset['valid'][s]
        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token = prepend_bos)
        
        # Get relevant parts of the Prompt
        query, target = prompt_data['query_target'].values()

        token_labels, prompt_string = get_token_meta_labels(prompt_data, tokenizer, query)
        idx_map, idx_avg = compute_duplicated_labels(token_labels, dummy_labels)
        
        sentence = [prompt_string]     

        attentions, value_weighted_attn = get_value_weighted_attention(sentence, model,model_config,tokenizer)

        attentionsv2 = attentions.clone()
        vw_v2 = value_weighted_attn.clone()

        # sum attention of multi-token phrases into single token (for avging purposes)
        for (i,j) in idx_avg.values():
            attentionsv2[:,:,:,i] = attentions[:,:,:,i:j+1].sum(axis=-1)
            attentionsv2[:,:,i+1:, i+1:j+1] = 0 

            vw_v2[:,:,:,i] = value_weighted_attn[:,:,:,i:j+1].sum(axis=-1)
            vw_v2[:,:,i+1:, i+1:j+1] = 0 
        
        del attentions
        del value_weighted_attn

        token_avgd_attention = attentionsv2[:,:,list(idx_map.keys())][:,:,:,list(idx_map.keys())]
        token_avgd_vw_attention = vw_v2[:,:,list(idx_map.keys())][:,:,:,list(idx_map.keys())]

        attn_storage[ind] = token_avgd_attention
        vw_attn_storage[ind] = token_avgd_vw_attention

    return attn_storage, vw_attn_storage, token_labels

def prefix_matching_score(model, model_config, min_token_idx=1000, max_token_idx=10000, seq_len=100, batch_size=4):
    """
    Computes the prefix matching score - part of checking whether an attention head is a traditional "induction head"
    
    Parameters:
    model: huggingface model
    model_config: dict with model information - n_layers, n_heads, etc.
    min_token_idx: vocabulary token index lower bound
    max_token_idx: vocabulary token index upper bound
    seq_len: length of sequence to be duplicated
    batch_size: number of sequences to test
    
    Returns:
    score_per_head: prefix-matching score for each head in the model of size (n_layers, n_heads)
    """
    rand_tokens = torch.randint(min_token_idx, max_token_idx, (batch_size, seq_len))
    rand_tokens_repeat =  torch.concat((rand_tokens, rand_tokens), dim=1).to(model.device)

    output = model(rand_tokens_repeat, output_attentions = True)
    attentions = torch.vstack(output.attentions) # (n_layers, n_heads, tokens, tokens)

    score = attentions.diagonal(1-seq_len, dim1=-2, dim2=-1)
    score_per_head = score.reshape(model_config['n_layers'],batch_size,model_config['n_heads'],-1).mean(axis=-1).mean(axis=1).cpu().T
    
    return score_per_head

def compute_function_vector(mean_activations, indirect_effect, model, model_config, n_top_heads = 10, token_class_idx=-1, gen=False):
    """
        Computes a "function vector" vector that communicates the task observed in ICL examples used for downstream intervention.
        
        Parameters:
        mean_activations: tensor of size (Layers, Heads, Tokens, head_dim) containing the average activation of each head for a particular task
        indirect_effect: tensor of size (N, Layers, Heads, class(optional)) containing the indirect_effect of each head across N trials
        model: huggingface model being used
        model_config: contains model config information (n layers, n heads, etc.)
        n_top_heads: The number of heads to use when computing the summed function vector
        token_class_idx: int indicating which token class to use, -1 is default for last token computations

        Returns:
        function_vector: vector representing the communication of a particular task
        top_heads: list of the top influential heads represented as tuples [(L,H,S), ...], (L=Layer, H=Head, S=Avg. Indirect Effect Score)         
    """
    model_resid_dim = model_config['resid_dim']
    model_n_heads = model_config['n_heads']
    model_head_dim = model_resid_dim//model_n_heads
    device = model.device

    li_dims = len(indirect_effect.shape)
    
    if li_dims == 3 and token_class_idx == -1:
        mean_indirect_effect = indirect_effect.mean(dim=0)
    else:
        assert(li_dims == 4)
        if gen:
            mean_indirect_effect = - indirect_effect[:,:,:,1].mean(dim=0) # Subset to token class of interest
        else:
            mean_indirect_effect = indirect_effect[:,:,:,0].mean(dim=0) # Subset to token class of interest

    # Compute Top Influential Heads (L,H)
    h_shape = mean_indirect_effect.shape 
    topk_vals, topk_inds  = torch.topk(mean_indirect_effect.view(-1), k=n_top_heads, largest=True)
    print(topk_vals[:n_top_heads],topk_inds[:n_top_heads])
    top_lh = list(zip(*np.unravel_index(topk_inds, h_shape), [round(x.item(),4) for x in topk_vals]))
    top_heads = top_lh[:n_top_heads]

    # Compute Function Vector as sum of influential heads
    function_vector = torch.zeros((1,1,model_resid_dim)).to(device)
    T = -1 # Intervention & values taken from last token

    for L,H,_ in top_heads:
        if 'gpt2-xl' in model_config['name_or_path']:
            out_proj = model.transformer.h[L].attn.c_proj
        elif 'gpt-j' in model_config['name_or_path']:
            out_proj = model.transformer.h[L].attn.out_proj
        elif 'llama' in model_config['name_or_path'].lower() or 'mistral' in model_config['name_or_path'].lower():
            out_proj = model.model.layers[L].self_attn.o_proj
        elif 'gpt-neox' in model_config['name_or_path'] or 'pythia' in model_config['name_or_path']:
            out_proj = model.gpt_neox.layers[L].attention.dense

        x = torch.zeros(model_resid_dim)
        x[H*model_head_dim:(H+1)*model_head_dim] = mean_activations[L,H,T]
        d_out = out_proj(x.reshape(1,1,model_resid_dim).to(device).to(model.dtype))

        function_vector += d_out
    
    function_vector = function_vector.to(model.dtype)
    function_vector = function_vector.reshape(1, model_resid_dim)

    return function_vector, top_heads

def compute_universal_function_vector(mean_activations, model, model_config, n_top_heads=10):
    """
        Computes a "function vector" vector that communicates the task observed in ICL examples used for downstream intervention
        using the set of heads with universally highest causal effect computed across a set of ICL tasks
        
        Parameters:
        mean_activations: tensor of size (Layers, Heads, Tokens, head_dim) containing the average activation of each head for a particular task
        model: huggingface model being used
        model_config: contains model config information (n layers, n heads, etc.)
        n_top_heads: The number of heads to use when computing the function vector

        Returns:
        function_vector: vector representing the communication of a particular task
        top_heads: list of the top influential heads represented as tuples [(L,H,S), ...], (L=Layer, H=Head, S=Avg. Indirect Effect Score)         
    """
    model_resid_dim = model_config['resid_dim']
    model_n_heads = model_config['n_heads']
    model_head_dim = model_resid_dim//model_n_heads
    device = model.device

    # Universal Set of Heads
    
    if 'gpt2' in model_config['name_or_path']:
        top_heads =[
            (21, 13, 0.0802), (25, 13, 0.0368), (20, 5, 0.035), (21, 19, 0.0346), (19, 18, 0.0321), (29, 2, 0.0319), 
            (20, 16, 0.0317), (23, 16, 0.03), (29, 0, 0.0285), (22, 15, 0.0264), (26, 16, 0.0246), (23, 20, 0.0234), 
            (20, 19, 0.0215), (23, 11, 0.0206), (23, 5, 0.0197), (24, 11, 0.0191), (17, 16, 0.0189), (16, 21, 0.0178), 
            (27, 15, 0.0163), (32, 20, 0.0161), (17, 1, 0.0152), (28, 2, 0.0151), (13, 0, 0.0148), (19, 20, 0.0137), 
            (19, 10, 0.0127), (22, 11, 0.0126), (26, 22, 0.0122), (25, 19, 0.0118), (36, 8, 0.0112), (28, 19, 0.0111), 
            (24, 9, 0.0104), (39, 18, 0.0099), (19, 5, 0.0094), (18, 22, 0.0092), (33, 9, 0.009), (21, 3, 0.0085), 
            (34, 19, 0.0081), (26, 6, 0.0079), (23, 8, 0.0073), (17, 6, 0.0072), (24, 21, 0.0071), (42, 20, 0.007), 
            (21, 22, 0.007), (34, 22, 0.0069), (16, 3, 0.0068)
        ]
    elif 'gpt-j' in model_config['name_or_path']:
        top_heads = [(15, 5, 0.0587), (9, 14, 0.0584), (12, 10, 0.0526), (8, 1, 0.0445), (11, 0, 0.0445), (13, 13, 0.019), (8, 0, 0.0184), (14, 9, 0.016), (9, 2, 0.0127), (24, 6, 0.0113), (15, 11, 0.0092),
                     (6, 6, 0.0069), (14, 0, 0.0068), (17, 8, 0.0068), (21, 2, 0.0067), (10, 11, 0.0066), (11, 2, 0.0057), (17, 0, 0.0054), (20, 11, 0.0051), (23, 0, 0.0047), (20, 0, 0.0046), (15, 7, 0.0045),
                     (27, 2, 0.0045), (21, 15, 0.0044), (11, 4, 0.0044), (18, 6, 0.0043), (9, 6, 0.0042), (4, 12, 0.004), (11, 15, 0.004), (20, 2, 0.0036), (10, 0, 0.0035), (16, 9, 0.0031), (11, 14, 0.0031),
                     (12, 4, 0.003), (9, 7, 0.003), (18, 3, 0.003), (19, 5, 0.003), (22, 5, 0.0027), (25, 3, 0.0026), (18, 9, 0.0025)]
    elif 'Llama-2-7b' in model_config['name_or_path']:
        top_heads = [(14, 1, 0.0391), (11, 2, 0.0225), (9, 25, 0.02), (12, 15, 0.0196), (12, 28, 0.0191), (13, 7, 0.0171), (11, 18, 0.0152), (12, 18, 0.0113), (16, 10, 0.007), (14, 16, 0.007),
                     (14, 14, 0.0048), (16, 1, 0.0042), (18, 1, 0.0042), (19, 16, 0.0041), (13, 30, 0.0034), (18, 26, 0.0032), (14, 7, 0.0032), (16, 0, 0.0031), (16, 29, 0.003), (29, 30, 0.003),
                     (16, 6, 0.0029), (15, 11, 0.0027), (12, 11, 0.0026), (11, 22, 0.0023), (16, 19, 0.0021), (15, 23, 0.002), (16, 20, 0.0019), (15, 9, 0.0019), (17, 28, 0.0019), (14, 18, 0.0018),
                     (8, 26, 0.0018), (29, 26, 0.0018), (15, 8, 0.0018), (13, 13, 0.0017), (30, 9, 0.0017), (13, 23, 0.0017), (13, 10, 0.0016), (11, 30, 0.0016), (12, 26, 0.0015), (19, 27, 0.0015),
                     (14, 9, 0.0014), (14, 10, 0.0013), (31, 17, 0.0013), (31, 4, 0.0013), (15, 17, 0.0013), (10, 5, 0.0012), (14, 11, 0.0012), (19, 12, 0.0012), (16, 7, 0.0012), (15, 24, 0.0011),
                     (26, 28, 0.0011), (11, 15, 0.0011), (15, 25, 0.0011), (17, 12, 0.0011), (13, 2, 0.0011), (14, 5, 0.0011), (14, 3, 0.001), (26, 30, 0.001), (27, 29, 0.001), (25, 12, 0.0009),
                     (15, 13, 0.0009), (10, 14, 0.0009), (28, 13, 0.0009), (17, 19, 0.0008), (19, 2, 0.0008), (12, 23, 0.0008), (15, 26, 0.0008), (28, 21, 0.0008), (15, 10, 0.0008), (12, 0, 0.0007),
                     (6, 16, 0.0007), (7, 28, 0.0007), (27, 7, 0.0007), (11, 28, 0.0007), (29, 15, 0.0006), (13, 8, 0.0006), (13, 17, 0.0006), (8, 0, 0.0006), (22, 17, 0.0006), (22, 20, 0.0006), 
                     (12, 2, 0.0006), (26, 9, 0.0006), (31, 26, 0.0006), (22, 27, 0.0005), (16, 26, 0.0005), (13, 1, 0.0005), (26, 2, 0.0005), (30, 10, 0.0005), (11, 25, 0.0005), (29, 20, 0.0005),
                     (19, 15, 0.0005), (12, 10, 0.0005), (12, 3, 0.0005), (30, 5, 0.0004), (6, 9, 0.0004), (15, 16, 0.0004), (23, 28, 0.0004), (22, 5, 0.0004), (31, 19, 0.0004), (26, 14, 0.0004)]
    elif 'Llama-3.1-8B' in model_config['name_or_path']:
        top_heads = [(27, 28, 0.1217), (13, 27, 0.1147), (15, 28, 0.1142), (17, 8, 0.0937), (21, 2, 0.0867), (10, 12, 0.0717), (15, 16, 0.067), (15, 2, 0.0668), (15, 1, 0.066), (31, 24, 0.0605), 
                    (15, 17, 0.0592), (29, 21, 0.0592), (28, 5, 0.049), (21, 1, 0.0459), (19, 0, 0.0445), (27, 14, 0.0433), (19, 13, 0.0417), (15, 30, 0.0415), (30, 24, 0.0404), (27, 29, 0.0394), 
                    (10, 5, 0.0351), (30, 18, 0.0346), (29, 20, 0.0335), (19, 15, 0.0321), (18, 23, 0.0302), (16, 1, 0.0284), (18, 20, 0.026), (20, 12, 0.0258), (17, 24, 0.0253), (21, 10, 0.0238), 
                    (13, 31, 0.0227), (31, 30, 0.0218), (17, 25, 0.0217), (22, 19, 0.0209), (19, 2, 0.0208), (14, 19, 0.0195), (16, 25, 0.0193), (28, 10, 0.0187), (30, 11, 0.0186), (10, 7, 0.0181), 
                    (31, 0, 0.018), (18, 28, 0.0177), (28, 26, 0.0175), (30, 15, 0.0173), (25, 23, 0.0167), (30, 26, 0.0167), (15, 3, 0.0165), (10, 14, 0.0163), (11, 30, 0.0161), (16, 3, 0.016),
                    (16, 29, 0.0157), (16, 14, 0.0154), (31, 6, 0.0152), (24, 13, 0.0149), (31, 1, 0.0148), (16, 28, 0.0141), (30, 16, 0.0138), (12, 23, 0.0136), (14, 23, 0.0135), (2, 21, 0.0132), 
                    (2, 22, 0.013), (16, 20, 0.0126), (13, 8, 0.0121), (17, 18, 0.0121), (15, 21, 0.012), (27, 20, 0.0118), (20, 13, 0.0117), (26, 3, 0.0113), (30, 22, 0.0112), (19, 1, 0.0111), 
                    (29, 25, 0.011), (10, 2, 0.0108), (30, 27, 0.0108), (27, 13, 0.0108), (19, 20, 0.0108), (17, 4, 0.0104), (31, 7, 0.0104), (30, 6, 0.0104), (12, 17, 0.0101), (2, 20, 0.0098), 
                    (15, 19, 0.0095), (24, 26, 0.0095), (13, 29, 0.009), (11, 5, 0.0089), (14, 12, 0.0088), (16, 13, 0.0086), (21, 19, 0.0085), (9, 25, 0.0083), (17, 13, 0.0083), (29, 16, 0.0081), 
                    (27, 7, 0.0081), (19, 12, 0.0079), (17, 5, 0.0079), (29, 30, 0.0077), (2, 25, 0.0075), (19, 19, 0.0075), (26, 13, 0.0073), (12, 9, 0.0072), (22, 6, 0.0072), (17, 10, 0.007)]
    elif '_Mistral-7B' in model_config['name_or_path']:
        top_heads = [(14, 31, 0.1186), (26, 29, 0.0876), (12, 4, 0.0791), (12, 7, 0.0709), (30, 4, 0.0674), (30, 9, 0.0601), (22, 30, 0.0588), (14, 19, 0.0564), (11, 10, 0.047), (18, 1, 0.0429), 
                    (30, 6, 0.0422), (2, 21, 0.0412), (16, 17, 0.0409), (30, 12, 0.0396), (17, 31, 0.0373), (30, 29, 0.0367), (11, 8, 0.0352), (16, 16, 0.0348), (28, 25, 0.0336), (30, 10, 0.0328), 
                    (14, 30, 0.032), (2, 23, 0.031), (26, 5, 0.0309), (18, 2, 0.0289), (13, 5, 0.0283), (15, 12, 0.0274), (20, 21, 0.0271), (5, 5, 0.0262), (20, 23, 0.0256), (31, 22, 0.0253), 
                    (15, 25, 0.0238), (28, 21, 0.0237), (2, 0, 0.0236), (26, 9, 0.0233), (16, 19, 0.0228), (19, 9, 0.0228), (17, 8, 0.0213), (18, 0, 0.0212), (30, 17, 0.021), (20, 9, 0.0205), 
                    (30, 30, 0.0201), (31, 13, 0.02), (25, 31, 0.0195), (19, 16, 0.0192), (11, 14, 0.0192), (15, 16, 0.0191), (16, 14, 0.0189), (5, 11, 0.0185), (17, 5, 0.0184), (16, 3, 0.0178), 
                    (19, 18, 0.0176), (31, 19, 0.0172), (30, 3, 0.0171), (30, 31, 0.017), (14, 6, 0.0165), (25, 25, 0.0158), (18, 29, 0.0158), (15, 15, 0.0156), (31, 4, 0.0155), (17, 2, 0.0149), 
                    (19, 8, 0.0148), (14, 18, 0.0147), (21, 19, 0.0144), (13, 23, 0.0141), (12, 17, 0.014), (11, 13, 0.0134), (30, 7, 0.0132), (31, 25, 0.0126), (31, 14, 0.0125), (21, 11, 0.0125), 
                    (18, 24, 0.0122), (31, 8, 0.0116), (19, 24, 0.0115), (31, 9, 0.0114), (16, 12, 0.0113), (4, 15, 0.0111), (18, 9, 0.0107), (20, 10, 0.0106), (11, 2, 0.0105), (2, 22, 0.0104), 
                    (14, 28, 0.0104), (0, 3, 0.0103), (9, 3, 0.0102), (20, 11, 0.0102), (17, 22, 0.01), (2, 15, 0.0099), (3, 23, 0.0099), (28, 31, 0.0097), (18, 31, 0.0096), (29, 12, 0.0096), 
                    (5, 8, 0.0092), (30, 8, 0.0092), (21, 3, 0.0091), (15, 27, 0.0091), (29, 2, 0.009), (26, 21, 0.0089), (13, 6, 0.0087), (21, 1, 0.0086), (21, 30, 0.0085), (31, 16, 0.0084)]
    elif 'Llama-2-13b' in model_config['name_or_path']:
        top_heads = [(13, 13, 0.0402), (12, 17, 0.0332), (15, 38, 0.0269), (14, 34, 0.0209), (19, 2, 0.0116), (19, 36, 0.0106), (13, 4, 0.0106), (18, 11, 0.01), (10, 15, 0.0087), (13, 23, 0.0077),
                     (14, 7, 0.0074), (15, 36, 0.0046), (12, 8, 0.0046), (17, 7, 0.0044), (38, 29, 0.0043), (15, 32, 0.0037), (17, 18, 0.0034), (16, 9, 0.0033), (14, 23, 0.0032), (39, 13, 0.0029),
                     (39, 14, 0.0027), (18, 22, 0.0026), (21, 32, 0.0026), (15, 18, 0.0026), (13, 14, 0.0026), (11, 31, 0.0025), (14, 39, 0.0024), (19, 14, 0.0023), (36, 23, 0.0021), (21, 7, 0.0021),
                     (8, 23, 0.002), (18, 18, 0.002), (17, 28, 0.002), (17, 9, 0.0019), (13, 27, 0.0017), (13, 34, 0.0017), (13, 12, 0.0016), (21, 2, 0.0016), (16, 16, 0.0015), (15, 31, 0.0015),
                     (26, 35, 0.0015), (10, 18, 0.0014), (11, 27, 0.0014), (13, 25, 0.0014), (15, 26, 0.0013), (5, 32, 0.0013), (20, 12, 0.0013), (18, 15, 0.0013), (16, 23, 0.0013), (25, 5, 0.0013),
                     (34, 6, 0.0012), (15, 2, 0.0012), (15, 27, 0.0012), (18, 20, 0.0012), (16, 19, 0.0011), (37, 4, 0.001), (19, 7, 0.001), (19, 3, 0.0009), (38, 14, 0.0009), (20, 21, 0.0009),
                     (21, 30, 0.0009), (16, 11, 0.0009), (13, 24, 0.0009), (9, 31, 0.0008), (14, 13, 0.0008), (16, 29, 0.0008), (15, 17, 0.0008), (19, 6, 0.0008), (23, 36, 0.0008), (18, 17, 0.0007),
                     (15, 34, 0.0007), (14, 29, 0.0007), (15, 7, 0.0007), (13, 17, 0.0007), (20, 11, 0.0007), (35, 16, 0.0007), (39, 27, 0.0007), (29, 27, 0.0006), (30, 24, 0.0006), (19, 37, 0.0006),
                     (39, 21, 0.0006), (13, 36, 0.0006), (37, 30, 0.0006), (16, 36, 0.0006), (15, 3, 0.0006), (19, 13, 0.0006), (13, 10, 0.0006), (14, 19, 0.0005), (36, 3, 0.0005), (15, 25, 0.0005),
                     (16, 0, 0.0005), (16, 10, 0.0005), (20, 29, 0.0005), (25, 13, 0.0005), (14, 36, 0.0005), (36, 7, 0.0005), (17, 0, 0.0005), (11, 37, 0.0005), (23, 18, 0.0005), (35, 10, 0.0005)]
    elif 'Llama-2-70b' in model_config['name_or_path']:
        top_heads = [(33, 63, 0.0315), (36, 3, 0.0313), (29, 7, 0.0193), (40, 50, 0.0147), (26, 57, 0.0136), (40, 57, 0.0134), (40, 54, 0.0127), (36, 0, 0.011), (29, 3, 0.0109), (39, 61, 0.0085),
                     (77, 8, 0.0082), (14, 29, 0.0079), (39, 26, 0.0074), (37, 17, 0.0069), (40, 55, 0.0066), (34, 40, 0.0064), (39, 56, 0.0063), (34, 41, 0.0061), (36, 54, 0.0058), (29, 1, 0.0058),
                     (38, 20, 0.0053), (40, 48, 0.0051), (39, 30, 0.005), (34, 60, 0.0048), (34, 42, 0.0045), (26, 62, 0.0044), (77, 15, 0.0044), (77, 14, 0.0042), (43, 63, 0.0041), (31, 27, 0.004),
                     (31, 20, 0.004), (40, 6, 0.0038), (44, 63, 0.0036), (36, 41, 0.0034), (79, 34, 0.0033), (46, 31, 0.0033), (29, 4, 0.0033), (39, 36, 0.0032), (42, 10, 0.0031), (14, 30, 0.003),
                     (26, 25, 0.0029), (40, 61, 0.0028), (40, 39, 0.0028), (34, 25, 0.0028), (39, 59, 0.0027), (34, 56, 0.0025), (26, 31, 0.0025), (43, 4, 0.0025), (11, 21, 0.0024), (47, 44, 0.0023),
                     (76, 44, 0.0022), (38, 18, 0.0022), (75, 62, 0.0022), (21, 32, 0.0021), (51, 41, 0.002), (36, 32, 0.002), (44, 59, 0.0019), (43, 27, 0.0019), (40, 51, 0.0019), (32, 3, 0.0019),
                     (38, 11, 0.0018), (32, 11, 0.0018), (35, 2, 0.0018), (25, 13, 0.0018), (42, 12, 0.0017), (25, 3, 0.0017), (24, 0, 0.0017), (38, 3, 0.0017), (34, 46, 0.0016), (31, 5, 0.0016),
                     (38, 55, 0.0016), (40, 21, 0.0016), (40, 33, 0.0016), (77, 25, 0.0015), (42, 18, 0.0015), (35, 34, 0.0015), (7, 63, 0.0014), (24, 45, 0.0014), (39, 34, 0.0014), (27, 35, 0.0014),
                     (38, 34, 0.0014), (38, 19, 0.0013), (41, 33, 0.0013), (18, 61, 0.0013), (22, 36, 0.0013), (38, 51, 0.0013), (25, 7, 0.0013), (29, 17, 0.0012), (28, 45, 0.0012), (35, 8, 0.0012),
                     (69, 17, 0.0012), (72, 26, 0.0012), (44, 18, 0.0012), (43, 7, 0.0012), (76, 34, 0.0011), (10, 62, 0.0011), (14, 31, 0.0011), (45, 57, 0.0011), (25, 14, 0.0011), (30, 15, 0.0011),
                     (47, 1, 0.0011), (15, 46, 0.0011), (27, 57, 0.001), (37, 37, 0.001), (30, 9, 0.001), (16, 28, 0.001), (28, 7, 0.001), (29, 18, 0.001), (35, 5, 0.001), (14, 28, 0.001), (72, 24, 0.001),
                     (37, 10, 0.001), (26, 63, 0.001), (72, 29, 0.001), (39, 13, 0.001), (77, 59, 0.0009), (76, 36, 0.0009), (23, 59, 0.0009), (39, 35, 0.0009), (43, 16, 0.0009), (33, 49, 0.0009),
                     (33, 31, 0.0009), (29, 19, 0.0009), (43, 2, 0.0009), (40, 45, 0.0009), (76, 50, 0.0009), (38, 35, 0.0009), (39, 28, 0.0009), (20, 4, 0.0009), (36, 2, 0.0008), (38, 12, 0.0008),
                     (20, 47, 0.0008), (78, 44, 0.0008), (39, 57, 0.0008), (30, 26, 0.0008), (63, 52, 0.0008), (7, 62, 0.0008), (30, 6, 0.0008), (25, 10, 0.0008), (76, 32, 0.0008), (36, 45, 0.0008),
                     (27, 44, 0.0008), (38, 58, 0.0008), (38, 6, 0.0008), (36, 46, 0.0008), (31, 21, 0.0008), (22, 38, 0.0007), (36, 44, 0.0007), (71, 61, 0.0007), (37, 15, 0.0007), (39, 31, 0.0007),
                     (27, 48, 0.0007), (24, 41, 0.0007), (43, 49, 0.0007), (40, 26, 0.0007), (13, 31, 0.0007), (21, 34, 0.0007), (26, 61, 0.0007), (36, 11, 0.0007), (28, 34, 0.0007), (22, 18, 0.0007),
                     (34, 3, 0.0007), (40, 52, 0.0007), (32, 37, 0.0006), (76, 13, 0.0006), (74, 58, 0.0006), (43, 24, 0.0006), (17, 2, 0.0006), (21, 4, 0.0006), (59, 50, 0.0006), (37, 44, 0.0006),
                     (27, 46, 0.0006), (69, 28, 0.0006), (29, 11, 0.0006), (31, 25, 0.0006), (20, 18, 0.0006), (40, 63, 0.0006), (37, 19, 0.0006), (36, 23, 0.0006), (34, 13, 0.0006), (69, 19, 0.0006),
                     (44, 17, 0.0006), (44, 32, 0.0005), (26, 23, 0.0005), (42, 13, 0.0005), (34, 18, 0.0005), (75, 56, 0.0005), (37, 14, 0.0005), (25, 50, 0.0005), (42, 61, 0.0005), (43, 1, 0.0005),
                     (77, 27, 0.0005), (40, 24, 0.0005), (63, 50, 0.0005), (24, 25, 0.0005), (43, 30, 0.0005), (79, 23, 0.0005), (38, 62, 0.0005), (23, 9, 0.0005), (35, 30, 0.0005), (32, 34, 0.0005),
                     (39, 60, 0.0005), (29, 63, 0.0005), (55, 8, 0.0005), (6, 12, 0.0005), (39, 47, 0.0005), (44, 14, 0.0005), (36, 47, 0.0005), (6, 34, 0.0005), (41, 8, 0.0005), (36, 1, 0.0005),
                     (30, 22, 0.0005), (52, 20, 0.0005), (52, 56, 0.0004), (64, 23, 0.0004), (74, 5, 0.0004), (41, 63, 0.0004), (67, 23, 0.0004), (17, 23, 0.0004), (49, 23, 0.0004), (76, 39, 0.0004),
                     (49, 59, 0.0004), (18, 30, 0.0004), (37, 8, 0.0004), (23, 27, 0.0004), (36, 43, 0.0004), (57, 3, 0.0004), (39, 37, 0.0004), (37, 61, 0.0004), (39, 25, 0.0004), (25, 25, 0.0004),
                     (23, 38, 0.0004), (38, 49, 0.0004), (35, 27, 0.0004), (32, 9, 0.0004), (69, 30, 0.0004), (25, 9, 0.0004), (39, 32, 0.0004), (34, 57, 0.0004), (40, 47, 0.0004), (19, 51, 0.0004),
                     (16, 0, 0.0004), (20, 19, 0.0004), (44, 57, 0.0004), (40, 34, 0.0004), (79, 25, 0.0004), (69, 27, 0.0004), (76, 26, 0.0004), (26, 30, 0.0004), (72, 31, 0.0004), (26, 29, 0.0004),
                     (55, 15, 0.0004), (33, 58, 0.0004), (18, 25, 0.0004), (25, 2, 0.0004), (33, 27, 0.0004), (20, 40, 0.0004), (24, 27, 0.0004), (17, 3, 0.0004), (18, 62, 0.0004), (47, 7, 0.0004),
                     (33, 28, 0.0004), (31, 11, 0.0004), (24, 28, 0.0004), (37, 7, 0.0004), (40, 7, 0.0004), (32, 61, 0.0004)]
    elif 'gpt-neox' in model_config['name_or_path']:
        top_heads = [(9, 42, 0.0293), (12, 4, 0.0224), (9, 28, 0.019), (11, 57, 0.0079), (10, 43, 0.0073), (12, 14, 0.0069), (14, 31, 0.0065), (9, 23, 0.0057), (11, 21, 0.0054), (11, 4, 0.0052),
                     (9, 21, 0.0052), (18, 23, 0.005), (13, 9, 0.0048), (14, 49, 0.0048), (12, 20, 0.0047), (8, 30, 0.0045), (12, 59, 0.0043), (16, 42, 0.0039), (11, 34, 0.0038), (9, 33, 0.0038),
                     (9, 3, 0.0036), (11, 48, 0.0035), (14, 63, 0.0032), (18, 11, 0.0032), (13, 7, 0.003), (9, 27, 0.0029), (11, 23, 0.0029), (16, 30, 0.0027), (10, 17, 0.0026), (9, 55, 0.0024),
                     (11, 38, 0.0024), (11, 59, 0.0024), (20, 8, 0.0024), (15, 42, 0.0023), (11, 47, 0.0023), (9, 15, 0.0023), (8, 47, 0.0023), (10, 40, 0.0023), (18, 18, 0.0022), (9, 1, 0.0021),
                     (13, 12, 0.0021), (14, 5, 0.002), (16, 18, 0.0019), (13, 63, 0.0019), (9, 20, 0.0018), (26, 38, 0.0018), (21, 60, 0.0017), (17, 55, 0.0016), (17, 30, 0.0016), (10, 56, 0.0015),
                     (12, 3, 0.0015), (10, 16, 0.0014), (10, 0, 0.0013), (15, 62, 0.0013), (12, 15, 0.0013), (9, 34, 0.0013), (12, 18, 0.0013), (23, 46, 0.0012), (16, 53, 0.0012), (11, 1, 0.0011),
                     (9, 2, 0.0011), (10, 27, 0.0011), (23, 54, 0.0011), (16, 54, 0.0011), (12, 30, 0.0011), (11, 14, 0.0011), (16, 44, 0.001), (14, 27, 0.001), (26, 31, 0.001), (15, 0, 0.001),
                     (13, 46, 0.001), (15, 57, 0.001), (15, 17, 0.001), (19, 12, 0.0009), (9, 49, 0.0009), (10, 7, 0.0009), (19, 46, 0.0009), (8, 21, 0.0009), (25, 24, 0.0008), (19, 29, 0.0008),
                     (12, 21, 0.0008), (8, 18, 0.0008), (12, 35, 0.0008), (9, 10, 0.0008), (19, 40, 0.0008), (38, 5, 0.0008), (13, 31, 0.0007), (10, 38, 0.0007), (10, 12, 0.0007), (11, 31, 0.0007),
                     (10, 1, 0.0007), (23, 15, 0.0007), (13, 40, 0.0007), (9, 5, 0.0007), (22, 33, 0.0007), (13, 36, 0.0006), (8, 32, 0.0006), (16, 21, 0.0006), (14, 11, 0.0006), (13, 61, 0.0006)]
    
    top_heads = top_heads[:n_top_heads]

    # Compute Function Vector as sum of influential heads
    function_vector = torch.zeros((1,1,model_resid_dim)).to(device)
    T = -1 # Intervention & values taken from last token

    for L,H,_ in top_heads:
        if 'gpt2-xl' in model_config['name_or_path']:
            out_proj = model.transformer.h[L].attn.c_proj
        elif 'gpt-j' in model_config['name_or_path']:
            out_proj = model.transformer.h[L].attn.out_proj
        elif 'llama' in model_config['name_or_path'].lower():
            out_proj = model.model.layers[L].self_attn.o_proj
        elif 'mistral' in model_config['name_or_path'].lower():
            out_proj = model.model.layers[L].self_attn.o_proj
        elif 'gpt-neox' in model_config['name_or_path']:
            out_proj = model.gpt_neox.layers[L].attention.dense

        x = torch.zeros(model_resid_dim)
        x[H*model_head_dim:(H+1)*model_head_dim] = mean_activations[L,H,T]
        d_out = out_proj(x.reshape(1,1,model_resid_dim).to(device).to(model.dtype))

        function_vector += d_out
        function_vector = function_vector.to(model.dtype)
    function_vector = function_vector.reshape(1, model_resid_dim)

    return function_vector, top_heads