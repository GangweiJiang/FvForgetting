from src.baukit import TraceDict, get_module
import torch
import re
import bitsandbytes as bnb

def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def replace_activation_w_avg(layer_head_token_pairs, avg_activations, model, model_config, batched_input=False, last_token_only=False):
    """
    An intervention function for replacing activations with a computed average value.
    This function replaces the output of one (or several) attention head(s) with a pre-computed average value 
    (usually taken from another set of runs with a particular property).
    The batched_input flag is used for systematic interventions where we are sweeping over all attention heads for a given (layer,token)
    The last_token_only flag is used for interventions where we only intervene on the last token (such as zero-shot or concept-naming)

    Parameters:
    layer_head_token_pairs: list of tuple triplets each containing a layer index, head index, and token index [(L,H,T), ...]
    avg_activations: torch tensor of the average activations (across ICL prompts) for each attention head of the model.
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    idx_map: dict mapping prompt label indices to ground truth label indices
    batched_input: whether or not to batch the intervention across all heads
    last_token_only: whether our intervention is only at the last token

    Returns: 
    rep_act: A function that specifies how to replace activations with an average when given a hooked pytorch module.
    """
    edit_layers = [x[0] for x in layer_head_token_pairs]

    def rep_act(output, layer_name, inputs):
        current_layer = int(layer_name.split('.')[2])
        if current_layer in edit_layers:    
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            
            # Determine shapes for intervention
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
            inputs = inputs.view(*new_shape) # inputs shape: (batch_size , tokens (n), heads, hidden_dim)
            
            # Perform Intervention:
            # if batched_input:
            # # Patch activations from avg activations into baseline sentences (i.e. n_head baseline sentences being modified in this case)
            #     for i in range(model_config['n_heads']):
            #         layer, head_n, token_n = layer_head_token_pairs[i]
            #         inputs[i, token_n, head_n] += avg_activations[layer, head_n, idx_map[token_n]]
            # if last_token_only:
            # Patch activations only at the last token for interventions like
            for (layer,head_n,token_n) in layer_head_token_pairs:
                if layer == current_layer:
                    # print(layer,head_n,token_n,idx_map[token_n], len(avg_activations[0,0,:]))
                    inputs[-1,token_n,head_n] += avg_activations[layer,head_n,-1]
            # else:
            # # Patch activations into baseline sentence found at index, -1 of the batch (targeted & multi-token patching)
            #     for (layer, head_n, token_n) in layer_head_token_pairs:
            #         if layer == current_layer:
            #             inputs[-1, token_n, head_n] += avg_activations[layer,head_n,idx_map[token_n]]
            
            inputs = inputs.view(*original_shape)
            proj_module = get_module(model, layer_name)
            out_proj = proj_module.weight

            if 'gpt2-xl' in model_config['name_or_path']: # GPT2-XL uses Conv1D (not nn.Linear) & has a bias term, GPTJ does not
                out_proj_bias = proj_module.bias
                new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj)
                
            elif 'gpt-j' in model_config['name_or_path']:
                new_output = torch.matmul(inputs, out_proj.T)

            elif 'gpt-neox' in model_config['name_or_path'] or 'pythia' in model_config['name_or_path']:
                out_proj_bias = proj_module.bias
                new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj.T)
            
            elif 'llama' in model_config['name_or_path'].lower() or 'mistral' in model_config['name_or_path'].lower():
                if '70b' in model_config['name_or_path']:
                    # need to dequantize weights
                    out_proj_dequant = bnb.functional.dequantize_4bit(out_proj.data, out_proj.quant_state)
                    new_output = torch.matmul(inputs, out_proj_dequant.T)
                else:
                    new_output = torch.matmul(inputs, out_proj.T)
            
            return new_output
        else:
            return output

    return rep_act

def add_function_vector(edit_layer, fv_vector, device, idx=-1):
    """
    Adds a vector to the output of a specified layer in the model

    Parameters:
    edit_layer: the layer to perform the FV intervention
    fv_vector: the function vector to add as an intervention
    device: device of the model (cuda gpu or cpu)
    idx: the token index to add the function vector at

    Returns:
    add_act: a fuction specifying how to add a function vector to a layer's output hidden state
    """
    def add_act(output, layer_name):
        current_layer = int(layer_name.split(".")[2])
        if current_layer == edit_layer:
            if isinstance(output, tuple):
                output[0][:, idx] += fv_vector.to(device)
                return output
            else:
                return output
        else:
            return output

    return add_act

def function_vector_intervention(sentence, target, edit_layer, function_vector, model, model_config, tokenizer, compute_nll=False,
                                  generate_str=False, compute_clean=True, dataname=None):
    """
    Runs the model on the sentence and adds the function_vector to the output of edit_layer as a model intervention, predicting a single token.
    Returns the output of the model with and without intervention.

    Parameters:
    sentence: the sentence to be run through the model
    target: expected response of the model (str, or [str])
    edit_layer: layer at which to add the function vector
    function_vector: torch vector that triggers execution of a task
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    compute_nll: whether to compute the negative log likelihood of a teacher-forced completion (used to compute perplexity (PPL))
    generate_str: whether to generate a string of tokens or predict a single token

    Returns:
    fvi_output: a tuple containing output results of a clean run and intervened run of the model
    """
    # Clean Run, No Intervention:
    device = model.device
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    if generate_str:
        original_pred_idx = -1
    else:
        original_pred_idx = len(inputs.input_ids.squeeze()) - 1

    if compute_nll:
        target_completion = "".join(sentence + target)
        nll_inputs = tokenizer(target_completion, return_tensors='pt').to(device)
        nll_targets = nll_inputs.input_ids.clone()
        target_len = len(nll_targets.squeeze()) - len(inputs.input_ids.squeeze()) 
        nll_targets[:,:-target_len] = -100  # This is the accepted value to skip indices when computing loss (see nn.CrossEntropyLoss default)
    
    clean_output = None
    clean_nll = None
    if compute_clean:
        if compute_nll:
            output = model(**nll_inputs, labels=nll_targets)
            clean_nll = output.loss.item()
            clean_output = output.logits[:,original_pred_idx,:]
        elif generate_str:
            MAX_NEW_TOKENS = 16
            output = model.generate(inputs.input_ids, top_p=0.9, temperature=0.1,
                                    max_new_tokens=MAX_NEW_TOKENS)
            clean_output = tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])
        else:
            clean_output = model(**inputs).logits[:,-1,:]

    # Perform Intervention

    intervention_fn = add_function_vector(edit_layer, function_vector.reshape(1, model_config['resid_dim']), model.device, idx=original_pred_idx)
    with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=intervention_fn):     
        if compute_nll:
            output = model(**nll_inputs, labels=nll_targets)
            intervention_nll = output.loss.item()
            intervention_output = output.logits[:,original_pred_idx,:]
        elif generate_str:

            if dataname in ["mmlu_stem_mc","piqa_mc","arc_c_mc","cmmlu_stem_mc","icl_arc_c","icl_hellaswag","icl_mmlu_stem",
                        "mmlu_stem_icl","hellaswag_icl","arc_c","arc_c_icl",'coqa', 'icl_coqa',
                        "word_sort","ob_count","ob_count_icl", "icl_ob_count","boolq",
                        'ni339', 'ni363', 'ni1292', 'ni1310', 'ni1343', 'ni195', 
                        'ni339_icl', 'ni363_icl', 'ni1292_icl', 'ni1310_icl', 'ni1343_icl', 'ni195_icl',
                        'ni472', 'ni061', 'ni611', 'ni214', 'ni141', 'ni198', 
                        'ni220', 'ni231', 'ni163', 'ni141', 'ni273', 'ni224',
                        'ni472_icl', 'ni061_icl', 'ni611_icl', 'ni214_icl', 'ni141_icl', 'ni198_icl', 
                        'ni220_icl', 'ni231_icl', 'ni163_icl', 'ni141_icl', 'ni273_icl', 'ni224_icl',
                        'lambada', 'icl_lambada', 'summedits', 'wic', 'icl_wic', 'obqa_train', 'obqa_icl_train']:
                MAX_NEW_TOKENS = 6
            elif dataname in ['ni1510_icl', 'ni1510_icl', 'ni292_icl', 'ni292', 'nq', 'icl_nq']:
                MAX_NEW_TOKENS = 12
            elif dataname in ["hellaswag", 'mmlu_stem', 'ni002_icl', 'ni002']:
                MAX_NEW_TOKENS = 20
            elif dataname in [ 'ni618', 'ni618_icl', 'ni589', 'ni589_icl', 
                                'ni1355', 'ni024', 'ni488', 'ni1355_icl', 'ni024_icl', 'ni488_icl']:
                MAX_NEW_TOKENS = 30
            elif 'gsm8k' in dataname or dataname in ['ni130_icl', 'ni130']:
                MAX_NEW_TOKENS = 128
            else:
                MAX_NEW_TOKENS = 64
            output = model.generate(inputs.input_ids, top_p=0.9, temperature=0.1,
                                    max_new_tokens=MAX_NEW_TOKENS)
            intervention_output = tokenizer.decode(output.squeeze()[inputs.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        else:
            intervention_output = model(**inputs).logits[:,-1,:] # batch_size x n_tokens x vocab_size, only want last token prediction
    
    fvi_output = (clean_output, intervention_output)
    if compute_nll:
        fvi_output += (clean_nll, intervention_nll)
    
    return fvi_output


def fv_intervention_natural_text(sentence, edit_layer, function_vector, model, model_config, tokenizer, max_new_tokens=16, num_interv_tokens=None, do_sample=False):
    """
    Allows for intervention in natural text where we generate and intervene on several tokens in a row.

    Parameters:
    sentence: sentence to intervene on with the FV
    edit_layer: layer at which to add the function vector
    function_vector: vector to add to the model that triggers execution of a task
    model: huggingface model
    model_config: dict with model config parameters (n_layers, n_heads, etc.)
    tokenizer: huggingface tokenizer
    max_new_tokens: number of tokens to generate
    num_interv_tokens: number of tokens to apply the intervention for (defaults to all subsequent generations)
    do_sample: whether to sample from top p tokens (True) or have deterministic greedy decoding (False)

    Returns:
    clean_output: tokens of clean output
    intervention_output: tokens of intervention output

    """
    # Clean Run, No Intervention:
    device = model.device
    inputs = tokenizer(sentence, return_tensors='pt').to(device)    
    clean_output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    # Perform Intervention
    intervention_fn = add_function_vector(edit_layer, function_vector, model.device)
    
    if num_interv_tokens is not None and num_interv_tokens < max_new_tokens: # Intervene only for a certain number of tokens
        num_extra_tokens = max_new_tokens - num_interv_tokens
        with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=intervention_fn):     
            intervention_output = model.generate(**inputs, max_new_tokens = num_interv_tokens, do_sample=do_sample, pad_token_id=tokenizer.eos_token_id)
        intervention_output = model.generate(intervention_output, max_new_tokens=num_extra_tokens, pad_token_id=tokenizer.eos_token_id, do_sample=do_sample)
    else:
        with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=intervention_fn):     
            intervention_output = model.generate(**inputs, max_new_tokens = max_new_tokens, do_sample=do_sample, pad_token_id=tokenizer.eos_token_id)

    return clean_output, intervention_output


def add_avg_to_activation(layer_head_token_pairs, avg_activations, model, model_config, batched_input=False, last_token_only=False):
    """
    An intervention function for adding a computed average value to activations.
    This function adds a pre-computed average value to the output of one (or several) attention head(s) 
    (usually taken from another set of runs with a particular property).
    The batched_input flag is used for systematic interventions where we are sweeping over all attention heads for a given (layer,token)
    The last_token_only flag is used for interventions where we only intervene on the last token (such as zero-shot or concept-naming)

    Parameters:
    layer_head_token_pairs: list of tuple triplets each containing a layer index, head index, and token index [(L,H,T), ...]
    avg_activations: torch tensor of the average activations (across ICL prompts) for each attention head of the model.
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    batched_input: whether or not to batch the intervention across all heads
    last_token_only: whether our intervention is only at the last token    

    Returns:
    add_act: A function that specifies how to replace activations with an average when given a hooked pytorch module.
    """
    edit_layers = [x[0] for x in layer_head_token_pairs]
    device = model.device

    def add_act(output, layer_name, inputs):
        current_layer = int(layer_name.split('.')[2])
        if current_layer in edit_layers:    
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            
            # Determine shapes for intervention
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
            inputs = inputs.view(*new_shape) # inputs shape: (batch_size , tokens (n), heads, hidden_dim)
            
            # Perform Intervention:
            if batched_input:
            # Patch activations from avg activations into baseline sentences (i.e. n_head baseline sentences being modified in this case)
                for i in range(model_config['n_heads']):
                    layer, head_n, token_n = layer_head_token_pairs[i]
                    inputs[i, token_n, head_n] += avg_activations[layer, head_n, token_n].to(device)
            elif last_token_only:
            # Patch activations only at the last token for interventions like: (zero-shot, concept-naming, etc.)
                for (layer,head_n,token_n) in layer_head_token_pairs:
                    if layer == current_layer:
                        inputs[-1,-1,head_n] += avg_activations[layer,head_n,token_n].to(device)
            else:
            # Patch activations into baseline sentence found at index, -1 of the batch (targeted & multi-token patching)
                for (layer, head_n, token_n) in layer_head_token_pairs:
                    if layer == current_layer:
                        inputs[-1, token_n, head_n] += avg_activations[layer,head_n,token_n].to(device)
            
            inputs = inputs.view(*original_shape)
            proj_module = get_module(model, layer_name)
            out_proj = proj_module.weight

            if 'gpt2-xl' in model_config['name_or_path']: # GPT2-XL uses Conv1D (not nn.Linear) & has a bias term, GPTJ does not
                out_proj_bias = proj_module.bias
                new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj)

            elif 'gpt-j' in model_config['name_or_path']:
                new_output = torch.matmul(inputs, out_proj.T)

            elif 'gpt-neox' in model_config['name_or_path'] or 'pythia' in model_config['name_or_path']:
                out_proj_bias = proj_module.bias
                new_output = torch.addmm(out_proj_bias, inputs.squeeze(), out_proj.T)
            
            elif 'llama' in model_config['name_or_path'].lower() or 'mistral' in model_config['name_or_path'].lower():
                if '70b' in model_config['name_or_path']:
                    # need to dequantize weights
                    out_proj_dequant = bnb.functional.dequantize_4bit(out_proj.data, out_proj.quant_state)
                    new_output = torch.matmul(inputs, out_proj_dequant.T)
                else:
                    new_output = torch.matmul(inputs, out_proj.T)
            
            return new_output
        else:
            return output

    return add_act


def function_vector_intervention_all(sentence, target, edit_layer, avg_activitions, model, model_config, tokenizer, compute_nll=False,
                                  generate_str=False, compute_clean=True):
    """
    Runs the model on the sentence and adds the function_vector to the output of edit_layer as a model intervention, predicting a single token.
    Returns the output of the model with and without intervention.

    Parameters:
    sentence: the sentence to be run through the model
    target: expected response of the model (str, or [str])
    edit_layer: layer at which to add the function vector
    function_vector: torch vector that triggers execution of a task
    model: huggingface model
    model_config: contains model config information (n layers, n heads, etc.)
    tokenizer: huggingface tokenizer
    compute_nll: whether to compute the negative log likelihood of a teacher-forced completion (used to compute perplexity (PPL))
    generate_str: whether to generate a string of tokens or predict a single token

    Returns:
    fvi_output: a tuple containing output results of a clean run and intervened run of the model
    """
    # Clean Run, No Intervention:
    device = model.device
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    original_pred_idx = len(inputs.input_ids.squeeze()) - 1

    if compute_nll:
        target_completion = "".join(sentence + target)
        nll_inputs = tokenizer(target_completion, return_tensors='pt').to(device)
        nll_targets = nll_inputs.input_ids.clone()
        target_len = len(nll_targets.squeeze()) - len(inputs.input_ids.squeeze()) 
        nll_targets[:,:-target_len] = -100  # This is the accepted value to skip indices when computing loss (see nn.CrossEntropyLoss default)
    
    clean_output = None
    clean_nll = None
    if compute_clean:
        if compute_nll:
            output = model(**nll_inputs, labels=nll_targets)
            clean_nll = output.loss.item()
            clean_output = output.logits[:,original_pred_idx,:]
        elif generate_str:
            MAX_NEW_TOKENS = 16
            output = model.generate(inputs.input_ids, top_p=0.9, temperature=0.1,
                                    max_new_tokens=MAX_NEW_TOKENS)
            clean_output = tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])
        else:
            clean_output = model(**inputs).logits[:,-1,:]

    # Perform Intervention
    pos = [(14, 1, 0.0391), (11, 2, 0.0225), (9, 25, 0.02), (12, 15, 0.0196), (12, 28, 0.0191), (13, 7, 0.0171), (11, 18, 0.0152), (12, 18, 0.0113), (16, 10, 0.007), (14, 16, 0.007),]
    intervention_locations = [(layer, head_n, -1) for (layer, head_n, _) in pos[:10]]
    intervention_fn = replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
                                                model=model, model_config=model_config,
                                                batched_input=False, idx_map=idx_map, last_token_only=last_token_only)
    intervention_fn = add_function_vector(edit_layer, function_vector.reshape(1, model_config['resid_dim']), model.device)
    with TraceDict(model, layers=model_config['layer_hook_names'], edit_output=intervention_fn):     
        if compute_nll:
            output = model(**nll_inputs, labels=nll_targets)
            intervention_nll = output.loss.item()
            intervention_output = output.logits[:,original_pred_idx,:]
        elif generate_str:
            output = model.generate(inputs.input_ids, top_p=0.9, temperature=0.1,
                                    max_new_tokens=MAX_NEW_TOKENS)
            intervention_output = tokenizer.decode(output.squeeze()[-MAX_NEW_TOKENS:])
        else:
            intervention_output = model(**inputs).logits[:,-1,:] # batch_size x n_tokens x vocab_size, only want last token prediction
    
    fvi_output = (clean_output, intervention_output)
    if compute_nll:
        fvi_output += (clean_nll, intervention_nll)
    
    return fvi_output