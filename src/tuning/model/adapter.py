import torch

import logging

logger = logging.getLogger(__name__)


def init_adapter(
    model,
    model_args,
    training_args,
    is_trainable=True
):
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """

    if (not is_trainable) and model_args.adapter_name_or_path is None:
        logger.info("Adapter is not found at evaluation, load the base model.")
        return model

    if training_args.finetuning_type == "full" and is_trainable:
        logger.info("Fine-tuning method: Full")
        model = model.float()

    if training_args.finetuning_type == "lora":
        from peft import PeftModel, TaskType, LoraConfig, get_peft_model
        logger.info("Fine-tuning method: LoRA")
        adapter_to_resume = None
        print(model_args.adapter_name_or_path)
        adapter_name_or_path = model_args.adapter_name_or_path.split(",") if model_args.adapter_name_or_path is not None else None
        if adapter_name_or_path is not None:
            is_mergeable = True
            if (is_trainable and not training_args.create_new_adapter) or (not is_mergeable):
                adapter_to_merge = adapter_name_or_path[:-1]
                adapter_to_resume = adapter_name_or_path[-1]
            else:
                adapter_to_merge = adapter_name_or_path

            print(adapter_to_merge)
            for adapter in adapter_to_merge:
                model = PeftModel.from_pretrained(model, adapter)
                # for n, p in model.named_parameters():
                #     # if p.requires_grad:
                #     # if any([x in n for x in ["router", "A", "z"]]):
                #     print(n, p.size())
                # if not ("13b" in model_args.model_name_or_path or  "70b" in model_args.model_name_or_path):
                model = model.merge_and_unload()
            
            if len(adapter_to_merge) > 0:
                logger.info("Merged {} adapter(s).".format(len(adapter_to_merge)))
                print("Merged {} adapter(s).".format(len(adapter_to_merge)))

            if adapter_to_resume is not None: # resume lora training
                model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_trainable)

        if is_trainable and adapter_to_resume is None: # create new lora weights while training

            target_modules = training_args.lora_target.split(',')

            peft_kwargs = {
                "r": training_args.lora_rank,
                "target_modules": target_modules,
                "lora_alpha": training_args.lora_alpha,
                "lora_dropout": training_args.lora_dropout
            }

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                **peft_kwargs
            )
            # model = "a"
            model = get_peft_model(model, lora_config)

    elif training_args.finetuning_type == "mylora":
        from src.model.mpeft import PeftModel, TaskType, LoraConfig, get_peft_model
        
        # from llmtuner.model.my_peft import PeftModel, TaskType, LoraConfig, get_peft_model
        logger.info("Fine-tuning method: MyLoRA")
        adapter_to_resume = None

        adapter_name_or_path = model_args.adapter_name_or_path.split(",") if model_args.adapter_name_or_path is not None else None
        
        if adapter_name_or_path is not None:
            model = PeftModel.from_pretrained(model, adapter_name_or_path[0])
        else:
            target_modules = training_args.lora_target.split(',')
                
            peft_kwargs = {
                "r": training_args.lora_rank,
                "target_modules": target_modules,
                "lora_alpha": training_args.lora_alpha,
                "lora_dropout": training_args.lora_dropout
            }
            lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    **peft_kwargs
                )
            model = get_peft_model(model, lora_config)

        for name, param in model.named_parameters():
            if name.find("loranew_") != -1:
                param.requires_grad = True
            elif name.find("lora_") != -1:
                param.requires_grad = False
            # this module should always be frozen because we change the vocabulary
            elif name.find("shared") != -1:
                param.requires_grad = False

    # elif training_args.finetuning_type == "slora":
        
    #     if model_args.load_checkpoint_from:
    #         print("----------Loading Previous Query Projection Layer----------")
    #         model.model.trans_input.load_state_dict(torch.load(model_args.load_checkpoint_from))
    #         print("----------Loading Previous Query Projection Layer Done----------")

    #     adapter_name_or_path = model_args.adapter_name_or_path.split(",") if model_args.adapter_name_or_path is not None else None
        
    #     if adapter_name_or_path is not None:
    #         print(adapter_name_or_path)
    #         print("----------Loading Previous LoRA Weights----------")
    #         for i, path in enumerate(adapter_name_or_path):
    #             lora_A = torch.load(os.path.join(path, "lora_weights_A.pt"))
    #             lora_B = torch.load(os.path.join(path, "lora_weights_B.pt"))
    #             ## Loading LoRA weights for LLaMA-2
    #             for j in range(len(model.model.layers)):
    #                 model.model.layers[j].self_attn.previous_lora_weights_q[i].lora_A.data.copy_(
    #                     lora_A[f"model.layers.{j}.self_attn.lora_q.lora_A"]
    #                 )
    #                 model.model.layers[j].self_attn.previous_lora_weights_q[i].lora_B.data.copy_(
    #                     lora_B[f"model.layers.{j}.self_attn.lora_q.lora_B"]
    #                 )
    #                 model.model.layers[j].self_attn.previous_lora_weights_v[i].lora_A.data.copy_(
    #                     lora_A[f"model.layers.{j}.self_attn.lora_v.lora_A"]
    #                 )
    #                 model.model.layers[j].self_attn.previous_lora_weights_v[i].lora_B.data.copy_(
    #                     lora_B[f"model.layers.{j}.self_attn.lora_v.lora_B"]

    #     for name, param in model.named_parameters():
    #         param.requires_grad = False
    #         if ("lora" in name and "previous_lora_weights" not in name) or "trans_input" in name or "prompt_key" in name:
    #             param.requires_grad = True

    # for n, p in model.named_parameters():
    #     # if p.requires_grad:
    #     # if any([x in n for x in ["router", "A", "z"]]):
    #     print(n, p.size())
    # exit()
    # total_params, params = 0, 0
    # for n, p in model.named_parameters():
    #     # if p.requires_grad:
    #     # if any([x in n for x in ["router", "A", "z"]]):
    #     print(n, p.size())
    #     total_params += p.numel()
    #     params += p.numel()
    # print(params)
        
    return model
