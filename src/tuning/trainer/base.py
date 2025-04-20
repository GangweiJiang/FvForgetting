import os
import json
import torch
import numpy as np
import torch.nn as nn
import pickle
from torch.nn import CrossEntropyLoss
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from transformers import Seq2SeqTrainer
from transformers.trainer import *
from transformers.trainer_pt_utils import nested_numpify, nested_concat
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from dataclasses import dataclass

import tqdm
from copy import deepcopy
import torch.distributed as dist
import torch.nn.functional as F
import math
from transformers.training_args import TrainingArguments

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput

from src.baukit import TraceDict
IGNORE_INDEX=-100
ANSWER_PREFIX=['Output:','Answer:', '答案:', 'summary:', 'A:', 'Assistant:']

import logging
logger = logging.getLogger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        function_vector: str = None,
        mean_activations: str = None,
        head_score: str = None,
        fv_kl= False,
        kl_alpha1= 0.0,
        kl_alpha2= 0.05,
        inter_scale: str = None,
        **kwargs
    ):  
        super().__init__(model, **kwargs)
        
        self.fv_kl = self.args.fv_kl
        self.fv_pr = self.args.fv_pr
        self.kl_alpha1 = self.args.kl_alpha1
        self.kl_alpha2 = self.args.kl_alpha2
        self.pr_alpha = self.args.pr_alpha
        
        self.use_fv = False        
        
        self.function_vector = None
        func_path = self.args.func_path
        if func_path is not None:
            if 'random' in func_path:
                function_vector = torch.load(func_path[len('random'):])
                self.function_vector = torch.sqrt(torch.var(function_vector))*torch.randn(1,self.model.config.hidden_size).to(function_vector.device)+torch.mean(function_vector)        
            else:
                function_vector = torch.load(func_path)
                self.function_vector = function_vector.reshape(1,self.model.config.hidden_size)
            self.use_fv = True
            
        
        if self.args.head_score is not None:
            indirect_effect = torch.load(head_score)
            mean_indirect_effect = - indirect_effect[:,:,:,1].mean(dim=0) # Subset to token class of interest

            h_shape = mean_indirect_effect.shape 
            topk_vals, topk_inds  = torch.topk(mean_indirect_effect.view(-1), k=10, largest=True)
            top_lh = list(zip(*np.unravel_index(topk_inds, h_shape), [round(x.item(),4) for x in topk_vals]))
            self.pos = top_lh[:10]
        else:
            if 'Llama-2-7b' in self.args.model_name_or_path:
                pos = [(14, 1, 0.0391), (11, 2, 0.0225), (9, 25, 0.02), (12, 15, 0.0196), (12, 28, 0.0191), (13, 7, 0.0171), (11, 18, 0.0152), (12, 18, 0.0113), (16, 10, 0.007), (14, 16, 0.007),]
            elif 'Llama-3.1-8B' in self.args.model_name_or_path:
                pos = [(27, 28, 0.1217), (13, 27, 0.1147), (15, 28, 0.1142), (17, 8, 0.0937), (21, 2, 0.0867), (10, 12, 0.0717), (15, 16, 0.067), (15, 2, 0.0668), (15, 1, 0.066), (31, 24, 0.0605),]
            elif 'Mistral-7B' in self.args.model_name_or_path:
                pos = [(14, 31, 0.1186), (26, 29, 0.0876), (12, 4, 0.0791), (12, 7, 0.0709), (30, 4, 0.0674), (30, 9, 0.0601), (22, 30, 0.0588), (14, 19, 0.0564), (11, 10, 0.047), (18, 1, 0.0429), ]

            self.pos = pos[:10]
        
        self.head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        self.layer_hook_names = [f'module.base_model.model.model.layers.{layer}' for layer in range(32)]
        self.head_hook_names = [f'module.base_model.model.model.layers.{layer}.self_attn.o_proj' for layer in range(32)]

        self.cnt = 0

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

        model.train()
        inputs = self._prepare_inputs(inputs)
        inputs.pop("input_ids_wo_label")
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        row_indices = [i for i in range(len(inputs['labels']))]
        depth_indices = [(inputs["labels"][i]+100).nonzero()[0][0]-1 for i in range(inputs["labels"].shape[0])]
        row_indices = torch.tensor(row_indices)
        depth_indices = torch.tensor(depth_indices)

        if self.fv_kl:
            if self.use_fv and inputs['labels'].device != self.function_vector.device:
                self.function_vector=self.function_vector.to(inputs['labels'].device).to(torch.bfloat16)
                # self.function_vector=torch.zeros_like(self.function_vector)
            with torch.no_grad():
                model.module.disable_adapter_layers()
                
                if self.use_fv:
                    intervention_fn = add_function_vector(self.args.edit_layer, self.function_vector, depth_indices, degree=1)
                else:
                    intervention_fn = None

                with TraceDict(model, layers=self.head_hook_names, edit_output=intervention_fn):
                    outputs = model(**inputs, output_attentions=self.fv_pr)  
                    ori_logits = deepcopy(outputs.logits.detach())
                    del outputs


                if self.fv_pr:
                    outputs = model(**inputs, output_attentions=self.fv_pr)  
                    ori_heads = []
                    for i, (l,h,_) in enumerate(self.pos):
                        hidden_state = outputs["attentions"][l]
                        # print(model.module.model.fv_projection.weight[0][0])
                        ori_heads.append(deepcopy(hidden_state[row_indices, \
                            depth_indices][:, self.head_dim*h:self.head_dim*(1+h)].detach()))
                    del outputs
                model.module.enable_adapter_layers()


        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        
        if dist.get_rank()==0 and self.state.global_step%20 == 0:
            print(self.state.global_step, loss.item())

        if self.fv_kl:
            kl1 = self.KD_loss(outputs.logits[row_indices,depth_indices,:], ori_logits[row_indices,depth_indices,:], 2)
            # kl1 = self.KD_loss(outputs.logits[arow_indices,adepth_indices,:], ori_logits[arow_indices,adepth_indices,:], 2)
            dis_loss = (self.kl_alpha1 if kl1 > 0.1 else 0) * kl1

            kl2 = self.KD_loss(outputs.logits, ori_logits, 2)
            dis_loss += (self.kl_alpha2 if kl2 > 0.1 else 0 ) * kl2

            loss += dis_loss
        
        if self.fv_pr:
            pr_loss = 0

            for i, (l, h, _) in enumerate(self.pos):
                # v = self.mean_activations[l,h]
                hidden_state = outputs["attentions"][l]
                
                if self.args.pr_loss_type == "linear":
                    p_item = F.mse_loss(model.module.model.fv_projections[i](hidden_state[row_indices, \
                        depth_indices][:, self.head_dim*h:self.head_dim*(1+h)]), ori_heads[i])
                    pr_loss += (self.pr_alpha if p_item > 0.05 else 0) * p_item
                elif self.args.pr_loss_type == "ind":
                    p_item = F.mse_loss(hidden_state[row_indices, \
                        depth_indices][:, self.head_dim*h:self.head_dim*(1+h)], ori_heads[i])
                    pr_loss += (self.pr_alpha if p_item > 0.01 else 0) * p_item
                    
            loss += pr_loss
        ######################################################
        # print("loss:", loss)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

            
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # print("backward")
            self.accelerator.backward(loss)

        # print(model.module.model.fv_projection.weight.grad[0][0])

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        outputs = model(**inputs, output_attentions=self.fv_pr)  
        
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        if return_outputs:
            return loss, outputs
        else:
            return loss

    def KD_loss(self, new_logits, prev_logits, T):
        kd_loss = F.kl_div(F.log_softmax((prev_logits) / T, dim=-1),
                        F.softmax((new_logits) / T, dim=-1),
                        reduction='batchmean')
        
        return kd_loss
        
    def on_training_begin(self):
        return
    
    def on_training_end(self):
        return

    def _save_checkpoint(self, _, trial, metrics=None):
        """ Don't save base model, optimizer etc.
            but create checkpoint folder (needed for saving adapter) """
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.epoch}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (self.state.best_metric is None or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)):
                self.state.best_metric = metric_value

                self.state.best_model_checkpoint = output_dir

        os.makedirs(output_dir, exist_ok=True)

        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def _pad_tensors_to_target_len(
        self,
        src_tensor: torch.Tensor,
        tgt_tensor: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1]:] = src_tensor # adopt left-padding
        return padded_tensor.contiguous() # in contiguous memory

    def save_predictions(
        self,
        predict_results: "PredictionOutput"
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)
        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate((preds[i][pad_len[0]:], preds[i][:pad_len[0]]), axis=-1) # move pad token to last

        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for label, pred in zip(decoded_labels, decoded_preds):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        
        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped


        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None


        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        metrics = None
        
        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            
            # if is_torch_tpu_available():
            #     xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and self.accelerator.sync_gradients:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        
        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        metrics["global_step"] = self.state.global_step

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        # has_labels = False
        inputs = self._prepare_inputs(inputs)
        self.cnt+=1
        if self.cnt==1:
            print(inputs["input_ids_wo_label"][0])
            print(self.tokenizer.decode(inputs["input_ids_wo_label"][0]))
            # exit()
        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_new_tokens": 128,
            # "synced_gpus": True if self.is_deepspeed_enabled else False,
            "synced_gpus": False,
            "num_beams": 1,
            "temperature": 1.0,
            "repetition_penalty": 1.0,
            "eos_token_id": 2,
            "pad_token_id": 1,

            # 'top_p': 0.9, 
            # 'temperature': 0.1,
            # 'pad_token_id': self.tokenizer.eos_token_id
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)[:,:inputs["input_ids_wo_label"].shape[1]]
        # print(inputs["input_ids_wo_label"].shape, gen_kwargs["attention_mask"].shape)
        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        if inputs.get("input_ids_wo_label", None) is not None:
            generated_tokens = self.model.generate(
                input_ids=inputs["input_ids_wo_label"],
                **gen_kwargs,
            )
        
        else:
            generated_tokens = self.model.generate(
                input_ids=generation_inputs,
                **gen_kwargs,
            )
        
        bs, source_len = inputs['input_ids_wo_label'].shape
        # print(source_len)
        max_length = source_len + gen_kwargs["max_new_tokens"]

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, max_length)

        # with torch.no_grad():
        #     if has_labels:
        #         with self.autocast_smart_context_manager():
        #             outputs = model(**inputs)
        #         if self.label_smoother is not None:
        #             loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
        #         else:
        #             loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
        #     else:
        loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_new_tokens"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_new_tokens"])
        else:
            labels = None

        return (loss, generated_tokens, labels)


@dataclass
class CusDataCollator:

    tokenizer: PreTrainedTokenizerBase
    padding: bool= True
    max_length: int = None
    pad_to_multiple_of: int = None
    label_pad_token_id: int = -100
    return_tensors: str = 'pt'

    def __call__(self, features, return_tensors=None):

        # print("init", features)
        # for feature in features:
        #     print("init", len(feature["input_ids"]), len(feature["attention_mask"]), len(feature["labels"]), len(feature["input_ids_wo_label"]))
        
        features = [{"input_ids": feature["input_ids"], 
                    "attention_mask": feature["attention_mask"], 
                    "labels": feature["labels"], 
                    'input_ids_wo_label': feature['input_ids_wo_label']} for feature in features]

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        max_label_length = max(len(l) for l in labels)
        max_length_wo_label = max(len(l['input_ids_wo_label']) for l in features)
        if self.pad_to_multiple_of is not None:
            max_label_length = (
                (max_label_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of
            )

        padding_side = self.tokenizer.padding_side
        for feature in features:
            remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
            wremainder = [self.tokenizer.pad_token_id] * (max_length_wo_label - len(feature["input_ids_wo_label"]))
            iremainder = [0] * (max_label_length - len(feature["labels"]))
            if isinstance(feature["labels"], list):
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
                if "instruct_indicates" in feature:
                    feature["instruct_indicates"] = (
                        feature["instruct_indicates"] + iremainder if padding_side == "right" else iremainder + feature["instruct_indicates"]
                    )
                if "input_ids_wo_label" in feature:
                    feature["input_ids_wo_label"] = (
                        feature["input_ids_wo_label"] + wremainder if padding_side == "right" else wremainder + feature["input_ids_wo_label"]
                    )
            elif padding_side == "right":
                feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                if "instruct_indicates" in feature:
                    feature["instruct_indicates"] = np.concatenate([feature["instruct_indicates"], iremainder]).astype(np.int64)
                if "input_ids_wo_label" in feature:
                    feature["input_ids_wo_label"] = np.concatenate([feature["input_ids_wo_label"], wremainder]).astype(np.int64)
            else:
                feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
                if "instruct_indicates" in feature:
                    feature["instruct_indicates"] = np.concatenate([iremainder, feature["instruct_indicates"]]).astype(np.int64)
                if "input_ids_wo_label" in feature:
                    feature["input_ids_wo_label"] = np.concatenate([wremainder, feature["input_ids_wo_label"]]).astype(np.int64)
        
        # print("befre", features)
        # for feature in features:
        #     print("before", len(feature["input_ids"]), len(feature["attention_mask"]), len(feature["labels"]), len(feature["input_ids_wo_label"]))
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        # print(features)
        return features



class PeftSavingCallback(TrainerCallback):
    """ Correctly save PEFT model and not full model """
    def __init__(self):
        super().__init__()
        self.curr_save_id=0

    def _save(self, model, folder):
        if folder is None:
            folder = ""
        peft_model_path = os.path.join(folder, "adapter")
        model.save_pretrained(peft_model_path)

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
            control: TrainerControl, **kwargs):
        """ Save final best model adapter """
        # return
        if not args.no_save:
            kwargs['model'].save_pretrained(args.output_dir)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if args.save_step:
            save_step = [20,150,350,1200,2400,4800,8000,16000]
            if self.curr_save_id < len(save_step) and abs(state.global_step-save_step[self.curr_save_id])<10:
                folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{save_step[self.curr_save_id]}")
                print(f"save model at step {save_step[self.curr_save_id]} at {folder}")
                self._save(kwargs['model'], folder)
                self.curr_save_id += 1
        

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState,
            control: TrainerControl, **kwargs):
        """ Save intermediate model adapters in case of interrupted training """
        
        print(f"on epoch {state.epoch} end ")
        # save_epoch = [5.0,10.0,15.0,20.0]
        # if self.curr_save_id < len(save_epoch) and abs(state.epoch-save_epoch[self.curr_save_id])<0.2:
        #     folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{save_epoch[self.curr_save_id]}")
        #     print(f"save model at epoch {state.epoch} at {save_epoch[self.curr_save_id]}")
        #     self._save(kwargs['model'], folder)
        #     self.curr_save_id += 1
        
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.epoch}"
        )       
        self._save(kwargs['model'], checkpoint_folder)

        # peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        # kwargs["model"].save_pretrained(peft_model_path)
        return control


def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)

def add_function_vector(edit_layer, fv_vector, token_n, degree=1):
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
        # print(layer_name)
        current_layer = int(layer_name.split(".")[5])
        if current_layer == edit_layer:
            if isinstance(output, tuple):
                print(output[0].shape)
                for i in range(output[0].shape[0]):
                    output[0][i, token_n[i]: token_n[i]+1] += degree * fv_vector
                    
                    # output[0][:, idx] += degree*fv_vector.to(device)
                return output
            else:
                for i in range(output.shape[0]):
                    output[i, token_n[i]: token_n[i]+1] += degree * fv_vector
                
                return output
        else:
            return output

    return add_act



def skip_instructions(model, predictions_ids, tokenizer, ignore_idx=-100):
    predictions_ids = np.where(predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)

    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    final_predictions = []

    for pred in predictions:
        flag = 1
        for answer_pre in ANSWER_PREFIX:
            if answer_pre in pred:
                splits = pred.split(answer_pre)
                final_predictions.append(splits[-1].strip())
                flag = 0
                break
        if flag == 1:
            final_predictions.append('')

    return final_predictions
