import transformers
from transformers import (
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    set_seed, )

import os
import pickle
import logging
import datasets
import sys
import json
import time
import numpy as np
import torch.distributed as dist

sys.path.append("/disk/jianggangwei/fv_guided_traning")

from src.tuning.model.hparam import ModelArguments, DataTrainingArguments, CustomTrainingArguments
from src.tuning.model.loader import load_model_and_tokenizer
from src.tuning.data.loader import *
from src.tuning.data.ni_collator import DataCollatorForNI
from src.tuning.trainer.base import PeftSavingCallback, skip_instructions, CusDataCollator

os.environ['WANDB_DISABLED'] = "True"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger(__name__)

def run():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.model_name_or_path = model_args.model_name_or_path
    # Setup logging

    if dist.get_rank() == 0:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        log_level = logging.INFO
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
        log_level = logging.ERROR
    logger.setLevel(log_level)
    # logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    print(model_args.adapter_name_or_path)
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)
    
    dataset = get_ni_dataset(model_args, data_args, training_args)
    
    # set data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForNI(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        # add_task_name=data_args.add_task_name,
    )
    

    training_args.remove_unused_columns = False

    trainer_kargs, predict_dataset = split_dataset(dataset, data_args, training_args)
    
    # if training_args.predict_with_generate and "superni" in data_args.data_dir:
    if training_args.predict_with_generate:
        from src.tuning.utils.compute_metrics import compute_metrics, compute_grouped_metrics
        def compute_ni_metrics(dataset, preds, save_prefix=None):
            # preds=np.where(preds!=-100, preds, tokenizer.pad_token_id)
            # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

            decoded_preds = skip_instructions(model, preds, tokenizer)
            references = [e["Instance"]["output"] for e in dataset]
            result = compute_metrics(predictions=decoded_preds, references=references)
            result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=dataset["Task"])
            result.update(result_per_task)
            # categories = ["_".join(it[0].lower().split()) for it in dataset["Categories"]]
            # result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=categories)
            # result.update(result_per_category)
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            if save_prefix is not None:
                with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                    for example, pred in zip(dataset, decoded_preds):
                        fout.write(json.dumps({
                            "Task": example["Task"],
                            "Definition": example["Definition"],
                            "Instance": example["Instance"],
                            "Prediction": pred
                        }) + "\n")
            return result
        


    callbacks = [PeftSavingCallback()]
    if training_args.algo == "naive":
        from src.tuning.trainer.base import CustomSeq2SeqTrainer
    elif training_args.algo == "olora":
        from src.tuning.trainer.olora import ContinualLearnSeq2SeqTrainer, OursCallback
        CustomSeq2SeqTrainer = ContinualLearnSeq2SeqTrainer
        callbacks=[OursCallback()]
    
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **trainer_kargs
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        if "13b" in model_args.model_name_or_path or  "70b" in model_args.model_name_or_path:
            trainer.save_model()
            # trainer.log_metrics("train", train_result.metrics)
        if not training_args.no_save:
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()

    # Predict
    if training_args.do_predict:
        logger.info("do predict!!!")
        max_new_tokens = (
            training_args.generation_max_length
            if training_args.generation_max_length is not None
            else data_args.max_target_length
        )
        num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
        repetition_penalty = data_args.repetition_penalty

        predict_results = trainer.predict(
            predict_dataset,
            # batch_size=1,
            metric_key_prefix="predict",
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id
        )

        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log(metrics)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    run()
