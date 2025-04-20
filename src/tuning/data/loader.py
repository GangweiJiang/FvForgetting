# from src.tuning.data.preprocess import *
import os
from hashlib import md5
from datasets import load_dataset, load_from_disk
from typing import TYPE_CHECKING, Any, Dict, List, Union
from datasets import DatasetDict, concatenate_datasets


def gen_ni_cache_path(cache_dir, data_args):
    hash_str = data_args.data_dir + data_args.task_name + \
               str(data_args.max_num_instances_per_task) + str(data_args.max_num_instances_per_eval_task)
    print(hash_str)
    hash_obj = md5(hash_str.encode("utf-8"))
    hash_id = hash_obj.hexdigest()
    cache_path = os.path.join(cache_dir, str(hash_id))

    return cache_path

def get_ni_dataset(model_args, data_args, training_args):

    data_cache_dir = gen_ni_cache_path(model_args.cache_data_dir, data_args)
    
    print(data_args.data_dir,data_args.task_config_dir)

    raw_datasets = load_dataset(
        "./src/tuning/data/ni_dataset.py",
        data_dir=data_args.data_dir,
        task_name=data_args.task_name,
        cache_dir=data_cache_dir,  # for debug, change dataset size, otherwise open it
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        trust_remote_code=True
    )

    raw_datasets.cleanup_cache_files()
    
    return raw_datasets



def convert_format_trace(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    # convert dataset from sharegpt format to alpaca format
    outputs = {"prompt": examples["prompt"], 
               "query": ["" for _ in examples["answer"]], 
               "response": examples["answer"]}
    return outputs


def split_dataset(
    raw_datasets,
    data_args,
    training_args
):
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            datasize = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(datasize))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    return {"train_dataset": train_dataset if training_args.do_train else None,
            "eval_dataset": eval_dataset if training_args.do_eval else None}, predict_dataset if training_args.do_predict else None
