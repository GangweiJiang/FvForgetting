import logging
import random
import string
from transformers.data.data_collator import *

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForNI:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None
    max_target_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    tk_instruct: bool = False
    text_only: bool=False
    

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        model_inputs = []
        sources = []
        for instance in batch:
            
            add_task_name = self.add_task_name
            add_task_definition = self.add_task_definition
            num_pos_examples = self.num_pos_examples
            num_neg_examples = self.num_neg_examples
            add_explanation = self.add_explanation 

            task_input = ""
            # add the input first.
            task_input += "Now complete the following example -\n"
            task_input += f"Input: {instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output:"
            
            task_name = ""
            if add_task_name:
                task_name += instance["Task"] + ". "

            definition = ""
            if add_task_definition:
                if isinstance(instance["Definition"], list):
                    definition = "Definition: " + instance["Definition"][0].strip() # TODO: should we use <Definition>?
                else:
                    definition = "Definition: " + instance["Definition"].strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"
            
            # try to add positive examples.
            pos_examples = []
            for idx, pos_example in enumerate(instance["Positive Examples"][:num_pos_examples]):
                pos_example_str = f" Positive Example {idx+1} -\n"
                pos_example_str += f"Input: {pos_example['input'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                pos_example_str += f" Output: {pos_example['output'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n" 
                if add_explanation and "explanation" in pos_example:
                    pos_example_str += f" Explanation: {pos_example['explanation'].strip()}"
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                pos_example_str += "\n"
                if len(self.tokenizer(definition + " ".join(pos_examples) + pos_example_str + task_input)["input_ids"]) <= self.max_source_length:
                    pos_examples.append(pos_example_str)
                else:
                    break
            
            # try to add negative examples.
            neg_examples = []
            for idx, neg_example in enumerate(instance["Negative Examples"][:num_neg_examples]):
                neg_example_str = f" Negative Example {idx+1} -\n"
                neg_example_str += f"Input: {neg_example['input'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                neg_example_str += f" Output: {neg_example['output'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                if add_explanation and "explanation" in neg_example:
                    neg_example_str += f" Explanation: {neg_example['explanation'].strip()}"
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                neg_example_str += "\n"
                if len(self.tokenizer(definition + " ".join(pos_examples) + " ".join(neg_examples) + neg_example_str + task_input)["input_ids"]) <= self.max_source_length:
                    neg_examples.append(neg_example_str)
                else:
                    break 
            
            source =  self.tokenizer.bos_token + task_name + definition + "".join(pos_examples) + "".join(neg_examples) + task_input
            label = random.choice(instance['Instance']["output"]) + self.tokenizer.eos_token
        
            source_ids = self.tokenizer(source, add_special_tokens=False)["input_ids"]
            target_ids = self.tokenizer(label, add_special_tokens=False)["input_ids"]
            
            source_len, target_len = len(source_ids), len(target_ids)

            if source_len >  self.max_source_length:
                source_ids = source_ids[:self.max_source_length]
            if target_len > self.max_target_length:
                target_ids = target_ids[:self.max_target_length]

            source_mask = [self.label_pad_token_id] * len(source_ids)

            input_ids_wo_label = source_ids
            input_ids = source_ids + target_ids
            labels = source_mask + target_ids

            model_input = {}
            model_input["input_ids"] = input_ids
            model_input["attention_mask"] = [1] * len(input_ids)
            model_input["labels"] = labels
            model_input['input_ids_wo_label'] = input_ids_wo_label
            model_inputs.append(model_input)
        labels = [feature["labels"] for feature in model_inputs] if "labels" in model_inputs[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            max_length_wo_label = max(len(l['input_ids_wo_label']) for l in model_inputs)
            padding_side = self.tokenizer.padding_side
            for feature in model_inputs:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
        
                remainder = [self.tokenizer.pad_token_id] * (max_length_wo_label - len(feature["input_ids_wo_label"]))
                feature["input_ids_wo_label"] = (
                    feature["input_ids_wo_label"] + remainder if padding_side == "right" else remainder + feature["input_ids_wo_label"]
                )
                # feature["attention_mask_wo_label"] = [1] * len(input_ids_wo_label)
    
        model_inputs = self.tokenizer.pad(
            model_inputs,
            padding=True,
            max_length=self.max_source_length+self.max_target_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # for i in range(model_inputs["input_ids"].shape[0]):
        #     print(self.tokenizer.decode(model_inputs["input_ids"][i]))
        #     print("------")
        #     print(self.tokenizer.decode(list(filter(lambda x: x != -100, model_inputs["labels"][i])), skip_special_tokens=False))
        # exit()

        return model_inputs