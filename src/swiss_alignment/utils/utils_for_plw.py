import logging
from functools import partial

import numpy as np
from accelerate.logging import get_logger
from accelerate.state import PartialState
from datasets import DatasetDict, Sequence, Value
from transformers.data.data_collator import DataCollatorForLanguageModeling

from swiss_alignment import utils

utils.config.register_resolvers()
acc_state = PartialState()
acc_logger = get_logger(__name__)
hydra_logger = logging.getLogger(__name__)


class PLWDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, pad_to_multiple_of=None):
        super(PLWDataCollator, self).__init__(
            tokenizer=tokenizer, mlm=mlm, pad_to_multiple_of=pad_to_multiple_of
        )

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # Delegate to parent __call__ for standard fields
        standard_fields = [
            {"input_ids": ex["input_ids"], "attention_mask": ex["attention_mask"]}
            for ex in features
        ]
        batch = super(PLWDataCollator, self).__call__(
            standard_fields, return_tensors=return_tensors
        )

        # Custom fields to pad separately
        max_length = batch["input_ids"].shape[1]  # Match length of padded input_ids

        # Pad and tensorize custom fields with framework-appropriate tensors
        for field_name in ["prompt_mask", "completion_mask"]:
            # First create as numpy array (neutral format)
            padded_field = np.zeros((len(features), max_length), dtype=np.int8)
            for i, ex in enumerate(features):
                length = min(len(ex[field_name]), max_length)
                padded_field[i, :length] = ex[field_name][:length]

            # Convert to appropriate tensor type
            if return_tensors == "pt":
                import torch

                batch[field_name] = torch.tensor(padded_field)
            elif return_tensors == "tf":
                import tensorflow as tf

                batch[field_name] = tf.convert_to_tensor(padded_field)
            elif return_tensors == "np":
                batch[field_name] = padded_field

        return batch


# All adapted from: https://github.com/davidsvaughn/prompt-loss-weight/blob/main/run_plw.py
def __tokenize_batch(batch, tokenizer):
    # tokenize and encode text
    tokenized_text = tokenizer(
        batch["text"],
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    data = {k: tokenized_text[k] for k in tokenized_text.keys()}

    # use offset_mappings to make prompt/completion masks (idx marks the start of each completion)
    prompt_masks, completion_masks = [], []
    for offset_mapping, idx in zip(data["offset_mapping"], batch["idx"]):
        prompt_masks.append([1 if o[1] < idx else 0 for o in offset_mapping])
        completion_masks.append([0 if o[1] < idx else 1 for o in offset_mapping])

    data["prompt_mask"] = prompt_masks
    data["completion_mask"] = completion_masks
    del data["offset_mapping"]
    return data


def __tokenize(dataset, tokenizer):
    # Tokenize dataset, remove original columns
    tokenized = dataset.map(
        partial(__tokenize_batch, tokenizer=tokenizer),
        batched=True,
        remove_columns=list(dataset.features),
    )

    # Cast mask columns to int8
    for col in ["prompt_mask", "completion_mask"]:
        tokenized = tokenized.cast_column(col, Sequence(Value("int8")))

    return tokenized


def prepare_dataset(dataset, tokenizer):
    # Helper function to format each sample
    def format_sample(sample):
        # Apply chat template to full conversation
        sample["text"] = tokenizer.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=False
        )
        # Get prompt length for completion index
        prompt_text = tokenizer.apply_chat_template(
            sample["messages"][:1], tokenize=False, add_generation_prompt=True
        )
        sample["idx"] = len(prompt_text)
        return sample

    processed = DatasetDict({})
    for k in dataset.keys():
        split = dataset[k].map(format_sample)
        split = __tokenize(split, tokenizer)
        processed[k] = split

    return processed
