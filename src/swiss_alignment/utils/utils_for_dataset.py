import copy
import logging
from functools import partial

import numpy as np
from accelerate.logging import get_logger
from accelerate.state import PartialState
from datasets import DatasetDict, Sequence, Value
from transformers import PreTrainedTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from swiss_alignment import utils

utils.config.register_resolvers()
acc_state = PartialState()
acc_logger = get_logger(__name__)
hydra_logger = logging.getLogger(__name__)


# All adapted from: https://github.com/davidsvaughn/prompt-loss-weight/blob/main/run_plw.py
def __tokenize_batch(batch, tokenizer):
    # tokenize and encode text
    tokenized_text = tokenizer(
        batch["text"],
        return_tensors="pt",
        padding=False,
        truncation=True,
        # max_length=max_seq_length,
        add_generation_prompt=False,
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
    # def format_sample(sample):
    #     # Apply chat template to full conversation
    #     sample["text"] = tokenizer.apply_chat_template(
    #         sample["messages"], tokenize=False, add_generation_prompt=False
    #     )
    #     # Get prompt length for completion index
    #     prompt_text = tokenizer.apply_chat_template(
    #         sample["messages"][:1], tokenize=False, add_generation_prompt=True
    #     )
    #     sample["idx"] = len(prompt_text)
    #     return sample

    processed = DatasetDict({})
    for k in dataset.keys():
        # split = dataset[k].map(format_sample)
        split = dataset[k].map(
            sft_tulu_tokenize_and_truncate, fn_kwargs={"tokenizer": tokenizer}
        )
        # split = __tokenize(split, tokenizer)
        processed[k] = split

    return processed


import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, load_from_disk

# ------------------------------------------------------------
# Dataset Transformation
# SFT dataset
DEFAULT_SFT_MESSAGES_KEY = "messages"
INPUT_IDS_KEY = "input_ids"
INPUT_IDS_PROMPT_KEY = "input_ids_prompt"
ATTENTION_MASK_KEY = "attention_mask"
PROMPT_MASK_KEY = "prompt_mask"
COMPLETION_MASK_KEY = "completion_mask"
LABELS_KEY = "labels"
TOKENIZED_SFT_DATASET_KEYS = [
    INPUT_IDS_KEY,
    ATTENTION_MASK_KEY,
    LABELS_KEY,
]


def sft_tokenize(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY,
):
    if len(row[sft_messages_key]) == 1:
        prompt = row[sft_messages_key]
    else:
        prompt = row[sft_messages_key][:-1]

    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
    )
    row[INPUT_IDS_KEY] = tokenizer.apply_chat_template(row[sft_messages_key])
    row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
    labels = copy.deepcopy(row[INPUT_IDS_KEY])
    row[LABELS_KEY] = labels
    return row


def sft_tokenize_mask_out_prompt(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY,
):
    """mask out the prompt tokens by manipulating labels"""
    if len(row[sft_messages_key]) == 1:
        prompt = row[sft_messages_key]
    else:
        prompt = row[sft_messages_key][:-1]

    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
    )
    row[INPUT_IDS_KEY] = tokenizer.apply_chat_template(row[sft_messages_key])
    row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
    labels = copy.deepcopy(row[INPUT_IDS_KEY])
    labels[: len(row[INPUT_IDS_PROMPT_KEY])] = [-100] * len(row[INPUT_IDS_PROMPT_KEY])
    row[LABELS_KEY] = labels
    return row


def sft_filter(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_prompt_token_length: Optional[int] = None,
    max_token_length: Optional[int] = None,
    need_contain_labels: bool = True,
):
    max_prompt_token_length_ok = True
    if max_prompt_token_length is not None:
        max_prompt_token_length_ok = (
            len(row[INPUT_IDS_PROMPT_KEY]) <= max_prompt_token_length
        )

    max_token_length_ok = True
    if max_token_length is not None:
        max_token_length_ok = len(row[INPUT_IDS_KEY]) <= max_token_length

    contain_some_labels = any(x != -100 for x in row[LABELS_KEY])
    return (
        max_prompt_token_length_ok
        and max_token_length_ok
        and (contain_some_labels or not need_contain_labels)
    )


def sft_tulu_tokenize_and_truncate(
    row: Dict[str, Any], tokenizer: PreTrainedTokenizer, max_seq_length
):
    """taken directly from https://github.com/allenai/open-instruct/blob/ba11286e5b9eb00d4ce5b40ef4cac1389888416a/open_instruct/finetune.py#L385"""
    messages = row["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )
    labels = input_ids.clone()
    prompt_mask = torch.zeros_like(
        input_ids
    )  # Initialize prompt mask (1 for prompt tokens)
    completion_mask = torch.zeros_like(
        input_ids
    )  # Initialize completion mask (1 for assistant tokens)

    # mask the non-assistant part for avoiding loss and set masks
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # we calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # calculate the end index of this non-assistant message
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # set the label to -100 for the non-assistant part and update masks
            labels[:, message_start_idx:message_end_idx] = -100
            prompt_mask[:, message_start_idx:message_end_idx] = 1  # Mark prompt tokens
            if max_seq_length and message_end_idx >= max_seq_length:
                break
        else:
            # For assistant messages, calculate start and end indices
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            message_end_idx = tokenizer.apply_chat_template(
                conversation=messages[: message_idx + 1],
                tokenize=True,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_seq_length,
                add_generation_prompt=False,
            ).shape[1]
            completion_mask[
                :, message_start_idx:message_end_idx
            ] = 1  # Mark assistant tokens

    attention_mask = torch.ones_like(input_ids)
    row[INPUT_IDS_KEY] = input_ids.flatten()
    row[LABELS_KEY] = labels.flatten()
    row[ATTENTION_MASK_KEY] = attention_mask.flatten()
    row[PROMPT_MASK_KEY] = prompt_mask.flatten()
    row[COMPLETION_MASK_KEY] = completion_mask.flatten()
    return row


def sft_tulu_filter(row: Dict[str, Any], tokenizer: PreTrainedTokenizer):
    return any(x != -100 for x in row[LABELS_KEY])


TRANSFORM_FNS = {
    "sft_tokenize": (sft_tokenize, "map"),
    "sft_tokenize_mask_out_prompt": (sft_tokenize_mask_out_prompt, "map"),
    "sft_filter": (sft_filter, "filter"),
    "sft_tulu_tokenize_and_truncate": (sft_tulu_tokenize_and_truncate, "map"),
    "sft_tulu_filter": (sft_tulu_filter, "filter"),
}


# ------------------------------------------------------------
# Dataset Configuration and Caching
@dataclass
class DatasetConfig:
    dataset_name: str
    dataset_path: str
    dataset_split_names: Dict[str, str]
    dataset_subsample: Optional[Dict[str, int]] = None
    transform_fn: List[str] = field(default_factory=list)
    transform_fn_args: List[Dict[str, Any]] = field(default_factory=list)
    target_columns: Optional[List[str]] = None


def load_dataset_flexible(dataset_identifier):
    # Check if dataset_identifier is a valid local path
    if os.path.exists(dataset_identifier):
        hydra_logger.info(f"Loading dataset from local path: {dataset_identifier}")
        return load_from_disk(dataset_identifier)
    else:
        hydra_logger.info(
            f"Loading dataset from Hugging Face Hub: {dataset_identifier}"
        )
        return load_dataset(dataset_identifier)


def save_dataset_flexible(dataset, output_path):
    # Check if output_path looks like a local path or HF Hub repo
    if os.path.isabs(output_path) or os.path.relpath(
        output_path, os.getcwd()
    ).startswith("."):
        dataset.save_to_disk(output_path)
        hydra_logger.info(f"Dataset saved to local directory: {output_path}")
    else:
        dataset.push_to_hub(output_path)
        hydra_logger.info(f"Dataset pushed to Hugging Face Hub: {output_path}")


# Performance tuning. Some rough numbers:
APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU = 400
FILTER_EXAMPLE_PER_SECOND_PER_CPU = 1130


def get_num_proc(
    dataset_len: int, num_available_cpus: int, example_per_second_per_cpu
) -> int:
    num_required_cpus = max(1, dataset_len // example_per_second_per_cpu)
    return min(num_required_cpus, num_available_cpus)


def get_dataset(dc: DatasetConfig, tokenizer):
    assert len(dc.transform_fn) == len(
        dc.transform_fn_args
    ), f"transform_fn and transform_fn_args must have the same length: {dc.transform_fn=} != {dc.transform_fn_args=}"

    ds = load_dataset_flexible(dc.dataset_path)
    ds = DatasetDict(
        {
            "train": ds[dc.dataset_split_names["train"]],
            "eval": ds[dc.dataset_split_names["eval"]],
        }
    )

    if dc.dataset_subsample["train"] > 0:
        ds["train"] = ds["train"].select(
            range(min(len(ds["train"]), dc.dataset_subsample["train"]))
        )
    if dc.dataset_subsample["eval"] > 0:
        ds["eval"] = ds["eval"].select(
            range(min(len(ds["eval"]), dc.dataset_subsample["eval"]))
        )

    # beaker specific logic; we may get assigned 15.5 CPU, so we convert it to float then int
    num_proc = int(
        float(os.environ.get("BEAKER_ASSIGNED_CPU_COUNT", multiprocessing.cpu_count()))
    )
    for fn_name, fn_args in zip(dc.transform_fn, dc.transform_fn_args):
        fn, fn_type = TRANSFORM_FNS[fn_name]
        # always pass in tokenizer and other args if needed
        fn_kwargs = {"tokenizer": tokenizer}
        fn_kwargs.update(fn_args)

        # perform the transformation
        target_columns = (
            ds.column_names if dc.target_columns is None else dc.target_columns
        )
        if fn_type == "map":
            ds = ds.map(
                fn,
                fn_kwargs=fn_kwargs,
                remove_columns=[
                    col for col in ds.column_names if col not in target_columns
                ],
                num_proc=get_num_proc(
                    len(ds), num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU
                ),
            )
        elif fn_type == "filter":
            ds = ds.filter(
                fn,
                fn_kwargs=fn_kwargs,
                num_proc=get_num_proc(
                    len(ds), num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU
                ),
            )
        else:
            raise ValueError(f"Unknown transform function type: {fn_type}")

    if len(ds) == 0:
        raise ValueError("No examples left after transformation")
    return ds
