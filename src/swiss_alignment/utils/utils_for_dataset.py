import copy
import logging
import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizer

hydra_logger = logging.getLogger(__name__)


# Adapted from: https://github.com/allenai/open-instruct/blob/main/open_instruct/dataset_transformation.py
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

    # Input ids
    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
    )
    row[INPUT_IDS_KEY] = tokenizer.apply_chat_template(row[sft_messages_key])
    row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])

    # Prompt/Completion masks
    completion_lenth = len(row[INPUT_IDS_KEY]) - len(row[INPUT_IDS_PROMPT_KEY])
    row[PROMPT_MASK_KEY] = [1] * len(row[INPUT_IDS_PROMPT_KEY]) + [0] * completion_lenth
    row[COMPLETION_MASK_KEY] = [0] * len(row[INPUT_IDS_PROMPT_KEY]) + [
        1
    ] * completion_lenth

    # Labels
    labels = copy.deepcopy(row[INPUT_IDS_KEY])
    row[LABELS_KEY] = labels
    return row


def sft_tokenize_mask_out_prompt(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY,
):
    """mask out the prompt tokens by manipulating labels"""
    row = sft_tokenize(row, tokenizer, sft_messages_key=sft_messages_key)

    # Mask out prompt tokens
    labels = torch.tensor(row[LABELS_KEY])
    prompt_mask = torch.tensor(row[PROMPT_MASK_KEY])
    row[LABELS_KEY] = torch.where(prompt_mask == 1, -100, labels).tolist()
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

    contain_some_labels = sum(row[PROMPT_MASK_KEY]) < len(row[PROMPT_MASK_KEY])
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

        if message["role"] != "assistant":
            chat_template_args = {
                "conversation": messages[: message_idx + 1],
                "tokenize": True,
                "return_tensors": "pt",
                "padding": False,
                "truncation": True,
                "max_length": max_seq_length,
            }

            # Identifying non-assistant prompts
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                chat_template_args["add_generation_prompt"] = True
            else:
                chat_template_args["add_generation_prompt"] = False
            message_end_idx = tokenizer.apply_chat_template(**chat_template_args).shape[
                1
            ]
            prompt_mask[:, message_start_idx:message_end_idx] = 1  # Mark prompt tokens
            if max_seq_length and message_end_idx >= max_seq_length:
                break
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
            completion_mask[
                :, message_start_idx:message_end_idx
            ] = 1  # Mark assistant tokens
            if max_seq_length and message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    row[INPUT_IDS_KEY] = input_ids.flatten()
    row[LABELS_KEY] = labels.flatten()
    row[ATTENTION_MASK_KEY] = attention_mask.flatten()
    row[PROMPT_MASK_KEY] = prompt_mask.flatten()
    row[COMPLETION_MASK_KEY] = completion_mask.flatten()
    return row


def sft_tulu_filter(row: Dict[str, Any], tokenizer: PreTrainedTokenizer):
    return sum(row[PROMPT_MASK_KEY]) < len(row[PROMPT_MASK_KEY])


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
    if len(dc.transform_fn) != len(dc.transform_fn_args):
        raise ValueError(
            f"transform_fn and transform_fn_args must have the same length: {dc.transform_fn=} != {dc.transform_fn_args=}"
        )

    ds = load_dataset_flexible(dc.dataset_name)
    ds = DatasetDict(
        {
            "train": ds[dc.dataset_split_names["train"]],
            **(
                {"eval": ds[dc.dataset_split_names["eval"]]}
                if dc.dataset_split_names["eval"] is not None
                else {}
            ),
        }
    )

    if dc.dataset_subsample["train"] > 0:
        ds["train"] = ds["train"].select(
            range(min(len(ds["train"]), dc.dataset_subsample["train"]))
        )
    if dc.dataset_split_names["eval"] is not None and dc.dataset_subsample["eval"] > 0:
        ds["eval"] = ds["eval"].select(
            range(min(len(ds["eval"]), dc.dataset_subsample["eval"]))
        )

    # beaker specific logic; we may get assigned 15.5 CPU, so we convert it to float then int
    num_proc = int(
        float(os.environ.get("BEAKER_ASSIGNED_CPU_COUNT", multiprocessing.cpu_count()))
    )
    for split in ds.keys():
        for fn_name, fn_args in zip(dc.transform_fn, dc.transform_fn_args):
            fn, fn_type = TRANSFORM_FNS[fn_name]
            # always pass in tokenizer and other args if needed
            fn_kwargs = {"tokenizer": tokenizer}
            fn_kwargs.update(fn_args)

            # perform the transformation
            target_columns = (
                ds[split].column_names
                if dc.target_columns is None
                else dc.target_columns
            )
            if fn_type == "map":
                ds[split] = ds[split].map(
                    fn,
                    fn_kwargs=fn_kwargs,
                    remove_columns=[
                        col
                        for col in ds[split].column_names
                        if col not in target_columns
                    ],
                    num_proc=get_num_proc(
                        len(ds[split]),
                        num_proc,
                        APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU,
                    ),
                )
            elif fn_type == "filter":
                ds[split] = ds[split].filter(
                    fn,
                    fn_kwargs=fn_kwargs,
                    num_proc=get_num_proc(
                        len(ds[split]), num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU
                    ),
                )
            else:
                raise ValueError(f"Unknown transform function type: {fn_type}")

        if len(ds[split]) == 0:
            raise ValueError("No examples left after transformation")

    return ds
