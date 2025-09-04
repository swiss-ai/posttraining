import copy
import logging
import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from accelerate import PartialState
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from transformers import PreTrainedTokenizer

hydra_logger = logging.getLogger(__name__)


# Adapted from: https://github.com/allenai/open-instruct/blob/main/open_instruct/utils.py
# ------------------------------------------------------------
# Dataset utilities
def __is_openai_format(messages) -> bool:
    """
    Check if the input messages are in OpenAI format.
    Args:
        messages (`Any`):
            Messages to check.
    Returns:
        `bool`: Whether the messages are in OpenAI format.
    """
    if isinstance(messages, list) and all(
        isinstance(message, dict) for message in messages
    ):
        return all("role" in message and "content" in message for message in messages)
    return False


# functions for handling different formats of messages
def __convert_alpaca_gpt4_to_messages(example):
    """
    Convert an instruction in inst-output to a list of messages.
    e.g. vicgalle/alpaca-gpt4"""
    messages = [
        {
            "role": "user",
            "content": (
                "Below is an instruction that describes a task, paired with an input that provides "
                "further context. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                "### Response:"
            ),
        },
        {"role": "assistant", "content": example["output"]},
    ]
    example["messages"] = messages
    return example


def __convert_codefeedback_single_turn_to_messages(example):
    """
    Convert a query-answer pair to a list of messages.
    e.g. m-a-p/CodeFeedback-Filtered-Instruction"""
    messages = [
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    example["messages"] = messages
    return example


def __convert_metamath_qa_to_messages(example):
    """
    Convert a query-response pair to a list of messages.
    e.g. meta-math/MetaMathQA"""
    messages = [
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": example["response"]},
    ]
    example["messages"] = messages
    return example


def __convert_code_alpaca_to_messages(example):
    """
    Convert a prompt-completion pair to a list of messages.
    e.g. HuggingFaceH4/CodeAlpaca_20K"""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    example["messages"] = messages
    return example


def __convert_aya_to_messages(example):
    """
    Convert a inputs-targets pair to a list of messages.
    e.g. CohereLabs/aya_dataset"""
    messages = [
        {"role": "user", "content": example["inputs"]},
        {"role": "assistant", "content": example["targets"]},
    ]
    example["messages"] = messages
    return example


def __convert_open_orca_to_messages(example):
    """
    Convert a question-response pair to a list of messages.
    e.g. Open-Orca/OpenOrca"""
    messages = [
        {"role": "system", "content": example["system_prompt"]},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["response"]},
    ]
    example["messages"] = messages
    return example


def __conversations_to_messages(example):
    """
    Convert from conversations format to messages.

    E.g. change "from": "user" to "role": "user"
        and "value" to "content"
        and "gpt" to "assistant"

    WizardLMTeam/WizardLM_evol_instruct_V2_196k
    """
    name_mapping = {
        "gpt": "assistant",
        "Assistant": "assistant",
        "assistant": "assistant",
        "user": "user",
        "User": "user",
        "human": "user",
    }
    messages = [
        {"role": name_mapping[conv["from"]], "content": conv["value"]}
        for conv in example["conversations"]
    ]
    example["messages"] = messages
    return example


def __convert_rejection_samples_to_messages(example):
    """
    Convert a rejection sampling dataset to messages.
    """
    example["messages"] = example["chosen"]
    return example


def get_mix_datasets(
    dataset_mixer: List[dict],
    columns_to_keep: Optional[List[str]] = None,
    need_columns: Optional[List[str]] = None,
    keep_ids: bool = False,
    shuffle: bool = True,
    save_data_dir: Optional[str] = None,
    seed: Optional[int] = 42,
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (List[dict]):
            Dictionary or list containing the dataset names and their training proportions.
            By default, all test proportions are 1. Lists are formatted as
            `key1 value1 key2 value2 ...` If a list is passed in, it will be converted to a dictionary.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        need_columns (Optional[List[str]], *optional*, defaults to `None`):
            Column names that are required to be in the dataset.
            Quick debugging when mixing heterogeneous datasets.
        keep_ids (`bool`, *optional*, defaults to `False`):
            Whether to keep ids for training that are added during mixing.
            Used primarily in mix_data.py for saving, or the saved dataset has IDs already.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
        save_data_dir (Optional[str], *optional*, defaults to `None`):
            Optional directory to save training/test mixes on.
        seed (Optional[int], *optional*, defaults to `42`):
            Defines seed used for shuffling.
    """
    # assert valid subsample_factor and duplication_factor
    for ds_config in dataset_mixer:
        assert (
            ds_config.get("subsample_factor", 1) >= 0
        ), "subsample_factor cannot be negative"
        assert (
            ds_config.get("duplication_factor", 1) > 0
        ), "duplication_factor cannot be negative or zero"

    if save_data_dir:
        print(f"Saving mixed dataset to {save_data_dir}")

    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_eval_datasets = []
    columns_to_keep = [] if columns_to_keep is None else columns_to_keep
    for ds_config in dataset_mixer:
        ds = load_dataset_flexible(ds_config["dataset_path"])
        for split in ds.keys():
            # only include splits specified in the config file
            if (
                split not in ds_config["train_splits"]
                and split not in ds_config["eval_splits"]
            ):
                continue

            # assert that needed columns are present
            if need_columns:
                if not all(col in ds[split].column_names for col in need_columns):
                    raise ValueError(
                        f"Needed column {need_columns} not found in dataset {ds[split].column_names}."
                    )

            # map datasets to conversational format
            if (
                "instruction" in ds[split].column_names
                and "output" in ds[split].column_names
                and "messages" not in ds[split].column_names
            ):
                ds[split] = ds[split].map(
                    __convert_alpaca_gpt4_to_messages, num_proc=50
                )
            elif (
                "prompt" in ds[split].column_names
                and "completion" in ds[split].column_names
                and "messages" not in ds[split].column_names
            ):
                ds[split] = ds[split].map(
                    __convert_code_alpaca_to_messages, num_proc=50
                )
            elif (
                "inputs" in ds[split].column_names
                and "targets" in ds[split].column_names
                and "messages" not in ds[split].column_names
            ):
                ds[split] = ds[split].map(__convert_aya_to_messages, num_proc=50)
            elif (
                "conversations" in ds[split].column_names
                and "messages" not in ds[split].column_names
            ):
                ds[split] = ds[split].map(__conversations_to_messages, num_proc=50)
            elif (
                "question" in ds[split].column_names
                and "response" in ds[split].column_names
                and "messages" not in ds[split].column_names
            ):
                ds[split] = ds[split].map(__convert_open_orca_to_messages, num_proc=50)
            elif (
                "query" in ds[split].column_names
                and "answer" in ds[split].column_names
                and "messages" not in ds[split].column_names
            ):
                ds[split] = ds[split].map(
                    __convert_codefeedback_single_turn_to_messages, num_proc=50
                )
            elif (
                "query" in ds[split].column_names
                and "response" in ds[split].column_names
                and "messages" not in ds[split].column_names
            ):
                ds[split] = ds[split].map(
                    __convert_metamath_qa_to_messages, num_proc=50
                )
            elif (
                "chosen" in ds[split].column_names
                and "rejected" in ds[split].column_names
                and "reference_completion" in ds[split].column_names
                and "messages" not in ds[split].column_names
            ):
                ds[split] = ds[split].map(
                    __convert_rejection_samples_to_messages, num_proc=50
                )

            # add id is not present in dataset split
            if "id" not in ds[split].column_names:
                ds[split] = ds[split].add_column(
                    "id",
                    [f"{ds_config['dataset_name']}_{i}" for i in range(len(ds[split]))],
                )

            # keep track of dataset source if requested
            if "source" not in ds[split].column_names:
                ds[split] = ds[split].add_column(
                    "source", [ds_config["dataset_name"]] * len(ds[split])
                )

            # remove redundant columns (to avoid schema conflicts on load)
            ds[split] = ds[split].remove_columns(
                [
                    col
                    for col in ds[split].column_names
                    if col not in (columns_to_keep + ["id", "source"])
                ]
            )

            # for cols in columns_to_keep, if not present, add "None" to the column
            for col in columns_to_keep:
                if col not in ds[split].column_names:
                    ds[split] = ds[split].add_column(col, [None] * len(ds[split]))

            # save dataset split to train/eval split
            if split in ds_config["train_splits"]:
                # subsample the train set only
                if ds_config.get("subsample_factor", 1) != 1:
                    num_samples = (
                        min(int(ds_config["subsample_factor"]), len(ds[split]))
                        if ds_config["subsample_factor"] > 1
                        else int(ds_config["subsample_factor"] * len(ds[split]))
                    )
                    if shuffle:
                        ds[split] = ds[split].select(
                            np.random.choice(
                                len(ds[split]), size=num_samples, replace=False
                            )
                        )
                    else:
                        ds[split] = ds[split].select(range(num_samples))

                # duplicate dataset splits (e.g. used for hardcoded prompts)
                if ds_config.get("duplication_factor", 1) > 1:
                    ds[split] = concatenate_datasets(
                        [ds[split]] * ds_config["duplication_factor"]
                    )

                if shuffle:
                    ds[split] = ds[split].shuffle(seed=seed)

                # save augmented split
                raw_train_datasets.append(ds[split])
            elif split in ds_config["eval_splits"]:
                raw_eval_datasets.append(ds[split])
            else:
                raise ValueError(
                    f"Split type {split} not recognized as one of test or train."
                )

    if len(raw_eval_datasets) == 0 and len(raw_train_datasets) == 0:
        raise ValueError("No datasets loaded.")
    target_features = (raw_train_datasets or raw_eval_datasets)[0].features

    if len(raw_train_datasets) > 0:
        raw_datasets["train"] = concatenate_datasets(
            [dataset.cast(target_features) for dataset in raw_train_datasets]
        )
    if len(raw_eval_datasets) > 0:
        raw_datasets["test"] = concatenate_datasets(
            [dataset.cast(target_features) for dataset in raw_eval_datasets]
        )

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized."
            "Check the dataset has been correctly formatted."
        )

    # optional save
    if save_data_dir:
        save_dataset_flexible(raw_datasets, save_data_dir)

    if not keep_ids:
        # remove id column
        if len(raw_train_datasets) > 0:
            if "id" in raw_datasets["train"].column_names:
                raw_datasets["train"] = raw_datasets["train"].remove_columns("id")
        if len(raw_eval_datasets) > 0:
            if "id" in raw_datasets["test"].column_names:
                raw_datasets["test"] = raw_datasets["test"].remove_columns("id")

    return raw_datasets


# Adapted from: https://github.com/allenai/open-instruct/blob/main/open_instruct/dataset_transformation.py
# ------------------------------------------------------------
# Dataset Transformation
# SFT dataset
DEFAULT_SFT_MESSAGES_KEY = "messages"
MESSAGES_ROLE_KEY = "role"
MESSAGES_CONTENT = "content"
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


SYSTEM_TOKEN = 61
END_SYSTEM_TOKEN = 62
DEVELOPER_TOKEN = 63
END_DEVELOPER_TOKEN = 64
USER_TOKEN = 65
END_USER_TOKEN = 66
ASSISTANT_TOKEN = 67
END_ASSISTANT_TOKEN = 68
INNER_TOKEN = 69
OUTER_TOKEN = 70
TOOL_CALLS_TOKEN = 71
END_TOOL_CALLS_TOKEN = 72


def sft_tokenize(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY,
):
    """Tokenize the messages and create the masks to distinguish the generated tokens from the others"""
    sft_messages = row[sft_messages_key]

    # We first get through the messages to get the lengths of the tool outputs
    tool_outputs_lengths = []
    for message in sft_messages:
        if message["role"] == "assistant":
            for block in message["content"]["blocks"]:
                if block["type"] == "tool_outputs":
                    tool_outputs = block["outputs"]

                    # We format the tool outputs as it is formatted in the chat template
                    tool_outputs_str = f"[{', '.join([tool_output["output"] for tool_output in tool_outputs])}]"
                    tool_outputs_lengths.append(len(tokenizer.encode(tool_outputs_str, add_special_tokens=False)))

    input_ids = np.reshape(tokenizer.apply_chat_template(row[sft_messages_key], return_tensors="np"), -1)

    # We use cumsum to get the different turns
    # Then we subtract to remove the first token of each turn because we don't want to train on it
    start_assistant = np.cumsum(input_ids == ASSISTANT_TOKEN, axis=0) - (input_ids == ASSISTANT_TOKEN).astype(np.int32)
    end_assistant = np.cumsum(input_ids == END_ASSISTANT_TOKEN, axis=0) - (input_ids == END_ASSISTANT_TOKEN).astype(np.int32)

    # The mask is 1 if the token is not an assistant token and 0 otherwise
    mask = (start_assistant == end_assistant)

    # We are searching the end of the tool calls (or the start of tool outputs) in the assistant tokens
    end_tool_calls = (start_assistant != end_assistant) & (input_ids == END_TOOL_CALLS_TOKEN)

    start_tool_output_indices = np.arange(stop=input_ids.shape[0])[end_tool_calls] + 1
    for i, tol in zip(start_tool_output_indices, tool_outputs_lengths):
        mask[i:i+tol] = 1

    row[INPUT_IDS_KEY] = input_ids.tolist()
    row[LABELS_KEY] = input_ids.copy().tolist()
    row[ATTENTION_MASK_KEY] = np.ones_like(input_ids).tolist()
    row[PROMPT_MASK_KEY] = mask.astype(np.int32).tolist()
    row[COMPLETION_MASK_KEY] = np.logical_not(mask).astype(np.int32).tolist()
    
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


def sft_tulu_tokenize_and_truncate(
    row: Dict[str, Any], tokenizer: PreTrainedTokenizer, max_seq_length: int
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
            chat_template_args = {
                "conversation": messages[:message_idx],
                "tokenize": True,
                "return_tensors": "pt",
                "padding": False,
                "truncation": True,
                "max_length": max_seq_length,
            }
            if message_idx > 0 and messages[message_idx]["role"] == "assistant":
                # required so the start idx for assistance doesn't include the generation prompt
                chat_template_args["add_generation_prompt"] = True
            else:
                chat_template_args["add_generation_prompt"] = False

            message_start_idx = tokenizer.apply_chat_template(
                **chat_template_args
            ).shape[1]

        chat_template_args = {
            "conversation": messages[: message_idx + 1],
            "tokenize": True,
            "return_tensors": "pt",
            "padding": False,
            "truncation": True,
            "max_length": max_seq_length,
        }
        if message["role"] != "assistant":
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
            chat_template_args["add_generation_prompt"] = False
            message_end_idx = tokenizer.apply_chat_template(**chat_template_args).shape[
                1
            ]
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


def sft_filter_has_assistant_tokens(
    row: Dict[str, Any], tokenizer: PreTrainedTokenizer
):
    return sum(row[PROMPT_MASK_KEY]) < len(row[PROMPT_MASK_KEY])


def sft_filter_by_token_lengths(
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

    contain_some_labels = sft_filter_has_assistant_tokens(row, tokenizer)
    return (
        max_prompt_token_length_ok
        and max_token_length_ok
        and (contain_some_labels or not need_contain_labels)
    )


def sft_filter_non_alternating_roles(
    row: Dict[str, Any], tokenizer: PreTrainedTokenizer
):
    # Conversation roles must alternate (system/)user/assistant/user/assistant/...
    messages = row[DEFAULT_SFT_MESSAGES_KEY]
    start_index = 1 if messages[0][MESSAGES_ROLE_KEY] == "system" else 0
    for i, message in enumerate(messages[start_index:]):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if message[MESSAGES_ROLE_KEY] != expected_role:
            return False
    return True


TRANSFORM_FNS = {
    "sft_tokenize": (sft_tokenize, "map"),
    "sft_tokenize_mask_out_prompt": (sft_tokenize_mask_out_prompt, "map"),
    "sft_tulu_tokenize_and_truncate": (sft_tulu_tokenize_and_truncate, "map"),
    "sft_filter_has_assistant_tokens": (sft_filter_has_assistant_tokens, "filter"),
    "sft_filter_by_token_lengths": (sft_filter_by_token_lengths, "filter"),
    "sft_filter_non_alternating_roles": (sft_filter_non_alternating_roles, "filter"),
}


# ------------------------------------------------------------
# Dataset Configuration and Caching
@dataclass
class DatasetConfig:
    dataset_name: str
    dataset_split_names: Dict[str, str]
    debug_oom: Optional[bool] = False
    dataset_subsample: Optional[Dict[str, int]] = None
    transform_fn: List[str] = field(default_factory=list)
    transform_fn_args: List[Dict[str, Any]] = field(default_factory=list)
    target_columns: Optional[List[str]] = None
    shuffle: Optional[bool] = False
    seed: int = 42


def load_dataset_flexible(dataset_identifier):
    # Check if dataset_identifier is a valid local path
    if os.path.exists(dataset_identifier):
        if dataset_identifier.endswith(".json") or dataset_identifier.endswith(
            ".jsonl"
        ):
            hydra_logger.info(
                f"Loading JSON dataset from local path: {dataset_identifier}"
            )
            return load_dataset("json", data_files=dataset_identifier)
        else:
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
    dataset_len: int, num_available_cpus: int, example_per_second_per_cpu: int
) -> int:
    num_required_cpus = max(1, dataset_len // example_per_second_per_cpu)
    return min(num_required_cpus, num_available_cpus)


def get_dataset_sft(
    dc: DatasetConfig, tokenizer: PreTrainedTokenizer, acc_state: PartialState
):
    if len(dc.transform_fn) != len(dc.transform_fn_args):
        raise ValueError(
            f"transform_fn and transform_fn_args must have the same length: {dc.transform_fn=} != {dc.transform_fn_args=}"
        )

    with acc_state.main_process_first():
        ds = load_dataset_flexible(dc.dataset_name)

        if isinstance(ds, DatasetDict):
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
        elif isinstance(ds, Dataset):
            # Convert Dataset to DatasetDict with train split
            ds = DatasetDict(
                {
                    "train": ds,
                }
            )

        # beaker specific logic; we may get assigned 15.5 CPU, so we convert it to float then int
        num_proc = int(multiprocessing.cpu_count())
        for split in ds.keys():
            if dc.debug_oom:

                def add_debug_max_len(row):
                    return {
                        "debug_max_len": len(
                            tokenizer.apply_chat_template(
                                row["messages"], tokenize=True
                            )
                        )
                    }

                ds[split] = (
                    ds[split]
                    .map(
                        add_debug_max_len,
                        num_proc=get_num_proc(
                            len(ds[split]),
                            num_proc,
                            APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU,
                        ),
                    )
                    .sort("debug_max_len", reverse=True)
                )
            else:
                if dc.shuffle:
                    ds[split] = ds[split].shuffle(seed=dc.seed)

            if split in dc.dataset_subsample and dc.dataset_subsample[split] > 0:
                ds[split] = ds[split].select(
                    range(min(len(ds[split]), dc.dataset_subsample[split]))
                )

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
