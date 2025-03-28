import logging
import os

import hydra
from datasets import DatasetDict, concatenate_datasets
from omegaconf import DictConfig

from swiss_alignment import utils
from swiss_alignment.utils.utils_for_dataset import (
    load_dataset_flexible,
    save_dataset_flexible,
)

utils.config.register_resolvers()
hydra_logger = logging.getLogger(__name__)


# Adapted from: https://github.com/allenai/open-instruct/blob/main/open_instruct/utils.py
# ----------------------------------------------------------------------------
# Dataset utilities
def is_openai_format(messages) -> bool:
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
def convert_alpaca_gpt4_to_messages(example):
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


def convert_codefeedback_single_turn_to_messages(example):
    """
    Convert a query-answer pair to a list of messages.
    e.g. m-a-p/CodeFeedback-Filtered-Instruction"""
    messages = [
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    example["messages"] = messages
    return example


def convert_metamath_qa_to_messages(example):
    """
    Convert a query-response pair to a list of messages.
    e.g. meta-math/MetaMathQA"""
    messages = [
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": example["response"]},
    ]
    example["messages"] = messages
    return example


def convert_code_alpaca_to_messages(example):
    """
    Convert a prompt-completion pair to a list of messages.
    e.g. HuggingFaceH4/CodeAlpaca_20K"""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    example["messages"] = messages
    return example


def convert_open_orca_to_messages(example):
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


def conversations_to_messages(example):
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


def convert_rejection_samples_to_messages(example):
    """
    Convert a rejection sampling dataset to messages.
    """
    example["messages"] = example["chosen"]
    return example


# Dataset mixture
@hydra.main(version_base=None, config_path="../configs", config_name="dataset_mixture")
def main(config: DictConfig) -> None:
    ############################ Config Setup ############################
    utils.seeding.seed_everything(config)

    # print save location
    if config.save_data_dir:
        print(f"Saving mixed dataset to {config.save_data_dir}")

    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_eval_datasets = []
    frac_or_sample_list = []
    columns_to_keep = [] if config.columns_to_keep is None else config.columns_to_keep
    for ds_config in config.dataset_mixer:
        frac_or_sample_list.append(ds_config.frac_or_samples)

        ds = load_dataset_flexible(ds_config.dataset_path)
        for split in ds.keys():
            # only include splits specific in the config file
            if (
                split not in ds_config.train_splits
                and split not in ds_config.eval_splits
            ):
                continue

            # shuffle dataset if set
            if config.shuffle:
                ds[split] = ds[split].shuffle(seed=42)

            # assert that needed columns are present
            if config.need_columns:
                if not all(
                    col in ds[split].column_names for col in config.need_columns
                ):
                    raise ValueError(
                        f"Needed column {config.need_columns} not found in dataset {ds[split].column_names}."
                    )

            # handle per-case conversions
            # if "instruction" and "output" columns are present and "messages" is not, convert to messages
            if (
                "instruction" in ds[split].column_names
                and "output" in ds[split].column_names
                and "messages" not in ds[split].column_names
            ):
                ds[split] = ds[split].map(convert_alpaca_gpt4_to_messages, num_proc=10)
            elif (
                "prompt" in ds[split].column_names
                and "completion" in ds[split].column_names
                and "messages" not in ds[split].column_names
            ):
                ds[split] = ds[split].map(convert_code_alpaca_to_messages, num_proc=10)
            elif (
                "conversations" in ds[split].column_names
                and "messages" not in ds[split].column_names
            ):
                ds[split] = ds[split].map(conversations_to_messages, num_proc=10)
            elif (
                "question" in ds[split].column_names
                and "response" in ds[split].column_names
                and "messages" not in ds[split].column_names
            ):
                ds[split] = ds[split].map(convert_open_orca_to_messages, num_proc=10)
            elif (
                "query" in ds[split].column_names
                and "answer" in ds[split].column_names
                and "messages" not in ds[split].column_names
            ):
                ds[split] = ds[split].map(
                    convert_codefeedback_single_turn_to_messages, num_proc=10
                )
            elif (
                "query" in ds[split].column_names
                and "response" in ds[split].column_names
                and "messages" not in ds[split].column_names
            ):
                ds[split] = ds[split].map(convert_metamath_qa_to_messages, num_proc=10)
            elif (
                "chosen" in ds[split].column_names
                and "rejected" in ds[split].column_names
                and "reference_completion" in ds[split].column_names
                and "messages" not in ds[split].column_names
            ):
                ds[split] = ds[split].map(
                    convert_rejection_samples_to_messages, num_proc=10
                )

            # if id not in dataset, create it as ds-{index}
            if "id" not in ds[split].column_names:
                id_col = [f"{ds}_{i}" for i in range(len(ds[split]))]
                ds[split] = ds[split].add_column("id", id_col)

            # Remove redundant columns to avoid schema conflicts on load
            ds[split] = ds[split].remove_columns(
                [
                    col
                    for col in ds[split].column_names
                    if col not in (columns_to_keep + ["id"])
                ]
            )

            # if add_source_col, add that column
            if config.add_source_col:
                source_col = [ds_config.dataset_name] * len(ds[split])
                ds[split] = ds[split].add_column("source", source_col)

            # for cols in columns_to_keep, if one is not present, add "None" to the column
            for col in columns_to_keep:
                if col not in ds[split].column_names:
                    ds[split] = ds[split].add_column(col, [None] * len(ds[split]))

            # add dataset to train/eval split
            if split in ds_config.train_splits:
                raw_train_datasets.append(ds[split])
            elif split in ds_config.eval_splits:
                raw_eval_datasets.append(ds[split])
            else:
                raise ValueError(
                    f"Split type {split} not recognized as one of test or train."
                )

    if len(raw_eval_datasets) == 0 and len(raw_train_datasets) == 0:
        raise ValueError("No datasets loaded.")
    elif len(raw_train_datasets) == 0:
        # target features are the features of the first dataset post load
        target_features = raw_eval_datasets[0].features
    else:
        # target features are the features of the first dataset post load
        target_features = raw_train_datasets[0].features

    if any(frac_or_samples < 0 for frac_or_samples in frac_or_sample_list):
        raise ValueError("Dataset fractions / lengths cannot be negative.")

    # if any > 1, use count
    if any(frac_or_samples > 1 for frac_or_samples in frac_or_sample_list):
        is_count = True
        # assert that all are integers
        if not all(
            isinstance(frac_or_samples, int) for frac_or_samples in frac_or_sample_list
        ):
            raise NotImplementedError("Cannot mix fractions and counts, yet.")
    else:
        is_count = False

    if len(raw_train_datasets) > 0:
        train_subsets = []
        # Manage proportions
        for dataset, frac_or_samples in zip(raw_train_datasets, frac_or_sample_list):
            dataset = dataset.cast(target_features)
            if is_count:
                train_subset = dataset.select(range(frac_or_samples))
            else:
                train_subset = dataset.select(
                    range(int(frac_or_samples * len(dataset)))
                )
            train_subsets.append(train_subset)

        raw_datasets["train"] = concatenate_datasets(train_subsets)

    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_eval_datasets) > 0:
        eval_subsets = []
        for dataset in raw_eval_datasets:
            eval_subsets.append(dataset.cast(target_features))

        raw_datasets["test"] = concatenate_datasets(eval_subsets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {config.dataset_mixer} not recognized."
            "Check the dataset has been correctly formatted."
        )

    # optional save
    if config.save_data_dir:
        save_dataset_flexible(raw_datasets, config.save_data_dir)

    # if not config.keep_ids:
    #     # remove id column
    #     if len(raw_train_datasets) > 0:
    #         if "id" in raw_datasets["train"].column_names:
    #             raw_datasets["train"] = raw_datasets["train"].remove_columns("id")
    #     if len(raw_eval_datasets) > 0:
    #         if "id" in raw_datasets["test"].column_names:
    #             raw_datasets["test"] = raw_datasets["test"].remove_columns("id")

    # return raw_datasets


if __name__ == "__main__":
    main()
