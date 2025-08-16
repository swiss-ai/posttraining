import copy
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from swiss_alignment import utils
from swiss_alignment.data_alignment.linearize_swissaiformat import (
    linearise_sample_for_sft,
)

utils.config.register_resolvers()
logger = logging.getLogger(__name__)


def compute_rewards_for_row(model, row, tokenizer, config):
    chats = []
    conv_branch_row = copy.deepcopy(row)
    for conv_branch in row["conversation_branches"]:
        conv_branch_row["conversation_branches"] = [conv_branch]
        chats.append(linearise_sample_for_sft(conv_branch_row))

    # TODO: Assumes we train on only the last message in the chat.
    # TODO: Would be incompatible if we want to train on say (response + verifiable-responses).
    tokenized_chats = tokenizer.apply_chat_template(
        chats,
        return_tensors="pt",
        padding=True,
        return_dict=True,
        truncation=True,
        max_length=config.max_seq_len,
    )
    chat_input_ids = tokenized_chats["input_ids"]
    chat_attention_mask = tokenized_chats["attention_mask"]

    all_rewards = []
    num_iters = math.ceil(len(chats) / config.batch_size)
    for i in range(num_iters):
        start_idx = i * config.batch_size
        end_idx = min((i + 1) * config.batch_size, len(chats))
        input_ids = chat_input_ids[start_idx:end_idx].to(model.device)
        attention_mask = chat_attention_mask[start_idx:end_idx].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            rewards = outputs.logits.cpu().float().reshape(-1)
            all_rewards.extend(rewards.tolist())

    new_row = copy.deepcopy(row)

    for i, conv_branch in enumerate(row["conversation_branches"]):
        new_row["conversation_branches"][i]["rewards"] = rewards[i]

    return new_row


def compute_rewards_batch(model, batch, tokenizer, config):
    rows_result = []
    for row in tqdm(batch, desc="Processing batch"):
        new_row = compute_rewards_for_row(
            model,
            row,
            tokenizer,
            config,
        )
        rows_result.append(new_row)

    return datasets.Dataset.from_list(rows_result)


def compute_subpartition_start_end_indices(
    partition_start_idx, partition_end_idx, subpartition_number, num_subpartitions
):
    subpartition_size = math.ceil(
        (partition_end_idx - partition_start_idx) / num_subpartitions
    )
    start_idx = partition_start_idx + subpartition_number * subpartition_size
    end_idx = partition_start_idx + (subpartition_number + 1) * subpartition_size
    end_idx = min(end_idx, partition_end_idx)

    return start_idx, end_idx


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="compute-ref-rewards-swissaiformat",
)
def main(config: DictConfig) -> None:
    config = utils.config.setup_config_and_resuming(config)
    random.seed(config.seed)

    tp_size = config.reward_model_distributed_config.tensor_parallel_size
    cuda_devices = [config.subpartition_number * tp_size + i for i in range(tp_size)]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_devices))
    logger.info(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Import after setting CUDA_VISIBLE_DEVICES to ensure the correct GPUs are used.
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.reward_model_args.pretrained_model_name_or_path, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        device_map="auto", **config.reward_model_args
    )

    # End index is exclusive
    num_subpartitions = (
        config.num_gpus_per_node // config.model_vllm_config.tensor_parallel_size
    )
    (
        subpartition_start_idx,
        subpartition_end_idx,
    ) = compute_subpartition_start_end_indices(
        config.partition_start_idx,
        config.partition_end_idx,
        config.subpartition_number,
        num_subpartitions,
    )

    if subpartition_start_idx >= subpartition_end_idx:
        logger.info("Subpartition is empty. Exiting.")
        return

    subpartition_data = datasets.load_from_disk(config.dataset_args.dataset_name)[
        config.split
    ].select(range(subpartition_start_idx, subpartition_end_idx))

    # Handle resuming.
    resuming_dir = Path.cwd()
    # Checkpoints are saved as `checkpoint-{last-relative-index-processed-in-the-subpartition}`.
    already_processed_samples = max(
        (
            int(item.name.split("-")[-1])
            for item in resuming_dir.iterdir()
            if item.is_dir() and item.name.startswith("checkpoint-")
        ),
        default=0,
    )
    if already_processed_samples == len(subpartition_data):
        logger.info(
            "All samples in the subpartition have already been processed. Exiting."
        )
        return

    local_start_idx = already_processed_samples  # 64, 128, ...
    if local_start_idx > 0:
        logger.info(
            f"Resuming from checkpoint-{local_start_idx}. Processing from sample {local_start_idx}."
        )

    pbar = tqdm(total=len(subpartition_data), desc="Computing rewards of completions")
    pbar.update(local_start_idx)
    while local_start_idx < len(subpartition_data):
        current_slice = (
            local_start_idx,
            min(local_start_idx + config.save_interval, len(subpartition_data)),
        )
        current_slice_data = subpartition_data.select(range(*current_slice))
        local_end_idx = local_start_idx + len(current_slice_data)

        current_slice_data = compute_rewards_batch(
            model, current_slice_data, tokenizer, config
        )
        save_path = resuming_dir / f"checkpoint-{local_end_idx}"
        current_slice_data.save_to_disk(save_path)
        logger.info(f"Saved checkpoint-{local_end_idx} successfully!")

        pbar.update(len(current_slice_data))
        # Update start index for the next chunk
        local_start_idx = local_end_idx

    logger.info("rewards computed successfully!")


if __name__ == "__main__":
    main()
