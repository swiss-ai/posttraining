import logging
import math
import os
from pathlib import Path

import datasets
import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm, trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from swiss_alignment import utils

utils.config.register_resolvers()
logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs",
    config_name="filter-model-sequences",
)
def main(config: DictConfig) -> None:
    config = utils.config.setup_config_and_resuming(config)
    utils.seeding.seed_everything(config)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_args.model_name_or_path, use_fast=True
    )

    data = datasets.load_from_disk(config.dataset_path)

    # To filter out rows that are too long
    def add_chat_num_tokens(row):
        chosen_tokens = tokenizer.apply_chat_template(row["chosen"], tokenize=True)
        rejected_tokens = tokenizer.apply_chat_template(row["rejected"], tokenize=True)
        chosen_text = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        rejected_text = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return {
            "chosen_model_tokens": chosen_tokens,
            "chosen_model_tokens_len": len(chosen_tokens),
            "chosen_model_text": chosen_text,
            "rejected_model_tokens": rejected_tokens,
            "rejected_model_tokens_len": len(rejected_tokens),
            "rejected_model_text": rejected_text,
            "max_chosen_rejected_model_tokens_len": max(
                len(chosen_tokens), len(rejected_tokens)
            ),
        }

    data = data.map(add_chat_num_tokens, num_proc=256)
    # Filter out the row with chat_num_tokens > training_args.max_seq_length
    data = data.filter(
        lambda max_chosen_rejected_model_tokens_len: max_chosen_rejected_model_tokens_len
        <= config.max_seq_length,
        input_columns=["max_chosen_rejected_model_tokens_len"],
        num_proc=256,
    )
    # Record the size of the dataset after filtering
    for split in data.values():
        logger.info(f"Filtered dataset size: {len(split)}")

    logger.info(f"Saving merged dataset to {config.save_path}")
    data.save_to_disk(config.save_path)
    logger.info("Dataset saved successfully.")
    logger.info("Dataset filtered successfully!")


if __name__ == "__main__":
    main()
