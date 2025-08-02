import json
import logging
from typing import Any, Dict

import hydra
import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from omegaconf import DictConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def replace_chosen_rejected(sample, best_idx, worst_idx, completions, rewards):
    # Overwrite chosen info.
    sample["chosen"][1]["content"] = completions[best_idx]
    sample["chosen_rewards"] = rewards[best_idx]
    sample["chosen_reward_tokens"] = sample["ref_completions_reward_tokens"][best_idx]
    sample["chosen_reward_tokens_len"] = sample["ref_completions_reward_tokens_len"][
        best_idx
    ]
    sample["chosen_reward_text"] = sample["ref_completions_reward_texts"][best_idx]
    # Overwrite rejected info.
    sample["rejected"][1]["content"] = completions[worst_idx]
    sample["rejected_rewards"] = rewards[worst_idx]
    sample["rejected_reward_tokens"] = sample["ref_completions_reward_tokens"][
        worst_idx
    ]
    sample["rejected_reward_tokens_len"] = sample["ref_completions_reward_tokens_len"][
        worst_idx
    ]
    sample["rejected_reward_text"] = sample["ref_completions_reward_texts"][worst_idx]
    sample["max_chosen_rejected_reward_tokens_len"] = max(
        sample["chosen_reward_tokens_len"], sample["rejected_reward_tokens_len"]
    )
    return sample


"""
1. offpolicy2best logic
"""


def process_row_offpolicy2best(
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    """
    For each row, among the first 6 completions, pick the best (max reward)
    as 'chosen' and the worst (min reward) as 'rejected'.
    """
    completions = json.loads(sample["ref_completions"][1]["content"])
    rewards = sample["ref_rewards"]

    # Consider only the first 6 completions/rewards.
    rewards_first_6 = rewards[:6]
    best_idx = max(range(6), key=lambda i: rewards_first_6[i])
    worst_idx = min(range(6), key=lambda i: rewards_first_6[i])

    return replace_chosen_rejected(sample, best_idx, worst_idx, completions, rewards)


"""
1. offpolicy2random logic
"""


def process_row_offpolicy2random(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    For each row, among the first 6 completions, pick the best (max reward)
    as 'chosen' and the worst (min reward) as 'rejected'.
    """
    completions = json.loads(sample["ref_completions"][1]["content"])
    rewards = sample["ref_rewards"]

    # TODO: remove path. Fix it in compute ref rewards.
    if hasattr(rewards[0], "__len__"):
        # flatten the rewards if they are lists.
        rewards = [r[0] for r in rewards]
    sample["ref_rewards"] = rewards

    if rewards[1] >= rewards[0]:
        best_idx = 1
        worst_idx = 0
    else:
        best_idx = 0
        worst_idx = 1

    return replace_chosen_rejected(sample, best_idx, worst_idx, completions, rewards)


def replicate_Kx(dataset: Dataset, K: int) -> Dataset:
    """
    Efficiently replicate the dataset Kx by selecting repeated indices and then adding
    a new column "pair_idx" which indicates which pair (0,1,2, ..., K-1) the row corresponds to.
    """
    logger.info("Replicating dataset using index-based selection...")
    N = len(dataset)
    # Create an index list where each original index is repeated 3 times.
    indices = np.repeat(np.arange(N), K).tolist()
    replicated_ds = dataset.select(indices)
    # Create the pair_idx column: for each original row, values 0,1,2, ..., K-1.
    pair_idx = np.tile(np.arange(K), N).tolist()
    replicated_ds = replicated_ds.add_column("pair_idx", pair_idx)
    logger.info(
        f"Replication complete: {N} rows expanded to {len(replicated_ds)} rows."
    )
    return replicated_ds


def pick_pair_offpolicyKrandom(sample: Dict[str, Any], K) -> Dict[str, Any]:
    """
    For each row with a given pair_idx, pick the chosen/rejected completions from:
       pair_idx = 0 -> use completions at indices (0, 1)
       pair_idx = 1 -> use completions at indices (2, 3)
       pair_idx = 2 -> use completions at indices (4, 5)
    The better (higher reward) becomes 'chosen', and the lower becomes 'rejected'.
    """
    completions = json.loads(sample["ref_completions"][1]["content"])
    rewards = sample["ref_rewards"]

    pair_idx_to_indices = {i: (i * 2, i * 2 + 1) for i in range(K)}
    i1, i2 = pair_idx_to_indices[sample["pair_idx"]]

    # Choose best/worst based on reward.
    if rewards[i1] >= rewards[i2]:
        best_idx, worst_idx = i1, i2
    else:
        best_idx, worst_idx = i2, i1

    return replace_chosen_rejected(sample, best_idx, worst_idx, completions, rewards)


@hydra.main(config_path="../configs", config_name="prepare-offpolicy-dataset")
def main(config: DictConfig) -> None:
    logger.info("Loading dataset from disk...")
    dataset = load_from_disk(config.dataset_path)

    # Identify dataset splits.
    split_names = [config.dataset_args.train_split.name]
    eval_split_name = config.dataset_args.eval_split.name
    if eval_split_name is not None:
        split_names.append(eval_split_name)

    new_dataset_dict = {}

    for split_name in split_names:
        logger.info(f"Processing split: {split_name}")
        if split_name not in dataset:
            raise ValueError(f"Split '{split_name}' not found in dataset.")
        split_ds = dataset[split_name]

        if config.mode == "offpolicy2best":
            logger.info(
                "Using offpolicy2best mode: processing each row without replication."
            )
            processed_split_ds = split_ds.map(
                process_row_offpolicy2best,
                batched=False,
                num_proc=240,
            )
        elif config.mode == "offpolicy2random":
            logger.info(
                "Using offpolicy2random mode: processing each row without replication."
            )
            processed_split_ds = split_ds.map(
                process_row_offpolicy2random,
                batched=False,
                num_proc=220,
            )
        elif config.mode == "offpolicy10random":
            logger.info(
                "Using offpolicy10random mode: replicating rows and processing pairs."
            )
            expanded_ds = replicate_Kx(split_ds, 10)
            processed_split_ds = expanded_ds.map(
                pick_pair_offpolicyKrandom,
                batched=False,
                num_proc=8,
                fn_kwargs={"K": 10},
            )
            processed_split_ds = processed_split_ds.remove_columns("pair_idx")

        else:
            raise ValueError(f"Unknown mode: {config.mode}")

        new_dataset_dict[split_name] = processed_split_ds

    new_dataset = DatasetDict(new_dataset_dict)
    logger.info("Saving processed dataset to disk...")
    new_dataset.save_to_disk(config.save_path)
    logger.info("Dataset processing complete.")


if __name__ == "__main__":
    main()
