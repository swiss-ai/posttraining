import logging
import random

import datasets
import hydra
import numpy as np
from omegaconf import DictConfig

from swiss_alignment import utils
from swiss_alignment.data_alignment.linearize_swissaiformat import (
    linearise_sample_for_sft,
)

utils.config.register_resolvers()
logger = logging.getLogger(__name__)


def select_chosen_rejected_pairs(row, num_pairs):
    # Sort conversation branches by reward.
    conv_branches = row["conversation_branches"]
    # Shuffle first to remove bias in the order of offline models.
    random.shuffle(conv_branches)
    conv_branches.sort(key=lambda x: x["quantile_reward"], reverse=True)
    row["conversation_branches"] = conv_branches

    # Heuristic to pick the training pairs.
    # For DPO it's usually better to pick rejected as worst and chosen as best.
    # For QRPO it doesn't matter there is no chosen and rejected, a good balance is good.
    # We have offline and off-policy samples. The quantile distribution is not clear.
    pairs = []

    worst_quantile_val = max(0.2, 1 / row["num_ref_rewards"])  # 0.2 is a heuristic
    best_quantile = [
        conv_branch
        for conv_branch in conv_branches
        if conv_branch["quantile_reward"] >= 1
    ]
    middle_quantile = [
        conv_branch
        for conv_branch in conv_branches
        if conv_branch["quantile_reward"] < 1
        and conv_branch["quantile_reward"] > worst_quantile_val
    ]
    worst_quantile = [
        conv_branch
        for conv_branch in conv_branches
        if conv_branch["quantile_reward"] <= worst_quantile_val
    ]

    # Pick one pair with one from the best and one from the worst quantile.
    if len(best_quantile) > 0:
        chosen = best_quantile.pop(0)
    elif len(middle_quantile) > 0:
        chosen = middle_quantile.pop(0)
    else:
        chosen = worst_quantile.pop(0)
    if len(worst_quantile) > 0:
        rejected = worst_quantile.pop(0)
    elif len(middle_quantile) > 0:
        rejected = middle_quantile.pop(0)
    else:
        rejected = best_quantile.pop(0)
    pairs.append((chosen, rejected))

    remaining_pairs = num_pairs - 1

    # Divide the remaining branches into > 0.5 and < 0.5 quantiles.
    if remaining_pairs > 0:
        remaining_convs = best_quantile + middle_quantile + worst_quantile

        top_quantile = [
            conv_branch
            for conv_branch in remaining_convs
            if conv_branch["quantile_reward"] >= 0.5
        ]
        bottom_quantile = [
            conv_branch
            for conv_branch in remaining_convs
            if conv_branch["quantile_reward"] < 0.5
        ]
        random.shuffle(top_quantile)
        random.shuffle(bottom_quantile)

        for _ in range(remaining_pairs):
            if len(top_quantile) > 0:
                chosen = top_quantile.pop(0)
            else:
                chosen = bottom_quantile.pop(0)

            if len(bottom_quantile) > 0:
                rejected = bottom_quantile.pop(0)
            else:
                rejected = top_quantile.pop(0)

            pair = [chosen, rejected]
            pair.sort(key=lambda x: x["quantile_reward"], reverse=True)
            pairs.append(pair)

    row["chosen_rejected_pairs"] = pairs
    return row


# 34342 is a good example.


def extract_reference_rewards(row):
    ref_rewards = []
    for i, conv_branch in enumerate(row["conversation_branches"]):
        if conv_branch["is_reference_completion"]:
            ref_rewards.append(conv_branch["reward"])
    row["ref_rewards"] = ref_rewards
    row["num_ref_rewards"] = len(ref_rewards)
    return row


def compute_quantile_rewards(row):
    ref_rewards = row["ref_rewards"]
    for i, conv_branch in enumerate(row["conversation_branches"]):
        r = conv_branch["reward"]
        quantile_reward = sum(ref_reward <= r for ref_reward in ref_rewards) / len(
            ref_rewards
        )
        row["conversation_branches"][i]["quantile_reward"] = quantile_reward
    return row


def convert_old_format_to_new_format(row):
    """Add missing keys to the row if they are not present."""
    for i, conv_branch in enumerate(row["conversation_branches"]):
        if "is_reference_completion" not in conv_branch.keys():
            is_reference_completion = conv_branch["messages"][-1]["parts"][0][
                "metadata"
            ].get("is_reference_completion", False)
            if is_reference_completion == None:
                is_reference_completion = False
            row["conversation_branches"][i][
                "is_reference_completion"
            ] = is_reference_completion
        if "rewards" in conv_branch.keys():
            row["conversation_branches"][i]["reward"] = row["conversation_branches"][i][
                "rewards"
            ]
            del row["conversation_branches"][i]["rewards"]
    return row


# To filter out rows that are too long
def convert_to_preference_dataset(row, idx, num_pairs):
    chosen_rejected_index = idx % num_pairs
    chosen, rejected = row["chosen_rejected_pairs"][chosen_rejected_index]
    row["conversation_branches"] = [chosen]
    chosen_linear_chat = linearise_sample_for_sft(row)
    row["conversation_branches"] = [rejected]
    rejected_linear_chat = linearise_sample_for_sft(row)

    # TODO: assumes chosen and rejected share a common prefix which is the prompt.
    # The completions is assumed to be a single chat from the assistant as a "response".

    new_row = {
        "chosen": chosen_linear_chat,
        "rejected": rejected_linear_chat,
        "ref_chosen_logprob": chosen["completion_ref_logprob"],
        "ref_rejected_logprob": rejected["completion_ref_logprob"],
        "chosen_reward": chosen["reward"],
        "rejected_reward": rejected["reward"],
        "ref_rewards": row["ref_rewards"],
        "chosen_quantile_reward": chosen["quantile_reward"],
        "rejected_quantile_reward": rejected["quantile_reward"],
    }

    return new_row


@hydra.main(
    config_path="../configs",
    config_name="prepare-train-dataset-swissaiformat",
)
def main(config: DictConfig) -> None:
    config = utils.config.setup_config_and_resuming(config)
    utils.seeding.seed_everything(config)

    data = datasets.load_from_disk(config.dataset_args.dataset_name)

    # For previous versions (deprecate when possible)
    data = data.map(
        convert_old_format_to_new_format,
        num_proc=256,
    )

    # sort by len(conversation branches)
    # Add num conv branches
    data = data.map(
        lambda x: {"num_conv_branches": len(x["conversation_branches"])},
        num_proc=256,
    )
    # Filter out rows that don't have enough conversation branches.
    min_num_conv_branches = config.num_pairs_per_prompt * 2
    data = data.filter(
        lambda x: x["num_conv_branches"] >= min_num_conv_branches,
        num_proc=256,
    )
    data = data.map(
        extract_reference_rewards,
        num_proc=256,
    )
    # Filter row with less than min reference rewards.
    min_num_ref_rewards = max(config.min_num_ref_rewards, 1)
    data = data.filter(
        lambda x: x["num_ref_rewards"] >= min_num_ref_rewards,
        num_proc=256,
    )
    data = data.map(
        compute_quantile_rewards,
        num_proc=256,
    )
    data = data.map(
        select_chosen_rejected_pairs,
        num_proc=256,
        fn_kwargs={"num_pairs": config.num_pairs_per_prompt},
    )

    # Save this intermediate dataset.
    logger.info(f"Saving swissaiformat dataset to {config.swissaiformat_save_path}")
    data.save_to_disk(config.swissaiformat_save_path)

    # preference format https://huggingface.co/docs/trl/main/en/dataset_formats#preference
    # Prepare the result linearized dataset.
    # Duplicate data for each pair and process in parallel.
    splits = {}
    for split_name, split in data.items():
        N, K = len(split), config.num_pairs_per_prompt
        indices = np.repeat(np.arange(N), K).tolist()
        split = split.select(indices)
        splits[split_name] = split

    data = datasets.DatasetDict(splits)
    data = data.map(
        convert_to_preference_dataset,
        num_proc=256,
        with_indices=True,
        remove_columns=data[split_name].column_names,
        fn_kwargs={"num_pairs": config.num_pairs_per_prompt},
    )

    logger.info(f"Saving dataset to {config.preference_save_path}")
    data.save_to_disk(config.preference_save_path)
    logger.info("Dataset saved successfully.")
    logger.info("Dataset filtered successfully!")


if __name__ == "__main__":
    main()
