import json
import logging

import hydra
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from omegaconf import DictConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def flatten_ref_rewards(sample):
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

    return sample


@hydra.main(config_path="../configs", config_name="mix-datasets-with-ref-rewards")
def main(config: DictConfig) -> None:
    logger.info("Loading dataset from disk...")

    datasets = [
        load_from_disk(config.dataset_path_prefix + mode_id)
        for mode_id in config.to_mix_ids
    ]

    split_names = [config.dataset_args.train_split.name]
    eval_split_name = config.dataset_args.eval_split.name
    if eval_split_name is not None:
        split_names.append(eval_split_name)

    new_dataset_dict = {}

    for split_name in split_names:
        logger.info(f"Processing split: {split_name}")
        split_ds = [d[split_name] for d in datasets]

        processed_split_ds = [
            split_d.map(
                flatten_ref_rewards,
                batched=False,
                num_proc=220,
            )
            for split_d in split_ds
        ]

        processed_split_ds = concatenate_datasets(processed_split_ds)
        new_dataset_dict[split_name] = processed_split_ds

    new_dataset = DatasetDict(new_dataset_dict)
    logger.info("Saving processed dataset to disk...")
    new_dataset.save_to_disk(config.save_path)
    logger.info("Dataset processing complete.")


if __name__ == "__main__":
    main()
