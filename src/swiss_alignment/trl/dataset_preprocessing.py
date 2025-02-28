import logging
import os

import hydra
from datasets import ClassLabel, DatasetDict, load_dataset, load_from_disk
from omegaconf import DictConfig

from swiss_alignment import utils

utils.config.register_resolvers()
hydra_logger = logging.getLogger(__name__)


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


@hydra.main(
    version_base=None, config_path="../configs", config_name="dataset_preprocessing"
)
def main(config: DictConfig) -> None:
    ############################ Config Setup ############################
    utils.seeding.seed_everything(config)

    ############################ Dataset Split ###########################
    # Loads datasets from both local paths and huggingface
    ds = load_dataset_flexible(config.dataset_args.dataset_name)

    if (
        "train" in ds.keys()
        and "validation" not in ds.keys()
        and "test" not in ds.keys()
    ):
        hydra_logger.info("No validation split found. Creating one from train set...")

        train_data = ds["train"]

        # If 'source' column specified, cast to ClassLabel
        if config.dataset_args.stratify_by_column is not None:
            unique_sources = sorted(
                set(train_data[config.dataset_args.stratify_by_column])
            )
            class_label_feature = ClassLabel(names=unique_sources)
            train_data = train_data.cast_column(
                config.dataset_args.stratify_by_column, class_label_feature
            )

        # Use train_test_split with stratification if specified
        split_dataset = train_data.train_test_split(
            test_size=config.dataset_args.val_ratio,
            seed=config.seed,
            shuffle=True,
            stratify_by_column=config.dataset_args.stratify_by_column,  # Preserve distribution
        )

        # Create new DatasetDict with both splits
        new_dataset = DatasetDict(
            {"train": split_dataset["train"], "test": split_dataset["test"]}
        )

        # Print some stats
        hydra_logger.info(f"Original train size: {len(train_data)}")
        hydra_logger.info(f"New train size: {len(new_dataset['train'])}")
        hydra_logger.info(f"New validation size: {len(new_dataset['validation'])}")

        # Save to disk or push to huggingface based on output_path
        if config.dataset_args.output_path:
            save_dataset_flexible(new_dataset, config.dataset_args.output_path)
    else:
        hydra_logger.info(
            "Dataset already has validation/test split or no train split."
        )
        return


if __name__ == "__main__":
    main()
