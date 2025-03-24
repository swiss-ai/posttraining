import logging
import os

import hydra
from datasets import ClassLabel
from omegaconf import DictConfig

from swiss_alignment import utils
from swiss_alignment.utils.utils_for_dataset import (
    load_dataset_flexible,
    save_dataset_flexible,
)

utils.config.register_resolvers()
hydra_logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../configs", config_name="dataset_preprocessing"
)
def main(config: DictConfig) -> None:
    ############################ Config Setup ############################
    utils.seeding.seed_everything(config)

    ############################ Dataset Split ###########################
    # Loads datasets from both local paths and huggingface
    ds = load_dataset_flexible(config.dataset_args.dataset_name)

    train_split_name = config.dataset_args.train_split.name
    eval_split_name = config.dataset_args.eval_split.name

    if train_split_name in ds and eval_split_name not in ds:
        hydra_logger.info("No validation split found. Creating one from train set...")
        train_data = ds[train_split_name]

        # Handle stratification if specified
        strat_col = config.dataset_args.stratify_by_column
        if strat_col:
            if isinstance(train_data.features[strat_col], ClassLabel):
                names = train_data.features[strat_col].names
            else:
                names = sorted(set(train_data[strat_col]))

            train_data = train_data.cast_column(strat_col, ClassLabel(names=names))

        # Split dataset
        split_dataset = train_data.train_test_split(
            test_size=config.dataset_args.eval_split.ratio,
            seed=config.seed,
            shuffle=True,
            stratify_by_column=strat_col,
        )

        # Update dataset with splits, converting strat_col back to labels if stratified
        if strat_col:
            class_label_feature = split_dataset["train"].features[strat_col]
            for split_key, ds_key in [
                ("train", train_split_name),
                ("test", eval_split_name),
            ]:
                values = [
                    class_label_feature.names[idx]
                    for idx in split_dataset[split_key][strat_col]
                ]
                ds[ds_key] = (
                    split_dataset[split_key]
                    .remove_columns(strat_col)
                    .add_column(strat_col, values)
                )
        else:
            # No stratification, just use the splits as-is
            ds[train_split_name] = split_dataset["train"]
            ds[eval_split_name] = split_dataset["test"]

        # Log stats
        hydra_logger.info(f"Original train size: {len(train_data)}")
        hydra_logger.info(f"New train size: {len(ds[train_split_name])}")
        hydra_logger.info(f"New validation size: {len(ds[eval_split_name])}")

        # Save if output path specified
        if config.dataset_args.output_path:
            save_dataset_flexible(ds, config.dataset_args.output_path)
    else:
        hydra_logger.info(
            "Dataset already has validation/test split or no train split."
        )


if __name__ == "__main__":
    main()
