import logging

import hydra
from omegaconf import DictConfig

from post_training import utils
from post_training.data_sft.utils_for_dataset import get_mix_datasets

utils.config.register_resolvers()
hydra_logger = logging.getLogger(__name__)


# Dataset mixture
@hydra.main(version_base=None, config_path="../configs", config_name="dataset-mixture")
def main(config: DictConfig) -> None:
    ############################ Config Setup ############################
    utils.seeding.seed_everything(config)

    ########################### Dataset Mixing ############################
    assert config.dataset_mixer is not None, "data_mixer is required in config"

    raw_datasets = get_mix_datasets(
        dataset_mixer=config.dataset_mixer,
        columns_to_keep=config.columns_to_keep,
        need_columns=config.need_columns,
        keep_ids=config.keep_ids,
        shuffle=config.shuffle,
        save_data_dir=config.save_data_dir,
        seed=config.seed,
    )

    # print first 5 samples of dataset
    hydra_logger.info("Printing first 5 training samples")
    for i in range(5):
        hydra_logger.info(raw_datasets["train"][i])


if __name__ == "__main__":
    main()
