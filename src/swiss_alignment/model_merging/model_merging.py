import logging
from pathlib import Path

import hydra
import yaml
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge
from omegaconf import DictConfig

from swiss_alignment import utils

utils.config.register_resolvers()
hydra_logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="model-merging")
def main(config: DictConfig) -> None:
    ############################ Config Setup ############################
    # Ensure model merging config is readable
    config_yml_path = Path(config.config_yml).resolve()
    try:
        with open(config_yml_path, "r", encoding="utf-8") as fp:
            merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_yml_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_yml_path}: {e}")

    # Ensure output directory exists
    output_path = Path(config.output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    hydra_logger.debug(f"Output directory ready: {output_path}")

    utils.seeding.seed_everything(config)

    ############################ Model Merging ############################
    run_merge(
        merge_config,
        out_path=config.output_path,
        options=MergeOptions(
            gpu_rich=config.gpu_rich,
            copy_tokenizer=config.copy_tokenizer,
            lazy_unpickle=config.lazy_unpickle,
            low_cpu_memory=config.low_cpu_memory,
            trust_remote_code=config.trust_remote_code,
            random_seed=config.seed,
        ),
    )
    hydra_logger.info("Model merging completed successfully")


if __name__ == "__main__":
    main()
