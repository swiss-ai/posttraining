import logging

import hydra
import matplotlib.pyplot as plt
import numpy as np
from accelerate.logging import get_logger
from accelerate.state import PartialState
from datasets import DatasetDict, load_from_disk
from omegaconf import DictConfig, OmegaConf
from trl import ScriptArguments

from swiss_alignment import utils
from swiss_alignment.trl.tokenization import TokenizerConfig, get_tokenizer
from swiss_alignment.utils import utils_for_gen_ratio, utils_for_trl

utils.config.register_resolvers()
acc_state = PartialState()
acc_logger = get_logger(__name__)
hydra_logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="trl-sft")
def main(config: DictConfig) -> None:
    ############################ Config Setup ############################
    config = utils_for_trl.setup_config_and_resuming(config, acc_state, acc_logger)
    script_args = ScriptArguments(**OmegaConf.to_container(config.script_args))
    utils.seeding.seed_everything(config)

    ############################ Tokenizer Setup ############################
    tc = TokenizerConfig(
        model_name_or_path=config.tokenizer_args.tokenizer_name_or_path,
        model_pad_token_id=config.tokenizer_args.model_pad_token_id,
        model_eos_token_id=config.tokenizer_args.model_eos_token_id,
        chat_template_name=config.dataset_args.chat_template_name,
        add_bos=False,
        trust_remote_code=config.tokenizer_args.trust_remote_code,
    )
    tokenizer = get_tokenizer(tc)

    ############################ Dataset Setup ############################
    # Make sure to download the dataset before.
    ds = load_from_disk(script_args.dataset_name)
    ds = DatasetDict(
        {
            "train": ds[script_args.dataset_train_split],
            "eval": ds[script_args.dataset_test_split],
        }
    )

    if config.dataset_args.debug_subsample.train > 0:
        ds["train"] = ds["train"].select(
            range(min(len(ds["train"]), config.dataset_args.debug_subsample.train))
        )
    if config.dataset_args.debug_subsample.eval > 0:
        ds["eval"] = ds["eval"].select(
            range(min(len(ds["eval"]), config.dataset_args.debug_subsample.eval))
        )

    # Calculating generation ratio
    gen_ratios = utils_for_gen_ratio.compute_generation_ratios(ds, tokenizer)

    # Computing mean value across dataset
    Rg = np.mean(gen_ratios)
    hydra_logger.info(f"{script_args.dataset_name} Rg\t= {Rg:.4g}")

    # Plotting distribution
    plt.hist(gen_ratios, bins=50)
    plt.show()


if __name__ == "__main__":
    main()
