import logging
from pathlib import Path

import accelerate
import hydra
from accelerate.logging import get_logger
from accelerate.state import PartialState
from datasets import DatasetDict, load_from_disk
from omegaconf import DictConfig, OmegaConf
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    get_kbit_device_map,
    get_quantization_config,
)

from swiss_alignment import utils
from swiss_alignment.trl.tokenization import TokenizerConfig, get_tokenizer
from swiss_alignment.utils import utils_for_trl

utils.config.register_resolvers()
acc_state = PartialState()
acc_logger = get_logger(__name__)
hydra_logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../src/swiss_alignment/configs", config_name="trl-sft")
def main(config: DictConfig) -> None:
    ############################ Config Setup ############################

    config = utils_for_trl.setup_config_and_resuming(config, acc_state, acc_logger)
    # full_config is a merge with the TRL arg dataclasses
    # The args dataclasses are used by the HF classes, and the full_config by the template.
    script_args = ScriptArguments(**OmegaConf.to_container(config.script_args))
    training_args = SFTConfig(
        **OmegaConf.to_container(config.training_args), output_dir=str(Path.cwd())
    )
    model_args = ModelConfig(**OmegaConf.to_container(config.model_args))
    tokenizer_args = TokenizerConfig(
        model_name_or_path=config.tokenizer_args.tokenizer_name_or_path,
        padding_side=config.tokenizer_args.padding_side,
        add_bos=config.tokenizer_args.add_bos,
        trust_remote_code=model_args.trust_remote_code,
        chat_template_name=config.tokenizer_args.chat_template_name,
        model_pad_token_id=config.tokenizer_args.model_pad_token_id,
        model_eos_token_id=config.tokenizer_args.model_eos_token_id,
    )

    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    utils.seeding.seed_everything(config)

    ############################ Tokenizer Setup ############################
    tokenizer = get_tokenizer(tokenizer_args)

    ############################ Dataset Setup ############################

    # Make sure to download the dataset before.
    ds = load_from_disk(script_args.dataset_name)
    ds = DatasetDict(
        {
            "train": ds[config.script_args.dataset_train_split],
        }
    )
    # Handle preference datasets:
    if "chosen" in ds["train"].column_names:
        ds = ds.map(lambda row: {"messages": row["chosen"]})
        # Drop the extra preference columns
        for extra_key in ["prompt", "completion", "chosen", "rejected"]:
            if extra_key in ds["train"].column_names:
                ds["train"] = ds["train"].remove_columns([extra_key])
            if extra_key in ds["eval"].column_names:
                ds["eval"] = ds["eval"].remove_columns([extra_key])


    if config.dataset_args.debug_subsample.train > 0:
        ds["train"] = ds["train"].select(
            range(min(len(ds["train"]), config.dataset_args.debug_subsample.train))
        )


    # Shuffle at the end to preserve previous cache across seeds.
    ds = ds.shuffle(seed=config.seed)
    ############################ Trainer Setup ############################

    # Find the last checkpoint
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=ds["train"],
        processing_class=tokenizer,
    )

    # Apply the token patches to the model
    if tokenizer_args.model_eos_token_id is not None:
        trainer.model.config.eos_token_id = tokenizer_args.model_eos_token_id
        trainer.model.generation_config.eos_token_id = tokenizer_args.model_eos_token_id
        acc_logger.info(
            f"Overriding model eos token id to {tokenizer_args.model_eos_token_id}"
        )

    trainer.train(resume_from_checkpoint=False)
    acc_logger.info("Training completed.")

    if training_args.num_train_epochs == 0:
        acc_logger.info("Training skipped. Saving the model.")
        trainer.save_model()

    acc_state.wait_for_everyone()
    accelerate.Accelerator().end_training()


if __name__ == "__main__":
    main()
