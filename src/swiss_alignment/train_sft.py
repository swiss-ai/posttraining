import dataclasses
import logging
import os
import subprocess
import sys
from pathlib import Path

import hydra
import wandb
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf, omegaconf
from transformers import AutoTokenizer
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from swiss_alignment import utils

utils.config.register_resolvers()
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train_sft")
def main(config: DictConfig) -> None:
    logger.info(f"Init directory: {Path.cwd()}")
    resuming_dir, resuming_hash = utils.config.setup_resuming_dir(config)
    logger.info(f"Run can be resumed from the directory: {resuming_dir}")
    if config.resuming.resume:
        os.chdir(resuming_dir)
        logger.info(f"Resuming from the directory: {Path.cwd()}")

    # full_config is a merge with the TRL arg dataclasses
    # The args dataclasses are used by the HF classes, and the full_config by the template.
    full_config, script_args, training_args, model_args = postprocess_and_save_config(
        config
    )

    wandb_run_id = config.wandb.run_id
    if wandb_run_id is None:
        if config.resuming.resume:
            wandb_run_id = resuming_hash
    wandb.init(
        id=wandb_run_id,
        resume="allow" if config.resuming.resume else "never",
        config=OmegaConf.to_container(full_config),
        project=config.wandb.project,
        tags=config.wandb.tags,
        mode=config.wandb.mode,
        anonymous=config.wandb.anonymous,
        dir=Path(config.wandb.dir).absolute(),
    )

    # Re-log to capture log with wandb.
    logger.info(f"Running command: {subprocess.list2cmdline(sys.argv)}")
    logger.info(f"Init directory: {config.run_dir}")
    logger.info(f"Run can be resumed from the directory: {resuming_dir}")
    logger.info(f"Working directory: {Path.cwd()}")
    logger.info(f"Running with config: \n{OmegaConf.to_yaml(config)}")
    if config.resuming.resume:
        logger.info(f"Resuming from the directory: {Path.cwd()}")

    # Update this function whenever you have a library that needs to be seeded.
    utils.seeding.seed_everything(config)

    # Code here from https://github.com/huggingface/trl/edit/main/trl/scripts/sft.py
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Training
    ################

    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Code above from https://github.com/huggingface/trl/edit/main/trl/scripts/sft.py

    # Train and save the model.
    resume_from_checkpoint = (
        config.resuming.resume
        and len([item for item in Path(config.resuming_dir).iterdir() if item.is_dir()])
        > 1  # counting the config dir.
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(resuming_dir)


def postprocess_and_save_config(config):
    """Here you can make some computations with the config to add new keys, correct some values, etc.
    E.g., read-only variables that can be useful when navigating the experiments on wandb
     for filtering, sorting, etc.
    Save the new config (as a file to record it) and pass it to wandb to record it with your experiment.
    """
    Path("config/").mkdir(exist_ok=True)
    utils.config.maybe_save_config(config, "config/config-before-postprocess.yaml")
    with omegaconf.open_dict(config):
        config.some_new_key = "bar"
    OmegaConf.resolve(config)

    script_args = ScriptArguments(**config.script_args)
    training_args = SFTConfig(**config.training_args, output_dir=str(Path.cwd()))
    model_args = ModelConfig(**config.model_args)

    # Code here from https://github.com/huggingface/trl/edit/main/trl/scripts/sft.py
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    # Code above from https://github.com/huggingface/trl/edit/main/trl/scripts/sft.py

    utils.config.maybe_save_config(config, "config/config-resolved.yaml")

    full_config = OmegaConf.create(
        {
            "script_args": dataclasses.asdict(script_args),
            "training_args": training_args.to_dict(),
            "model_args": dataclasses.asdict(model_args),
        }
    )
    full_config = OmegaConf.merge(full_config, config)

    utils.config.maybe_save_config(full_config, "config/full-config-resolved.yaml")

    return full_config, script_args, training_args, model_args


if __name__ == "__main__":
    main()
