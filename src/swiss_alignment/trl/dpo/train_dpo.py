import logging
from pathlib import Path

import accelerate
import hydra
import wandb
from accelerate.logging import get_logger
from accelerate.state import PartialState
from omegaconf import DictConfig, OmegaConf
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from swiss_alignment import utils
from swiss_alignment.trl.tokenization import TokenizerConfig, get_tokenizer
from swiss_alignment.utils import utils_for_trl
from swiss_alignment.utils.utils_for_dataset import DatasetConfig, get_dataset

utils.config.register_resolvers()
acc_state = PartialState()
acc_logger = get_logger(__name__)
hydra_logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="trl-dpo")
def main(config: DictConfig) -> None:
    ############################ Config Setup ############################

    config = utils_for_trl.setup_config_and_resuming(config, acc_state, acc_logger)
    # full_config is a merge with the TRL arg dataclasses
    # The args dataclasses are used by the HF classes, and the full_config by the template.
    script_args = ScriptArguments(**OmegaConf.to_container(config.script_args))
    training_args = DPOConfig(
        **OmegaConf.to_container(config.training_args), output_dir=str(Path.cwd())
    )
    model_args = ModelConfig(**OmegaConf.to_container(config.model_args))
    if config.tokenizer_args.chat_template_name == "tulu":
        # DPOTrainer manually adds EOS tokens to the end of chosen and rejected
        config.tokenizer_args.chat_template_name = "tulu_no_eos"

    tokenizer_args = TokenizerConfig(
        model_name_or_path=config.tokenizer_args.tokenizer_name_or_path,
        padding_side=config.tokenizer_args.padding_side,
        add_bos=config.tokenizer_args.add_bos,
        trust_remote_code=config.tokenizer_args.trust_remote_code,
        chat_template_name=config.tokenizer_args.chat_template_name,
        model_pad_token_id=config.tokenizer_args.model_pad_token_id,
        model_eos_token_id=config.tokenizer_args.model_eos_token_id,
    )

    dataset_config = DatasetConfig(
        dataset_name=script_args.dataset_name,
        dataset_split_names={
            "train": script_args.dataset_train_split,
            "eval": script_args.dataset_test_split,
        },
        dataset_subsample={
            "train": config.dataset_args.debug_subsample.train,
            "eval": config.dataset_args.debug_subsample.eval,
        },
        transform_fn=[
            # transformation done inside DPOTrainer class
        ],
        transform_fn_args=[
            # transformation done inside DPOTrainer class
        ],
        target_columns=[
            # target columns are not applicable
        ],
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
    # the same ref model config
    training_args.ref_model_init_kwargs = model_kwargs

    peft_config = get_peft_config(model_args)

    full_config = utils_for_trl.merge_and_save_config(
        config, script_args, training_args, model_args, acc_state
    )
    if acc_state.is_main_process:
        utils.config.setup_wandb(full_config, acc_logger)
    utils.seeding.seed_everything(config)

    ############################ Tokenizer Setup ############################
    tokenizer = get_tokenizer(tokenizer_args)

    ############################ Dataset Setup ############################
    ds = get_dataset(dataset_config, tokenizer, acc_state)

    # Shuffle at the end to preserve previous cache across seeds.
    ds = ds.shuffle(seed=config.seed)

    ############################ Trainer Setup ############################

    # Find the last checkpoint
    resuming_dir = Path.cwd()
    last_checkpoint_number = max(
        (
            int(item.name.split("-")[-1])
            for item in resuming_dir.iterdir()
            if item.is_dir() and item.name.startswith("checkpoint-")
        ),
        default=0,
    )
    if last_checkpoint_number > 0:
        acc_logger.info(
            f"TRL will attempt to resume from last checkpoint: {last_checkpoint_number}"
        )
        eval_file = resuming_dir / f"eval_{last_checkpoint_number}_results.json"
        if eval_file.exists():
            training_args.eval_on_start = False

    trainer_args = {
        "model": model_args.model_name_or_path,
        "ref_model": model_args.model_name_or_path,
        "args": training_args,
        "train_dataset": ds["train"],
        "processing_class": tokenizer,
        "peft_config": peft_config,
    }

    trainer = DPOTrainer(
        **trainer_args,
    )

    # Apply the token patches to the model
    if tokenizer_args.model_eos_token_id is not None:
        trainer.model.config.eos_token_id = tokenizer_args.model_eos_token_id
        trainer.model.generation_config.eos_token_id = tokenizer_args.model_eos_token_id
        acc_logger.info(
            f"Overriding model eos token id to {tokenizer_args.model_eos_token_id}"
        )

    trainer.train(resume_from_checkpoint=last_checkpoint_number > 0)
    acc_logger.info("Training completed.")

    if training_args.num_train_epochs == 0:
        acc_logger.info("Training skipped. Saving the model.")
        trainer.save_model()

    acc_state.wait_for_everyone()
    if acc_state.is_main_process:
        wandb.finish()
    acc_state.wait_for_everyone()
    accelerate.Accelerator().end_training()


if __name__ == "__main__":
    main()
