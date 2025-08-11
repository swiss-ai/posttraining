import logging
import math
from datetime import timedelta
from pathlib import Path

import accelerate
import hydra
import wandb
from accelerate.logging import get_logger
from accelerate.state import PartialState
from omegaconf import DictConfig, OmegaConf
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from swiss_alignment import utils
from swiss_alignment.data_sft.tokenization import TokenizerConfig, get_tokenizer
from swiss_alignment.data_sft.utils_for_dataset import DatasetConfig, get_dataset_sft
from swiss_alignment.trainers.sft import (
    CustomSFTTrainer,
    LengthNormalizedPLWTrainer,
    PLWDataCollator,
    PLWTrainer,
)
from swiss_alignment.utils import utils_for_trl

utils.config.register_resolvers()
acc_state = PartialState(
    **accelerate.InitProcessGroupKwargs(timeout=timedelta(hours=4)).to_kwargs()
)
acc_logger = get_logger(__name__)
hydra_logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train-sft")
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
        add_bos_to_chat_template=config.tokenizer_args.add_bos_to_chat_template,
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
        debug_oom=config.dataset_args.debug_oom,
        dataset_subsample={
            "train": config.dataset_args.debug_subsample.train,
            "eval": config.dataset_args.debug_subsample.eval,
        },
        transform_fn=[
            # "sft_to_chatml_format",
            # "sft_filter_non_alternating_roles",
            "sft_tulu_tokenize_and_truncate",
            "sft_filter_has_assistant_tokens",
        ],
        transform_fn_args=[
            # {},
            # {},
            {"max_seq_length": training_args.max_seq_length},
            {},
        ],
        target_columns=[
            "input_ids",
            "labels",
            "attention_mask",
            "prompt_mask",
            "completion_mask",
        ],
        shuffle=config.dataset_args.shuffle,
        seed=config.seed,
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
    peft_config = get_peft_config(model_args)

    full_config = utils_for_trl.merge_and_save_config(
        config, script_args, training_args, model_args, acc_state
    )
    if acc_state.is_main_process:
        utils.config.setup_wandb(full_config, acc_logger)
        utils.config.try_sync_wandb()
    utils.seeding.seed_everything(config)

    ############################ Tokenizer Setup ############################
    tokenizer = get_tokenizer(tokenizer_args)

    ############################ Dataset Setup ############################
    ds = get_dataset_sft(dataset_config, tokenizer, acc_state)

    ############################ Trainer Setup ############################
    # Find the last checkpoint
    resuming_dir = Path.cwd()
    # Handle resuming
    last_checkpoint_number = 0
    for item in resuming_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            if (item / "scheduler.pt").is_file() and (
                item / "trainer_state.json"
            ).is_file():
                last_checkpoint_number = max(
                    last_checkpoint_number, int(item.name.split("-")[-1])
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
        "args": training_args,
        "train_dataset": ds["train"],
        "eval_dataset": ds["eval"] if training_args.eval_strategy != "no" else None,
        "processing_class": tokenizer,
        "data_collator": PLWDataCollator(tokenizer=tokenizer, mlm=False),
        "peft_config": peft_config,
    }
    if config.trainer == "sft":
        acc_logger.info("Starting sft trainer.")
        trainer = CustomSFTTrainer(
            **trainer_args,
        )
    elif config.trainer == "plw":
        acc_logger.info(f"Starting plw={config.plw_args.prompt_loss_weight} trainer.")
        trainer = PLWTrainer(
            prompt_loss_weight=config.plw_args.prompt_loss_weight,
            sequence_level_loss=config.plw_args.sequence_level_loss,
            **trainer_args,
        )
    elif config.trainer == "ln-plw":
        acc_logger.info(
            f"Starting ln-plw={config.plw_args.prompt_loss_weight} trainer."
        )
        trainer = LengthNormalizedPLWTrainer(
            prompt_loss_weight=config.plw_args.prompt_loss_weight,
            sequence_level_loss=config.plw_args.sequence_level_loss,
            **trainer_args,
        )
    else:
        raise ValueError(f"Unknown trainer type: {config.trainer}")

    # Apply the token patches to the model
    if tokenizer_args.model_eos_token_id is not None:
        trainer.model.config.eos_token_id = tokenizer_args.model_eos_token_id
        trainer.model.generation_config.eos_token_id = tokenizer_args.model_eos_token_id
        acc_logger.info(
            f"Overriding model eos token id to {tokenizer_args.model_eos_token_id}"
        )

    # Computing the warmup steps for beta3 and alpha in AdEMAMix
    if training_args.optim == "ademamix":
        len_ds = len(ds["train"])
        total_batch_size = trainer.get_total_train_batch_size(training_args)
        num_steps_per_epoch = int(
            len_ds // total_batch_size
            if training_args.dataloader_drop_last
            else math.ceil(len_ds / total_batch_size)
        )
        total_steps = training_args.num_train_epochs * num_steps_per_epoch
        # TODO move the beta3 and alpha to the training_args.optim_args command line argument.
        # This is not trivial for write in a way that is sent in a correct format through all the layers down to hydra.
        training_args.optim_args = (
            f"'beta3=0.9999,alpha=8.0,t_beta3={total_steps},t_alpha={total_steps}"
        )
        acc_logger.info(f"AdEMAMix optim_args: {trainer.args.optim_args}")

    trainer.train(resume_from_checkpoint=last_checkpoint_number > 0)
    acc_logger.info("Training completed. Performing final evaluation.")

    last_eval_file = resuming_dir / f"eval_results.json"
    if training_args.eval_strategy != "no":
        if last_eval_file.exists():
            acc_logger.info("Last evaluation already performed.")
        else:
            torch.cuda.empty_cache()
            acc_logger.info("Performing final evaluation.")
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            acc_logger.info("Final evaluation completed.")

    if training_args.num_train_epochs == 0:
        acc_logger.info("Training skipped. Saving the model.")
        trainer.save_model()

    acc_state.wait_for_everyone()
    acc_logger.info("Training completed. Checkpoints saved.")
    if acc_state.is_main_process:
        wandb.finish()
        utils.config.try_sync_wandb()
    acc_state.wait_for_everyone()
    accelerate.Accelerator().end_training()


if __name__ == "__main__":
    main()
