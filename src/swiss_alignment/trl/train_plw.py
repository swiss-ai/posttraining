import logging
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
    SFTTrainer,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from swiss_alignment import utils
from swiss_alignment.trl.tokenization import get_tokenizer, TokenizerConfig
from swiss_alignment.utils import utils_for_trl
from swiss_alignment.trl.trainers import PLWTrainer, preprocess_logits_for_plw_metrics
from functools import partial

utils.config.register_resolvers()
acc_state = PartialState()
acc_logger = get_logger(__name__)
hydra_logger = logging.getLogger(__name__)


from datasets import DatasetDict, Sequence, Value, load_from_disk, Dataset
import numpy as np
from queue import PriorityQueue
from itertools import chain
import random

# shortest pack first histogram packing
def spfhp(seq_lens, chunk_length=2048):
    q = PriorityQueue()
    q.put((0,[]))
    idx = seq_lens.argsort()[::-1]
    for i in idx:
        n, pack = seq_lens[i], q.get()
        if n+pack[0] > chunk_length:
            q.put(pack)
            pack = (0,[])
        q.put((n+pack[0], pack[1]+[i]))
    return list(q.queue)

# pack sequences into chunks
def pack(sample, chunk_length=2048, pad_token_id=0):
    # compute packing arrangement
    seq_lens = np.array([len(t) for t in sample["input_ids"]])
    chunks = spfhp(seq_lens, chunk_length=chunk_length)
    random.shuffle(chunks)

    # pack sequences according to arrangement
    result = {}
    for k in sample.keys():
        result[k] = []
        pad_id = pad_token_id if k == "input_ids" else 0
        for chunk in chunks:
            item = list(chain(*[sample[k][i] for i in chunk[1]], [pad_id]*(chunk_length-chunk[0])))
            result[k].append(item)

    # add labels (same as input_ids!)
    result["labels"] = result["input_ids"].copy()
    return result

# draw simple ascii histogram
def ascii_hist(x, nb=10, maxlen=100):
    w = np.ptp(x)/nb  # get bin width from num bins
    min_val, max_val = np.min(x), np.max(x)     # get min/max vals
    bins = np.arange(min_val, max_val + 1, w)   # create bins
    hist, _ = np.histogram(x, bins)     # get histogram sizes
    scale = maxlen/hist.max()
    # draw histogram
    for i in range(len(hist)):
        print(f"{bins[i]:0.0f} - {bins[i]+w:0.0f}\t{'#' * int(scale*hist[i])}")

# Function to tokenize and encode a batch of samples, and creates prompt/completion masks.
# Note: This function assumes a single user/asst chat exchange (i.e. prompt + completion).
# For arbitrary length user/asst chat dialogues, a more general user-masking solution was proposed
# here: https://github.com/huggingface/trl/issues/632#issuecomment-1972630547
def tokenize_batch(batch, tokenizer):
    # tokenize and encode text
    tokenized_text = tokenizer(batch["text"], add_special_tokens=False, return_offsets_mapping=True,)
    data = {k: tokenized_text[k] for k in tokenized_text.keys()}

    # use offset_mappings to make prompt/completion masks (idx marks the start of each completion)
    prompt_masks, completion_masks = [],[]
    for offset_mapping, idx in zip(data["offset_mapping"], batch["idx"]):
        prompt_masks.append([1 if o[1] < idx else 0 for o in offset_mapping])
        completion_masks.append([0 if o[1] < idx else 1 for o in offset_mapping])

    data["prompt_mask"] = prompt_masks
    data["completion_mask"] = completion_masks
    del data["offset_mapping"]
    return data

# tokenize and pack dataset
def tokenize_and_pack(dataset, tokenizer, config):
    # Tokenize dataset, remove original columns
    tokenized = dataset.map(partial(tokenize_batch, tokenizer=tokenizer), batched=True, remove_columns=list(dataset.features))

    # Cast mask columns to int8
    for col in ["prompt_mask", "completion_mask"]:
        tokenized = tokenized.cast_column(col, Sequence(Value("int8")))

    # Filter sequences longer than max_seq_length
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) <= config.training_args.max_seq_length)

    # Print sample count
    print(f"Number of samples: {len(tokenized)}")

    # Pack sequences
    packed = tokenized.map(
        lambda x: pack(x, chunk_length=config.training_args.max_seq_length, pad_token_id=tokenizer.pad_token_id),
        batched=True
    )
    packed = packed.cast_column("labels", Sequence(Value("int32")))

    # Calculate and print packing stats
    seq_lens = [len(x) for x in tokenized["input_ids"]]
    total_packed = len(packed) * config.training_args.max_seq_length
    print(f"Packing density:     {100 * sum(seq_lens) / total_packed:.1f}%")
    print(f"Packing compression: {100 * len(packed) / len(tokenized):.1f}%")
    return packed

def prepare_dataset(dataset, tokenizer, config):
    # Helper function to format each sample
    def format_sample(sample):
        # Apply chat template to full conversation
        sample["text"] = tokenizer.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=False
        )
        # Get prompt length for completion index
        prompt_text = tokenizer.apply_chat_template(
            sample["messages"][:1], tokenize=False, add_generation_prompt=True
        )
        sample["idx"] = len(prompt_text)
        return sample

    processed = DatasetDict({})
    for k in dataset.keys():
        split = dataset[k].map(format_sample)
        split = tokenize_and_pack(split, tokenizer, config)
        processed[k] = split
        print(f"Total count of {k} packed sequences: {len(split)}")

    return processed


@hydra.main(version_base=None, config_path="../configs", config_name="trl-sft")
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

    full_config = utils_for_trl.postprocess_and_save_config(
        config, script_args, training_args, model_args, acc_state
    )
    if acc_state.is_main_process:
        utils.config.setup_wandb(full_config, acc_logger)
    utils.seeding.seed_everything(config)

    ############################ Tokenizer Setup ############################
    tc = TokenizerConfig(
        model_name_or_path=config.tokenizer_args.tokenizer_name_or_path,
        model_pad_token_id=config.tokenizer_args.model_pad_token_id,
        model_eos_token_id=config.tokenizer_args.model_eos_token_id,
        chat_template_name=config.dataset_args.chat_template_name,
        add_bos=False,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = get_tokenizer(tc)

    ############################ Dataset Setup ############################
    # Make sure to download the dataset before.
    ds = load_from_disk(script_args.dataset_name)
    ds = DatasetDict({
        "train": ds[config.script_args.dataset_train_split],
        "eval": ds[config.script_args.dataset_test_split],
    })

    if config.dataset_args.debug_subsample.train > 0:
        ds["train"] = ds["train"].select(
            range(min(len(ds["train"]), config.dataset_args.debug_subsample.train))
        )
    if config.dataset_args.debug_subsample.eval > 0:
        ds["eval"] = ds["eval"].select(
            range(min(len(ds["eval"]), config.dataset_args.debug_subsample.eval))
        )

    ds = prepare_dataset(ds, tokenizer, config)

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

    trainer = PLWTrainer(
        prompt_loss_weight=0.1, # script_args.prompt_loss_weight
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=peft_config,
        # compute_metrics=prepare_compute_metrics(ds, tokenizer),
        # preprocess_logits_for_metrics=preprocess_logits_for_plw_metrics,
    )

    # Apply the token patches to the model
    if tc.model_eos_token_id is not None:
        trainer.model.config.eos_token_id = tc.model_eos_token_id
        trainer.model.generation_config.eos_token_id = tc.model_eos_token_id
        acc_logger.info(f"Overriding model eos token id to {tc.model_eos_token_id}")

    trainer.train(resume_from_checkpoint=last_checkpoint_number > 0)
    acc_logger.info("Training completed. Performing final evaluation.")
    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    acc_logger.info("Final evaluation completed.")

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
