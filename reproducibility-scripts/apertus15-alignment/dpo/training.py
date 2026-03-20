import os
import sys

import yaml
import torch
from dataclasses import dataclass, field
from typing import Optional
from dataclasses import asdict

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset, load_from_disk

from trl import (
    ModelConfig,
    TrlParser,
    get_peft_config,
)

from dpo import NormedDPOTrainer, NormedDPOConfig

@dataclass
class ScriptArguments:
    """
    Arguments for dataset paths and custom processing logic.
    """
    dataset_path: str = field(
        metadata={"help": "Path to the training dataset (local path or HF Hub)."}
    )
    dataset_config: Optional[str] = field(
        default=None, metadata={"help": "Dataset configuration name (if using HF Hub)."}
    )
    debug_mode: bool = field(
        default=False, metadata={"help": "Whether to run in debug mode with a smaller dataset."}
    )


def process_dataset_split(dataset):
    """Normalization to handle different dataset formats (train vs train_prefs)."""
    if isinstance(dataset, dict):
        if "train_prefs" in dataset:
            return dataset["train_prefs"]
        elif "train" in dataset:
            return dataset["train"]
        elif "train_split" in dataset:
            return dataset["train_split"]
        else:
            raise ValueError("Could not find 'train' or 'train_prefs' or 'train_split' in dataset splits.")
    return dataset


def prepare_dataset_for_dpo(dataset, tokenizer, max_length, num_proc=os.cpu_count()):
    """
    Optimized filtering logic:
    1. Formats and tokenizes in a single map pass to calculate lengths.
    2. Filters based on calculated lengths.
    """
    def check_length_fn(examples):
        # Helper to apply template to a specific column key (chosen/rejected)
        def get_len(key):
            return [
                len(tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=False))
                for msg in examples[key]
            ]
        
        try:
            chosen_lens = get_len("chosen")
            rejected_lens = get_len("rejected")
        except Exception:
            # Fallback if columns are not standard list-of-dicts
            return [True] * len(examples["chosen"])

        # Create boolean mask: Keep if BOTH chosen and rejected fit in max_length
        return [
            (c <= max_length) and (r <= max_length) 
            for c, r in zip(chosen_lens, rejected_lens)
        ]

    print(f"Filtering dataset by max_length={max_length}...")
    original_len = len(dataset)
    
    dataset = dataset.filter(
        check_length_fn,
        batched=True,
        num_proc=num_proc
    )
    
    print(f"Filtered dataset: {original_len} -> {len(dataset)} samples.")
    
    dataset = dataset.filter(lambda x: len(x["chosen"]) > 0 and len(x["rejected"]) > 0 
                             and x["chosen"][0]['content'] != "" and x["rejected"][0]['content'] != "",
                             num_proc=num_proc)
    print(f"After removing empty samples: {len(dataset)} samples remain.")
    
    return dataset


def sanitize_generation_config(model):
    gen_config = model.generation_config
    has_temp = hasattr(gen_config, "temperature") and gen_config.temperature is not None
    has_top_p = hasattr(gen_config, "top_p") and gen_config.top_p is not None
    
    if (has_temp or has_top_p) and not gen_config.do_sample:
        print("🔧 Dynamic Fix: Sampling parameters detected. Setting do_sample=True.")
        gen_config.do_sample = True


def main(script_args, training_args, model_args):
    os.environ["WANDB_ENTITY"] = "apertus"
    os.environ["WANDB_PROJECT"] = "apertus-1.5-post-training-dpo"
    
    # 2. Set Seed (TRL/Transformers handles this via training_args, but we enforce explicit expectation)
    set_seed(training_args.seed)

    # 3. Load Model & Tokenizer
    # ModelConfig handles standard args like attn_implementation, trust_remote_code, etc.
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    
    sanitize_generation_config(model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Load & Process Dataset
    # We use main_process_first to ensure map/filter happens once and is cached correctly
    with training_args.main_process_first(desc="dataset loading and processing"):
        try:
            dataset = load_dataset(script_args.dataset_path, name=script_args.dataset_config)
        except Exception:
            dataset = load_from_disk(script_args.dataset_path)

        dataset = process_dataset_split(dataset)

        # Clean columns — drop prompt so TRL re-extracts it from chosen/rejected
        # (avoids Arrow type conflict when prompt column has mixed null/string values)
        cols_to_keep = {"chosen", "rejected"}
        cols_to_remove = [c for c in dataset.column_names if c not in cols_to_keep]
        if cols_to_remove:
            dataset = dataset.remove_columns(cols_to_remove)

        # Debug trimming
        if script_args.debug_mode:
            print("Debug mode: trimming dataset to 100 samples.")
            dataset = dataset.select(range(min(100, len(dataset))))

        # Length filtering
        if training_args.max_length:
            dataset = prepare_dataset_for_dpo(
                dataset, 
                tokenizer, 
                max_length=training_args.max_length
            )
        
        # Shuffle
        dataset = dataset.shuffle(seed=training_args.seed)
    
    # 5. Initialize Trainer
    # get_peft_config automatically extracts LoRA params from ModelConfig/YAML if present
    peft_config = get_peft_config(model_args)

    trainer = NormedDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 6. Train & Save
    trainer.train()
    
    print("Saving model to:", training_args.output_dir)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print("Model saved.")
    
    final_config = {
        "script_args": asdict(script_args),
        "model_args": asdict(model_args),
        "training_args": training_args.to_dict(), # training_args is a HF object, so we use .to_dict()
    }

    config_path = os.path.join(training_args.output_dir, "final_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(final_config, f, default_flow_style=False)


if __name__ == "__main__":
    # TrlParser parses arguments into the specified dataclasses
    parser = TrlParser((ScriptArguments, NormedDPOConfig, ModelConfig))
    
    script_args, training_args, model_args = parser.parse_args_and_config()

    # Print args for verification (similar to the SFT script)
    print("Script Arguments:", script_args)
    print("Training Arguments:", training_args)
    print("Model Arguments:", model_args)

    main(script_args, training_args, model_args)