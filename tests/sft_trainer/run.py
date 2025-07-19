import argparse
import multiprocessing
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

set_seed(1234)


def fine_tune_trl(output_dir, batch_size, gradient_accumulation_steps):
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        use_fast=True,
        padding_side="right",
        pad_token="<|finetune_right_pad_id|>",
        pad_token_id=128004,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True}
    )

    ds = load_dataset("timdettmers/openassistant-guanaco")
    ds = ds.map(
        # Add the EOS token to timdettmers/openassistant-guanaco
        lambda row: {"text": row["text"] + tokenizer.eos_token},
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )

    training_arguments = SFTConfig(
        dataset_text_field="text",
        dataset_kwargs={"skip_preprocessing": True},
        bf16=True,
        optim="paged_adamw_8bit",
        num_train_epochs=1,
        max_seq_length=512,
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=gradient_accumulation_steps,
        output_dir=output_dir,
        log_level="debug",
        logging_steps=25,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="no",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tokenizer,
        args=training_arguments,
    )

    # --code by Unsloth to track memory consumption: https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=pCqnaKmlO1U9
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

    trainer_ = trainer.train()

    with open(os.path.join(output_dir, "report.json"), "w") as f:
        import json

        json.dump(trainer.state.log_history, f)

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_trainer = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    trainer_percentage = round(used_memory_for_trainer / max_memory * 100, 3)

    with open(os.path.join(output_dir, "memory.txt"), "w") as f:
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.", file=f)
        print(f"{start_gpu_memory} GB of memory reserved.", file=f)

        print(f"{trainer_.metrics['train_runtime']} seconds used for training.", file=f)
        print(
            f"{round(trainer_.metrics['train_runtime'] / 60, 2)} minutes used for training.",
            file=f,
        )
        print(f"Peak reserved memory = {used_memory} GB.", file=f)
        print(
            f"Peak reserved memory for training = {used_memory_for_trainer} GB.", file=f
        )
        print(f"Peak reserved memory % of max memory = {used_percentage} %.", file=f)
        print(
            f"Peak reserved memory for training % of max memory = {trainer_percentage} %.",
            file=f,
        )


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Fine-tune a model using TRL SFTTrainer."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size per device for training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32,
        help="Number of gradient accumulation steps.",
    )
    args = parser.parse_args()

    # Call the fine-tune function with parsed arguments
    fine_tune_trl(args.output_dir, args.batch_size, args.gradient_accumulation_steps)
