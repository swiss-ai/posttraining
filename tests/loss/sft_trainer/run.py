import argparse

import torch, os, multiprocessing
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)
from trl import SFTTrainer, SFTConfig

set_seed(1234)

compute_dtype = torch.bfloat16
attn_implementation = 'flash_attention_2'


def fine_tune_trl(model_name, output_dir, batch_size, gradient_accumulation_steps):
    ds = load_dataset("timdettmers/openassistant-guanaco")

    if "meta" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = 128004
        tokenizer.padding_side = 'right'


    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = "<|im_end|>"
        tokenizer.pad_token_id = 2
        tokenizer.padding_side = 'left'

    # Add the EOS token
    def process(row):
        row["text"] = row["text"] + tokenizer.eos_token
        return row

    ds = ds.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map={"": 0}, attn_implementation=attn_implementation
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})


    training_arguments = SFTConfig(
        output_dir=output_dir,
        eval_strategy="steps",
        do_eval=True,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_level="debug",
        save_strategy="no",
        logging_steps=25,
        learning_rate=1e-5,
        bf16=True,
        eval_steps=25,
        num_train_epochs=1,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        dataset_text_field="text",
        max_seq_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        tokenizer=tokenizer,
        args=training_arguments,
    )

    # --code by Unsloth to track memory consumption: https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=pCqnaKmlO1U9

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)


    trainer_ = trainer.train()


    with open(os.path.join(output_dir, 'report.json'), 'w') as f:
        import json
        json.dump(trainer.state.log_history, f)

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_trainer = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    trainer_percentage = round(used_memory_for_trainer / max_memory * 100, 3)

    with open(os.path.join(output_dir, 'memory.txt'), 'w') as f:
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.", file=f)
        print(f"{start_gpu_memory} GB of memory reserved.", file=f)

        print(f"{trainer_.metrics['train_runtime']} seconds used for training.", file=f)
        print(f"{round(trainer_.metrics['train_runtime'] / 60, 2)} minutes used for training.", file=f)
        print(f"Peak reserved memory = {used_memory} GB.", file=f)
        print(f"Peak reserved memory for training = {used_memory_for_trainer} GB.", file=f)
        print(f"Peak reserved memory % of max memory = {used_percentage} %.", file=f)
        print(f"Peak reserved memory for training % of max memory = {trainer_percentage} %.", file=f)


# fine_tune_trl("meta-llama/Llama-3.2-1B", batch_size=8, gradient_accumulation_steps=2)
# fine_tune_trl("meta-llama/Llama-3.2-1B", batch_size=16, gradient_accumulation_steps=1)
# fine_tune_trl("meta-llama/Llama-3.2-1B", batch_size=1, gradient_accumulation_steps=16)
# fine_tune_trl("meta-llama/Llama-3.2-1B", batch_size=2, gradient_accumulation_steps=8)
# fine_tune_trl("meta-llama/Llama-3.2-1B", batch_size=4, gradient_accumulation_steps=4)


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Fine-tune a model using TRL SFTTrainer.")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for checkpoints and logs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32,
                        help="Number of gradient accumulation steps.")

    args = parser.parse_args()

    # Call the fine-tune function with parsed arguments
    fine_tune_trl("meta-llama/Llama-3.2-1B", args.output_dir, args.batch_size, args.gradient_accumulation_steps)
