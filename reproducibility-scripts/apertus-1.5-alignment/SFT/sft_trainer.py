import argparse
import os
from pathlib import Path
import datasets
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

os.environ.setdefault("WANDB_ENTITY", "apertus")
os.environ.setdefault("WANDB_PROJECT", "apertus-1.5-post-training-dpo")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHAT_TEMPLATE = SCRIPT_DIR / "chat_template.jinja"


VALID_ROLES = {"system", "user", "assistant", "tool"}


def normalize_messages(example):
    normalized = []
    for msg in example["messages"]:
        role = msg["role"]
        if role not in VALID_ROLES:
            example["messages"] = None
            return example
        c = msg.get("content")
        if c is None:
            c = ""
        elif isinstance(c, list):
            c = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in c
            )
        elif not isinstance(c, str):
            c = str(c)
        normalized.append({"role": role, "content": c})
    example["messages"] = normalized
    return example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat_template_path", default=str(DEFAULT_CHAT_TEMPLATE))
    ap.add_argument("--model_name_or_path", default="swiss-ai/Apertus-8B-2509")
    ap.add_argument("--dataset_name", default="allenai/Dolci-Instruct-SFT")
    ap.add_argument("--dataset_split", default="train")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--run_name", default="apertus-8b-sft-dolci")
    ap.add_argument("--max_seq_length", type=int, default=8192)
    ap.add_argument("--num_train_epochs", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler_type", default="cosine")
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--save_steps", type=int, default=2000)
    ap.add_argument("--eval_steps", type=int, default=512)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument("--subset_frac", type=float, default=1.0,
                    help="Fraction of dataset to use (e.g. 0.1 for 10%%)")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.chat_template = Path(args.chat_template_path).read_text()
    tokenizer.eos_token = "<|assistant_end|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
    )

    ds = load_dataset(args.dataset_name, split=args.dataset_split)
    datasets.disable_caching()
    ds = ds.map(normalize_messages, num_proc=128)
    pre_filter = len(ds)
    ds = ds.filter(lambda x: x["messages"] is not None, num_proc=128)
    print(f"Filtered {pre_filter - len(ds)} examples with invalid roles ({len(ds)} remaining)")
    if args.subset_frac < 1.0:
        n = int(len(ds) * args.subset_frac)
        ds = ds.shuffle(seed=42).select(range(n))

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        packing=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim="adamw_torch",
        eval_strategy="no",
        dataset_num_proc=128,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="wandb",
        run_name=args.run_name,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_args,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()