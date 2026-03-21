import argparse
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

os.environ.setdefault("WANDB_ENTITY", "apertus")
os.environ.setdefault("WANDB_PROJECT", "apertus-1.5-post-training-dpo")

# Simplified Apertus-native chat template (matches Apertus-8B-Instruct format)
# Uses tokens already in the pretrained vocabulary — no embedding resize needed
APERTUS_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}"
    "<|system_start|>{{ messages[0]['content'] }}<|system_end|>"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "<|system_start|>You are Apertus, a helpful assistant.<|system_end|>"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "<|developer_start|>Deliberation: disabled\nTool Capabilities: disabled<|developer_end|>"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "<|user_start|>{{ message['content'] }}<|user_end|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|assistant_start|>{{ message['content'] }}<|assistant_end|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|assistant_start|>"
    "{% endif %}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="swiss-ai/Apertus-8B-2509")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--gas", type=int, default=8)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--subset_frac", type=float, default=1.0,
                    help="Fraction of dataset to use (e.g. 0.1 for 10%%)")
    args = ap.parse_args()

    # Load tokenizer, set native Apertus chat template
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.chat_template = APERTUS_CHAT_TEMPLATE
    tokenizer.eos_token = "<|assistant_end|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"

    # Load model — no embedding resize needed, all tokens already in vocab
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
    )

    # Load dataset (already has 'messages' column with [{role, content}, ...])
    ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train")
    if args.subset_frac < 1.0:
        n = int(len(ds) * args.subset_frac)
        ds = ds.shuffle(seed=42).select(range(n))

    # SFT config
    sft_args = SFTConfig(
        output_dir=args.out,
        max_seq_length=args.max_length,
        packing=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.gas,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        warmup_ratio=0.05,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        eval_strategy="no",
        dataset_num_proc=128,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="wandb",
        run_name="apertus-8b-sft-dolci",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_args,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)


if __name__ == "__main__":
    main()
