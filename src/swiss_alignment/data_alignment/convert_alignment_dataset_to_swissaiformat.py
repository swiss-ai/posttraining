import argparse
import os
import datasets
from pathlib import Path
from transformers import AutoTokenizer


def make_convert_row(tokenizer):
    """Create a conversion function with access to the tokenizer."""

    def convert_row(row):
        """Convert chosen/rejected format to conversation_branches format."""
        branches = []
        for completion in [row["chosen"], row["rejected"]]:
            messages = []
            for msg in completion:
                if msg["role"] == "assistant":
                    part_type = "response"
                else:
                    part_type = "text"
                messages.append({
                    "role": msg["role"],
                    "parts": [{"type": part_type, "content": msg["content"], "metadata": {}}],
                })

            # Compute token counts needed by compute_ref_logprobs
            from src.swiss_alignment.data_alignment.linearize_swissaiformat import (
                linearise_sample_for_sft,
            )
            temp_row = {"conversation_branches": [{"messages": messages}]}
            if "prompt_id" in row:
                temp_row["prompt_id"] = row["prompt_id"]

            linear_chat = linearise_sample_for_sft(temp_row)
            chat_tokens = tokenizer.apply_chat_template(linear_chat, tokenize=True)
            context_tokens = tokenizer.apply_chat_template(
                linear_chat[:-1], tokenize=True, add_generation_prompt=True
            )

            # Add token counts to the last assistant part's metadata
            messages[-1]["parts"][-1]["metadata"]["chat_num_tokens"] = len(chat_tokens)
            messages[-1]["parts"][-1]["metadata"]["context_num_tokens"] = len(context_tokens)

            branch = {"messages": messages, "is_reference_completion": True}
            branches.append(branch)

        result = {
            "conversation_branches": branches,
            "max_branch_tokens": max(
                b["messages"][-1]["parts"][-1]["metadata"]["chat_num_tokens"]
                for b in branches
            ),
        }
        if "prompt_id" in row:
            result["prompt_id"] = row["prompt_id"]
        return result

    return convert_row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a chosen/rejected DPO dataset to SwissAI conversation_branches format."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input dataset (load_from_disk format).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the converted dataset.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the tokenizer (same model used for ref logprobs).",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="Maximum sequence length. Rows exceeding this will be filtered out.",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="train_split",
        help="Name of the split in the output DatasetDict (default: train_split).",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print(f"Loading dataset from {args.input_path}")
    ds = datasets.load_from_disk(args.input_path)
    print(f"Dataset: {ds}")

    convert_row = make_convert_row(tokenizer)

    def process_split(split_ds, split_name):
        original_len = len(split_ds)
        print(f"Converting split '{split_name}' ({original_len} rows)...")
        split_ds = split_ds.map(convert_row, remove_columns=["chosen", "rejected"], num_proc=os.cpu_count())

        # Filter out rows where any branch exceeds max_seq_len
        before_filter = len(split_ds)
        split_ds = split_ds.filter(
            lambda row: row["max_branch_tokens"] <= args.max_seq_len,
            num_proc=os.cpu_count(),
        )
        after_filter = len(split_ds)
        removed = before_filter - after_filter
        print(f"  Filtered: {removed}/{before_filter} rows removed (exceeded max_seq_len={args.max_seq_len})")
        print(f"  Remaining: {after_filter} rows")

        # Remove the helper column
        split_ds = split_ds.remove_columns(["max_branch_tokens"])
        return split_ds

    # Handle both Dataset and DatasetDict inputs
    if isinstance(ds, datasets.DatasetDict):
        # Take the "train" split from input
        if "train" not in ds:
            raise ValueError(f"Expected 'train' split in DatasetDict, found: {list(ds.keys())}")
        processed = process_split(ds["train"], "train")
        output = datasets.DatasetDict({"train_split": processed})
    else:
        # Plain Dataset — wrap it into a DatasetDict with "train_split"
        print(f"Input is a plain Dataset, wrapping into DatasetDict with split 'train_split'")
        processed = process_split(ds, "train_split")
        output = datasets.DatasetDict({"train_split": processed})

    print(f"\nOutput DatasetDict: {output}")
    output.save_to_disk(args.output_path)
    print(f"Saved converted dataset to {args.output_path}")