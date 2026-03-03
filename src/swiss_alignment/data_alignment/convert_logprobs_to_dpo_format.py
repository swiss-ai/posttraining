import argparse
import copy
import datasets
from swiss_alignment.data_alignment.linearize_swissaiformat import linearise_sample_for_sft


def flatten_messages(messages):
    """Convert SwissAI parts format to standard HF chat format.

    SwissAI format:  {"role": "user", "parts": [{"content": "...", "type": "text", "metadata": {...}}]}
    HF chat format:  {"role": "user", "content": "..."}
    """
    flat = []
    for msg in messages:
        if "parts" in msg:
            content = "".join(part["content"] for part in msg["parts"])
            flat.append({"role": msg["role"], "content": content})
        elif "content" in msg:
            flat.append({"role": msg["role"], "content": msg["content"]})
        else:
            raise ValueError(f"Message has neither 'parts' nor 'content': {msg}")
    return flat


def convert_row_to_dpo(row):
    branches = row["conversation_branches"]

    if len(branches) < 2:
        return None

    # Branch 0 = chosen, Branch 1 = rejected
    # This is the order set by convert_alignment_dataset_to_swissaiformat.py
    chosen_branch = branches[0]
    rejected_branch = branches[1]

    # Linearize using the same function as prepare_train_dataset_swissaiformat.py
    row_copy = copy.deepcopy(row)

    row_copy["conversation_branches"] = [chosen_branch]
    chosen_chat = linearise_sample_for_sft(row_copy)

    row_copy["conversation_branches"] = [rejected_branch]
    rejected_chat = linearise_sample_for_sft(row_copy)

    # Flatten to standard HF chat format (required by TRL's apply_chat_template)
    chosen_chat = flatten_messages(chosen_chat)
    rejected_chat = flatten_messages(rejected_chat)

    new_row = {
        "chosen": chosen_chat,
        "rejected": rejected_chat,
    }

    # Add ref logprobs if available (used by PreferenceTrainer when ref_logprobs_from_dataset=True)
    if "completion_ref_logprob" in chosen_branch and "completion_ref_logprob" in rejected_branch:
        new_row["ref_chosen_logprob"] = chosen_branch["completion_ref_logprob"]
        new_row["ref_rejected_logprob"] = rejected_branch["completion_ref_logprob"]

    # Preserve prompt_id if available
    if "prompt_id" in row:
        new_row["prompt_id"] = row["prompt_id"]

    return new_row


def main():
    parser = argparse.ArgumentParser(
        description="Convert SwissAI conversation_branches format with ref logprobs to HF DPO format."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input dataset in SwissAI format (with ref logprobs).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the converted dataset. Defaults to input_path + '-dpo-format'.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=64,
        help="Number of processes for dataset.map().",
    )
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = args.input_path.rstrip("/") + "-dpo-format"

    print(f"Loading dataset from {args.input_path}")
    data = datasets.load_from_disk(args.input_path)
    print(data)

    # Verify we have the right format
    first_split = list(data.keys())[0]
    row0 = data[first_split][0]
    assert "conversation_branches" in row0, "Dataset must have 'conversation_branches' column"
    assert len(row0["conversation_branches"]) >= 2, (
        f"Expected at least 2 branches (chosen + rejected), got {len(row0['conversation_branches'])}"
    )
    print(f"Branches per row: {len(row0['conversation_branches'])}")
    print(f"Branch keys: {list(row0['conversation_branches'][0].keys())}")

    for split_name in list(data.keys()):
        print(f"\nProcessing split: {split_name} ({len(data[split_name])} rows)")

        # Filter rows with at least 2 branches
        data[split_name] = data[split_name].filter(
            lambda x: len(x["conversation_branches"]) >= 2,
            num_proc=args.num_proc,
        )

        data[split_name] = data[split_name].map(
            convert_row_to_dpo,
            remove_columns=data[split_name].column_names,
            num_proc=args.num_proc,
            desc=f"Converting {split_name} to DPO format",
        )

        print(f"  Converted: {len(data[split_name])} rows")

    print(f"\nSaving to {args.output_path}")
    data.save_to_disk(args.output_path)
    print(f"Done! {data}")

    # Verify output format
    print("\n--- Verification ---")
    verified = datasets.load_from_disk(args.output_path)
    row = verified[list(verified.keys())[0]][0]
    print(f"Columns: {list(row.keys())}")
    print(f"Chosen[0]: {row['chosen'][0]}")
    print(f"Rejected[0]: {row['rejected'][0]}")
    assert isinstance(row["chosen"][0]["content"], str), "chosen content must be a string"
    assert isinstance(row["rejected"][0]["content"], str), "rejected content must be a string"
    if "ref_chosen_logprob" in row:
        print(f"ref_chosen_logprob: {row['ref_chosen_logprob']}")
        print(f"ref_rejected_logprob: {row['ref_rejected_logprob']}")
    print("Format verified ✅")


if __name__ == "__main__":
    main()