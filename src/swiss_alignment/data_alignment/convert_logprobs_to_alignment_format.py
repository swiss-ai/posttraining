import copy
import datasets


def flatten_branch_to_chat(branch):
    """Convert a SwissAI conversation branch to standard HF chat format.
    
    Extracts messages directly from the branch, concatenating all parts
    into a single content string per message.
    """
    flat = []
    for msg in branch["messages"]:
        content = "".join(part["content"] for part in msg["parts"])
        flat.append({"role": msg["role"], "content": content})
    return flat


def convert_row_to_dpo(row):
    branches = row["conversation_branches"]

    # Branch 0 = chosen, Branch 1 = rejected
    # This is the order from convert_alignment_dataset_to_swissaiformat.py
    chosen = branches[0]
    rejected = branches[1]

    chosen_chat = flatten_branch_to_chat(chosen)
    rejected_chat = flatten_branch_to_chat(rejected)

    new_row = {
        "prompt_id": row.get("prompt_id", ""),
        "chosen": chosen_chat,
        "rejected": rejected_chat,
        "ref_chosen_logprob": chosen["completion_ref_logprob"],
        "ref_rejected_logprob": rejected["completion_ref_logprob"],
    }

    return new_row


def main():
    dataset_path = "/users/dmelikidze/projects/posttraining/run/artifacts/shared/datasets/alignment-pipeline-swissaiformat/datasets-with-ref-logprobs/merged/dolci-instruct-dpo-regenerated-qwen06-qwen32-apertus-8b-sft-10T-mixture-7-7fea1f8c44336360-maxlen4096-Nref10-logprobs"

    print(f"Loading dataset from {dataset_path}")
    data = datasets.load_from_disk(dataset_path)
    print(data)

    # Verify branch order is preserved
    row0 = data["train_split"][0]
    print(f"Number of branches: {len(row0['conversation_branches'])}")
    assert len(row0["conversation_branches"]) == 2, "Expected exactly 2 branches (chosen + rejected)"

    for split_name in list(data.keys()):
        data[split_name] = data[split_name].map(
            convert_row_to_dpo,
            remove_columns=data[split_name].column_names,
            num_proc=64,
        )

    save_path = dataset_path + "-dpo-format"
    print(f"Saving to {save_path}")
    data.save_to_disk(save_path)
    print(f"Done! {data}")

    # Verify output format
    print("\n--- Verification ---")
    verified = datasets.load_from_disk(save_path)
    row = verified[list(verified.keys())[0]][0]
    print(f"Columns: {list(row.keys())}")
    print(f"Chosen[0]: {row['chosen'][0]}")
    print(f"Content type: {type(row['chosen'][0]['content'])}")
    assert isinstance(row["chosen"][0]["content"], str), "Content should be a string!"
    assert isinstance(row["rejected"][0]["content"], str), "Content should be a string!"
    print(f"ref_chosen_logprob: {row['ref_chosen_logprob']}")
    print(f"ref_rejected_logprob: {row['ref_rejected_logprob']}")
    print("Format verified ✅")


if __name__ == "__main__":
    main()