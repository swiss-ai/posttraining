import argparse
import json
from datasets import load_from_disk


TARGET_COLUMNS = [
    "prompt_id",
    "prompt_messages",
    "ref_rewards",
    "offline_trajectories",
    "offline_rewards",
    "tools",
]


def as_messages(value):
    """
    Handles both:
      - JSON string: '[{"role": "user", "content": "..."}]'
      - already-parsed list[dict]
    """
    if isinstance(value, str):
        return json.loads(value)
    return value


def same_message(a, b):
    return (
        isinstance(a, dict)
        and isinstance(b, dict)
        and a.get("role") == b.get("role")
        and a.get("content") == b.get("content")
    )


def strip_prompt(full_conversation, prompt_messages):
    """
    chosen/rejected look like:
        [prompt messages..., assistant response...]

    We want only:
        [assistant response...]
    """
    full_conversation = as_messages(full_conversation)
    prompt_messages = as_messages(prompt_messages)

    n_prompt = len(prompt_messages)

    # Preferred path: remove exact prompt prefix.
    if len(full_conversation) >= n_prompt:
        prefix_matches = all(
            same_message(full_conversation[i], prompt_messages[i])
            for i in range(n_prompt)
        )
        if prefix_matches:
            return full_conversation[n_prompt:]

    # Fallback: keep messages starting from the first assistant message.
    for i, msg in enumerate(full_conversation):
        if msg.get("role") == "assistant":
            return full_conversation[i:]

    return []


def convert_batch(batch, indices):
    prompt_ids = []
    prompt_messages_out = []
    offline_trajectories = []
    offline_rewards = []
    tools = []

    for i in range(len(indices)):
        prompt_messages = as_messages(batch["prompt_messages"][i])

        chosen_traj = strip_prompt(batch["chosen"][i], prompt_messages)
        rejected_traj = strip_prompt(batch["rejected"][i], prompt_messages)

        prompt_ids.append(str(indices[i]))
        prompt_messages_out.append(prompt_messages)

        offline_trajectories.append([
            chosen_traj,
            rejected_traj,
        ])

        offline_rewards.append([
            float(batch["chosen_reward"][i]),
            float(batch["rejected_reward"][i]),
        ])

        tools.append(None)

    return {
        "prompt_id": prompt_ids,
        "prompt_messages_normalized": prompt_messages_out,
        "offline_trajectories": offline_trajectories,
        "offline_rewards": offline_rewards,
        "tools": tools,
    }


def convert_split(ds, keep_extra_columns=False):
    ds = ds.map(
        convert_batch,
        batched=True,
        with_indices=True,
        desc="Converting dataset",
    )

    # Replace original prompt_messages with normalized list-of-dicts version.
    if "prompt_messages" in ds.column_names:
        ds = ds.remove_columns(["prompt_messages"])

    ds = ds.rename_column("prompt_messages_normalized", "prompt_messages")

    if not keep_extra_columns:
        keep = [col for col in TARGET_COLUMNS if col in ds.column_names]
        remove = [col for col in ds.column_names if col not in keep]
        ds = ds.remove_columns(remove)

        # Reorder columns.
        ds = ds.select_columns(TARGET_COLUMNS)

    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to dataset saved with save_to_disk()")
    parser.add_argument("--output", required=True, help="Where to save converted dataset")
    parser.add_argument(
        "--keep-extra-columns",
        action="store_true",
        help="Keep original columns like chosen/rejected/chosen_reward/etc.",
    )
    args = parser.parse_args()

    dataset = load_from_disk(args.input)

    if hasattr(dataset, "keys"):  # DatasetDict
        converted = dataset.map(
            lambda x: x,
            batched=True,
        )
        for split in dataset.keys():
            converted[split] = convert_split(
                dataset[split],
                keep_extra_columns=args.keep_extra_columns,
            )
    else:  # plain Dataset
        converted = convert_split(
            dataset,
            keep_extra_columns=args.keep_extra_columns,
        )

    converted.save_to_disk(args.output)

    print(converted)
    print(f"Saved converted dataset to: {args.output}")


if __name__ == "__main__":
    main()
