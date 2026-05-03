
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def normalize_messages(value: Any) -> List[Dict[str, Any]]:
    """
    Normalize a message column value into a list of message dicts.

    Handles:
      - list[dict]
      - single dict
      - None
    """
    if value is None:
        return []

    if isinstance(value, dict):
        return [value]

    if isinstance(value, list):
        return value

    # Unexpected format, e.g. string. Treat as no messages.
    return []


def role_breaks_inside(
    messages: Any,
    roles_to_check: Set[str],
) -> List[Dict[str, Any]]:
    """
    Finds broken alternation inside one message list only.

    A break means consecutive checked messages have the same role:

        user -> user
        assistant -> assistant

    Roles outside roles_to_check are ignored.
    """
    messages = normalize_messages(messages)

    breaks = []
    prev_role = None
    prev_idx = None

    for j, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")

        if role not in roles_to_check:
            continue

        if prev_role == role:
            breaks.append(
                {
                    "prev_message_idx": prev_idx,
                    "message_idx": j,
                    "repeated_role": role,
                }
            )

        prev_role = role
        prev_idx = j

    return breaks


def load_any_dataset(input_path: str):
    """
    Loads either:
      - a local dataset saved with save_to_disk
      - a local DatasetDict saved with save_to_disk
      - a Hugging Face dataset name/path via load_dataset
    """
    path = Path(input_path)

    if path.exists():
        return load_from_disk(str(path))

    return load_dataset(input_path)


def find_bad_indices(
    dataset: Dataset,
    columns: Iterable[str],
    roles_to_check: Set[str],
) -> Tuple[Set[int], Dict[str, List[int]], Dict[str, int], List[str]]:
    existing_columns = [col for col in columns if col in dataset.column_names]
    missing_columns = [col for col in columns if col not in dataset.column_names]

    bad_indices: Set[int] = set()
    bad_indices_by_column: Dict[str, List[int]] = {col: [] for col in existing_columns}
    break_counts_by_column: Dict[str, int] = {col: 0 for col in existing_columns}

    for i, example in enumerate(dataset):
        for col in existing_columns:
            breaks = role_breaks_inside(example[col], roles_to_check)

            if breaks:
                bad_indices.add(i)
                bad_indices_by_column[col].append(i)
                break_counts_by_column[col] += len(breaks)

    return bad_indices, bad_indices_by_column, break_counts_by_column, missing_columns


def clean_split(
    dataset: Dataset,
    split_name: str,
    columns: Iterable[str],
    roles_to_check: Set[str],
    max_print_indices: int,
) -> Tuple[Dataset, Dict[str, Any]]:
    bad_indices, bad_by_col, break_counts_by_col, missing_columns = find_bad_indices(
        dataset=dataset,
        columns=columns,
        roles_to_check=roles_to_check,
    )

    bad_indices_sorted = sorted(bad_indices)

    keep_indices = [
        i for i in range(len(dataset))
        if i not in bad_indices
    ]

    cleaned = dataset.select(keep_indices)

    stats = {
        "split": split_name,
        "num_rows_before": len(dataset),
        "num_rows_after": len(cleaned),
        "num_rows_dropped": len(bad_indices_sorted),
        "checked_columns": [col for col in columns if col in dataset.column_names],
        "missing_columns": missing_columns,
        "bad_counts_by_column": {
            col: len(indices)
            for col, indices in bad_by_col.items()
        },
        "break_counts_by_column": break_counts_by_col,
        "bad_indices": bad_indices_sorted,
        "bad_indices_preview": bad_indices_sorted[:max_print_indices],
    }

    return cleaned, stats


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Drop rows where role alternation is broken inside chosen/rejected "
            "message columns."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input dataset path. Can be a local save_to_disk path or HF dataset name.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output path where the cleaned dataset will be saved with save_to_disk.",
    )

    parser.add_argument(
        "--columns",
        nargs="+",
        default=["chosen", "rejected"],
        help="Message columns to check. Default: chosen rejected",
    )

    parser.add_argument(
        "--roles",
        nargs="+",
        default=["user", "assistant"],
        help="Roles to check for alternation. Default: user assistant",
    )

    parser.add_argument(
        "--max-print-indices",
        type=int,
        default=50,
        help="How many bad indices to print per split.",
    )

    parser.add_argument(
        "--stats-json",
        default=None,
        help="Optional path to save full stats JSON. Defaults to <output>/filter_stats.json.",
    )

    args = parser.parse_args()

    roles_to_check = set(args.roles)

    dataset_obj = load_any_dataset(args.input)

    all_stats = {}

    if isinstance(dataset_obj, DatasetDict):
        cleaned_splits = {}

        for split_name, split_dataset in dataset_obj.items():
            cleaned_split, split_stats = clean_split(
                dataset=split_dataset,
                split_name=split_name,
                columns=args.columns,
                roles_to_check=roles_to_check,
                max_print_indices=args.max_print_indices,
            )

            cleaned_splits[split_name] = cleaned_split
            all_stats[split_name] = split_stats

        cleaned_obj = DatasetDict(cleaned_splits)

    elif isinstance(dataset_obj, Dataset):
        cleaned_obj, split_stats = clean_split(
            dataset=dataset_obj,
            split_name="dataset",
            columns=args.columns,
            roles_to_check=roles_to_check,
            max_print_indices=args.max_print_indices,
        )

        all_stats["dataset"] = split_stats

    else:
        raise TypeError(f"Unsupported dataset object type: {type(dataset_obj)}")

    output_path = Path(args.output)
    cleaned_obj.save_to_disk(str(output_path))

    stats_path = Path(args.stats_json) if args.stats_json else output_path / "filter_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Filtering complete")
    print("=" * 80)

    total_before = 0
    total_after = 0
    total_dropped = 0

    for split_name, stats in all_stats.items():
        total_before += stats["num_rows_before"]
        total_after += stats["num_rows_after"]
        total_dropped += stats["num_rows_dropped"]

        print(f"\nSplit: {split_name}")
        print(f"  rows before: {stats['num_rows_before']}")
        print(f"  rows after:  {stats['num_rows_after']}")
        print(f"  dropped:     {stats['num_rows_dropped']}")

        print("  checked columns:", stats["checked_columns"])

        if stats["missing_columns"]:
            print("  missing columns:", stats["missing_columns"])

        print("  bad counts by column:")
        for col, count in stats["bad_counts_by_column"].items():
            print(f"    {col}: {count}")

        print("  break counts by column:")
        for col, count in stats["break_counts_by_column"].items():
            print(f"    {col}: {count}")

        print(
            f"  bad indices preview, first {args.max_print_indices}: "
            f"{stats['bad_indices_preview']}"
        )

    print("\nOverall")
    print(f"  rows before: {total_before}")
    print(f"  rows after:  {total_after}")
    print(f"  dropped:     {total_dropped}")
    print(f"\nSaved cleaned dataset to: {output_path}")
    print(f"Saved stats JSON to:      {stats_path}")


if __name__ == "__main__":
    main()