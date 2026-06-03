"""Prepare off-policy preference data for the hybrid on/off-policy DPO recipe.

Reads the same source dataset as prep_dolci.py but preserves the chosen and
rejected responses as separate columns so that the training loop can use them
directly without rollout or judge annotation.

Output schema (parquet):
    prompt:             list[{role, content}]   # chosen[:-1]  — only this is used for filtering
    chosen_response:    {role, content}          # chosen[-1]
    rejected_response:  {role, content}          # rejected[-1]
    data_source:        "activeultrafeedback"
    reward_model:       {"ground_truth": ""}     # required by RLHFDataset schema
    extra_info:         {"prompt": <messages>}
"""

import argparse
import os

from datasets import load_from_disk

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train_dolci_offpolicy.parquet")
DATASET_PATH = "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/processing_for_alignment/datasets/ahey/MaxMin_Tr_3600-Filtered-Decontaminated"


def to_offpolicy_row(example):
    chosen = example["chosen"]
    rejected = example["rejected"]
    prompt_msgs = chosen[:-1]
    return {
        "prompt": prompt_msgs,
        "chosen_response": chosen[-1],
        "rejected_response": rejected[-1],
        "data_source": "activeultrafeedback",
        "reward_model": {"ground_truth": ""},
        "extra_info": {"prompt": prompt_msgs},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=None, help="If set, take the first N rows.")
    parser.add_argument("--output", type=str, default=OUT_PATH, help="Output parquet path.")
    args = parser.parse_args()

    ds = load_from_disk(DATASET_PATH)
    if isinstance(ds, dict) or hasattr(ds, "keys"):
        split = list(ds.keys())[0]
        print(f"DatasetDict detected, using split: '{split}'")
        ds = ds[split]
    if args.num_samples is not None:
        ds = ds.select(range(min(args.num_samples, len(ds))))
    keep = ["prompt", "chosen_response", "rejected_response", "data_source", "reward_model", "extra_info"]
    ds = ds.map(to_offpolicy_row, remove_columns=[c for c in ds.column_names if c not in keep])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    ds.to_parquet(args.output)
    print(f"Wrote {len(ds)} rows -> {args.output}")


if __name__ == "__main__":
    main()
