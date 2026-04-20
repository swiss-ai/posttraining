"""Prepare allenai/Dolci-Instruct-DPO into a parquet usable by the SPIN recipe
with the active_ultrafeedback custom reward.

Each row's "chosen" column is a list of chat turns; we take chosen[:-1] as the
prompt conversation (the user/system context the model should respond to) and
drop the assistant's chosen reply.

Output schema:
    prompt:       list[{role, content}]   # consumed by RLHFDataset (prompt_key="prompt")
    data_source:  "activeultrafeedback"   # routes to recipe.spin.active_ultrafeedback_reward
    reward_model: {"ground_truth": ""}    # required by NaiveRewardManager, unused here
    extra_info:   {"prompt": <messages>}  # consumed by compute_score()
"""

import argparse
import os

from datasets import load_dataset

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train_dolci.parquet")
DATASET_ID = "allenai/Dolci-Instruct-DPO"


def to_prompt_row(example):
    chosen = example["chosen"]
    prompt_msgs = chosen[:-1]
    return {
        "prompt": prompt_msgs,
        "data_source": "activeultrafeedback",
        "reward_model": {"ground_truth": ""},
        "extra_info": {"prompt": prompt_msgs},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=None, help="If set, take the first N prompts.")
    parser.add_argument("--output", type=str, default=OUT_PATH, help="Output parquet path.")
    args = parser.parse_args()

    ds = load_dataset(DATASET_ID, split="train")
    if args.num_samples is not None:
        ds = ds.select(range(min(args.num_samples, len(ds))))
    keep = ["prompt", "data_source", "reward_model", "extra_info"]
    ds = ds.map(to_prompt_row, remove_columns=[c for c in ds.column_names if c not in keep])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    ds.to_parquet(args.output)
    print(f"Wrote {len(ds)} rows -> {args.output}")


if __name__ == "__main__":
    main()
