from datasets import load_from_disk, load_dataset
dataset = load_from_disk("/iopsstor/scratch/cscs/dmelikidze/posttraining-data/processing_for_alignment/datasets/ahey/MaxMin_Tr_3600-Filtered-Decontaminated")
dataset = dataset["train_split"]
print(dataset)
dataset_orig = load_dataset("allenai/Dolci-Instruct-DPO")
dataset_orig = dataset_orig["train"]
print(dataset_orig)
index = 153153
print(dataset["original_index"][index])
orig_index = dataset["original_index"][index]

print(dataset["chosen"][index])
print("\n\n----------------------\n\n")
print(dataset_orig["chosen"][orig_index])

exit()

"""Quick inspection of the local Dolci parquet dataset."""

import json
import os

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train_dolci.parquet")

df = pd.read_parquet(DATA_PATH)

print("=" * 60)
print("DATASET INFO")
print("=" * 60)
print(f"Source:       {DATA_PATH}")
print(f"Num rows:     {len(df)}")
print(f"Columns:      {list(df.columns)}")
print(f"\nDtypes:\n{df.dtypes}")

print("\n" + "=" * 60)
print("FIRST 3 EXAMPLES")
print("=" * 60)
for i in range(min(3, len(df))):
    print(f"\n--- Example {i} ---")
    row = df.iloc[i]
    for col in df.columns:
        val = row[col]
        val_str = json.dumps(val, ensure_ascii=False, indent=2, cls=NumpyEncoder) if isinstance(val, (dict, list, np.ndarray)) else repr(val)
        if len(val_str) > 500:
            val_str = val_str[:500] + "... [truncated]"
        print(f"\n  [{col}]:\n{val_str}")

print("\n" + "=" * 60)
print("COLUMN STATS")
print("=" * 60)
for col in df.columns:
    sample = df.iloc[0][col]
    col_type = type(sample).__name__
    print(f"\n  {col}:")
    print(f"    Python type:  {col_type}")
    print(f"    Pandas dtype: {df[col].dtype}")
    print(f"    Nulls:        {df[col].isna().sum()}")
    if isinstance(sample, np.ndarray):
        lengths = df[col].head(100).apply(len)
        print(f"    Array lengths (first 100): min={lengths.min()}, max={lengths.max()}, avg={lengths.mean():.1f}")
        if len(sample) > 0 and isinstance(sample[0], dict):
            print(f"    Element keys: {list(sample[0].keys())}")
    elif isinstance(sample, list):
        lengths = df[col].head(100).apply(len)
        print(f"    List lengths (first 100): min={lengths.min()}, max={lengths.max()}, avg={lengths.mean():.1f}")
        if lengths.iloc[0] > 0 and isinstance(sample[0], dict):
            print(f"    Element keys: {list(sample.keys())}")
    elif isinstance(sample, dict):
        print(f"    Dict keys:    {list(sample.keys())}")
    elif isinstance(sample, str):
        str_lengths = df[col].head(100).str.len()
        print(f"    Str lengths (first 100): min={str_lengths.min()}, max={str_lengths.max()}, avg={str_lengths.mean():.1f}")
