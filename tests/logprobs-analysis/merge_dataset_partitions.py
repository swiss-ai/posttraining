import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer

def find_dataset_partitions(base_path: str):
    """
    Given a base path like
    /.../dataset_name
    return a list of Paths for
    /.../dataset_name_partition0, _partition1, ...
    sorted by the numeric suffix.
    """
    base = Path(base_path)
    parent = base.parent
    name = base.name

    rx = re.compile(rf"^{re.escape(name)}_partition(\d+)$")
    results = []
    for p in parent.glob(name + "_partition*"):
        m = rx.fullmatch(p.name)
        if m:
            results.append((int(m.group(1)), p.resolve()))

    results.sort(key=lambda t: t[0])
    return [p for _, p in results]


import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None,
            config_path="./configs",
            config_name="merge-dataset-partitions")
def main(cfg: DictConfig):
    final_dataset_path = cfg.final_dataset_path
    partition_paths = find_dataset_partitions(final_dataset_path)
    assert len(partition_paths) == 4, f"Wrong number of partitions for {final_dataset_path}"

    print(f"Found {len(partition_paths)} partitions:")
    for p in partition_paths:
        print(f"  {p}")

    # Merge the datasets
    datasets = [load_from_disk(str(p)) for p in partition_paths]
    merged_dataset = concatenate_datasets(datasets)

    # Save the merged dataset
    merged_dataset.save_to_disk(final_dataset_path)
    print(f"Merged dataset saved to {final_dataset_path}")

if __name__ == "__main__":
    main()
