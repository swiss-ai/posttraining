import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None,
            config_path="./configs",
            config_name="plot-dpo-rewards")
def main(cfg: DictConfig):
    model_type = cfg.model_type
    dataset_completions_logps = load_from_disk(cfg.datasets.trained_model_completions_trained_model_logps)
    dataset_completions_ref_logps = load_from_disk(cfg.datasets.trained_model_completions_ref_model_logps)
    dataset_ref_completions_logps = load_from_disk(cfg.datasets.ref_model_completions_trained_model_logps)
    dataset_ref_completions_ref_logps = load_from_disk(cfg.datasets.ref_model_completions_ref_model_logps)

    if cfg.add_beta:
        if cfg.use_lengthnorm_beta:
            completions_logps = np.array(dataset_completions_logps[cfg.sample_idx]["lengthnorm_beta_scaled_logprobs"])
            completions_ref_logps = np.array(dataset_completions_ref_logps[cfg.sample_idx]["lengthnorm_beta_scaled_logprobs"])
            ref_completions_logps = np.array(dataset_ref_completions_logps[cfg.sample_idx]["lengthnorm_beta_scaled_logprobs"])
            ref_completions_ref_logps = np.array(dataset_ref_completions_ref_logps[cfg.sample_idx]["lengthnorm_beta_scaled_logprobs"])
        else:
            completions_logps = np.array(dataset_completions_logps[cfg.sample_idx]["beta_scaled_logprobs"])
            completions_ref_logps = np.array(dataset_completions_ref_logps[cfg.sample_idx]["beta_scaled_logprobs"])
            ref_completions_logps = np.array(dataset_ref_completions_logps[cfg.sample_idx]["beta_scaled_logprobs"])
            ref_completions_ref_logps = np.array(dataset_ref_completions_ref_logps[cfg.sample_idx]["beta_scaled_logprobs"])
    else:
        completions_logps = np.array(dataset_completions_logps[cfg.sample_idx]["logprobs"])
        completions_ref_logps = np.array(dataset_completions_ref_logps[cfg.sample_idx]["logprobs"])
        ref_completions_logps = np.array(dataset_ref_completions_logps[cfg.sample_idx]["logprobs"])
        ref_completions_ref_logps = np.array(dataset_ref_completions_ref_logps[cfg.sample_idx]["logprobs"])

    trained_model_dpo_rewards = completions_logps - completions_ref_logps
    ref_model_dpo_rewards = ref_completions_logps - ref_completions_ref_logps

    # Plot the rewards hists
    plt.figure(figsize=(10, 6))
    plt.hist(trained_model_dpo_rewards, bins=cfg.num_bins, alpha=cfg.alpha, label=f'{model_type} model')
    plt.hist(ref_model_dpo_rewards, bins=cfg.num_bins, alpha=cfg.alpha, label='Reference model')
    if cfg.xlim:
        plt.xlim(cfg.xlim)
    if cfg.ylim:
        plt.ylim(cfg.ylim)
    plt.xlabel(cfg.xlabel)
    plt.ylabel(cfg.ylabel)
    plt.title(cfg.title.format(sample_idx=cfg.sample_idx))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
