import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk

import hydra
from omegaconf import DictConfig

# --- Outlier removal helper (IQR by default) ---
def remove_outliers_iqr(arr, k=1.5):
    """Return arr with outliers removed using the IQR rule.
    k=1.5 is standard; use 3.0 for more aggressive trimming."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]  # drop NaN/inf if any
    if a.size == 0:
        return a
    q1, q3 = np.percentile(a, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return a  # nothing to trim if all values ~equal
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return a[(a >= lower) & (a <= upper)]

@hydra.main(version_base=None,
            config_path="./configs",
            config_name="plot-logps-dist")
def main(cfg: DictConfig):
    model_type = cfg.model_type
    dataset_completions_logps = load_from_disk(cfg.datasets.trained_model_completions_trained_model_logps)
    dataset_completions_ref_logps = load_from_disk(cfg.datasets.trained_model_completions_ref_model_logps)
    dataset_ref_completions_logps = load_from_disk(cfg.datasets.ref_model_completions_trained_model_logps)
    dataset_ref_completions_ref_logps = load_from_disk(cfg.datasets.ref_model_completions_ref_model_logps)

    completions_logps = dataset_completions_logps[cfg.sample_idx]['logprobs']
    completions_ref_logps = dataset_completions_ref_logps[cfg.sample_idx]['logprobs']
    ref_completions_logps = dataset_ref_completions_logps[cfg.sample_idx]['logprobs']
    ref_completions_ref_logps = dataset_ref_completions_ref_logps[cfg.sample_idx]['logprobs']

    if cfg.remove_outliers:
        completions_logps = remove_outliers_iqr(completions_logps, k=cfg.outlier_iqr_multiplier)
        completions_ref_logps = remove_outliers_iqr(completions_ref_logps, k=cfg.outlier_iqr_multiplier)
        ref_completions_logps = remove_outliers_iqr(ref_completions_logps, k=cfg.outlier_iqr_multiplier)
        ref_completions_ref_logps = remove_outliers_iqr(ref_completions_ref_logps, k=cfg.outlier_iqr_multiplier)

    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    ax[0].hist(completions_logps,
               bins=cfg.num_bins,
               alpha=cfg.alpha,
               label=f'{cfg.model_type} model logprobs')
    ax[0].hist(completions_ref_logps,
               bins=cfg.num_bins,
               alpha=cfg.alpha,
               label='ref model logprobs')
    ax[0].set_title(f'Log probabilities of {model_type} model completions')
    ax[0].set_xlabel('Log probability')
    ax[0].set_ylabel('Frequency')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].hist(ref_completions_logps,
               bins=cfg.num_bins,
               alpha=cfg.alpha,
               label=f'{model_type} model logprobs')
    ax[1].hist(ref_completions_ref_logps,
               bins=cfg.num_bins,
               alpha=cfg.alpha,
               label='ref model logprobs')
    ax[1].set_title(f'Log probabilities of ref model completions')
    ax[1].set_xlabel('Log probability')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    ax[1].grid(True)
    # Global title (applies to the whole figure)
    fig.suptitle(cfg.title.format(sample_idx=cfg.sample_idx), fontsize=16, weight="bold")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
