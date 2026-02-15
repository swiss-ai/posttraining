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

# --- Main plotting function ---
@hydra.main(version_base=None,
            config_path="./configs",
            config_name="plot-completion-length-hist")
def main(cfg: DictConfig):
    train_completions_lengths = np.array(load_from_disk(cfg.datasets.trained_model_completions_trained_model_logps)[cfg.sample_id]["lengths"])
    ref_completions_lengths = np.array(load_from_disk(cfg.datasets.ref_model_completions_ref_model_logps)[cfg.sample_id]["lengths"])

    if cfg.remove_outliers:
        train_completions_lengths = remove_outliers_iqr(train_completions_lengths, k=cfg.outlier_k)
        ref_completions_lengths = remove_outliers_iqr(ref_completions_lengths, k=cfg.outlier_k)

    plt.figure(figsize=(10, 6))
    plt.hist(train_completions_lengths, bins=cfg.num_bins, alpha=cfg.alpha, label=f'{cfg.model_type} Model Completions')
    plt.hist(ref_completions_lengths, bins=cfg.num_bins, alpha=cfg.alpha, label='Reference Model Completions')
    plt.xlabel('Completion Length')
    plt.ylabel('Density')
    plt.title(f'Prompt sample {cfg.sample_id}')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
