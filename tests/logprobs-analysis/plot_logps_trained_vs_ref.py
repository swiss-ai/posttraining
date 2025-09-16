import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk

import hydra
from omegaconf import DictConfig

# --- Outlier removal helper (IQR by default) ---
def remove_outliers_iqr(x, y, k=1.5):
    """Return (x_filtered, y_filtered) with joint outliers removed using per-axis IQR.
    k=1.5 is standard; use 3.0 for gentler trimming."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)  # keep pairs aligned
    x_f, y_f = x[finite], y[finite]
    if x_f.size == 0:
        return x_f, y_f
    # IQR for x
    q1x, q3x = np.percentile(x_f, [25, 75]); iqr_x = q3x - q1x
    if iqr_x == 0:
        within_x = np.ones_like(x_f, dtype=bool)
    else:
        lower_x, upper_x = q1x - k * iqr_x, q3x + k * iqr_x
        within_x = (x_f >= lower_x) & (x_f <= upper_x)
    # IQR for y
    q1y, q3y = np.percentile(y_f, [25, 75]); iqr_y = q3y - q1y
    if iqr_y == 0:
        within_y = np.ones_like(y_f, dtype=bool)
    else:
        lower_y, upper_y = q1y - k * iqr_y, q3y + k * iqr_y
        within_y = (y_f >= lower_y) & (y_f <= upper_y)
    mask = within_x & within_y
    return x_f[mask], y_f[mask]

@hydra.main(version_base=None,
            config_path="./configs",
            config_name="plot-logps-trained-vs-ref")
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
        completions_ref_logps, completions_logps = remove_outliers_iqr(completions_ref_logps,
                                                                       completions_logps,
                                                                       k=cfg.outlier_iqr_multiplier)
        ref_completions_ref_logps, ref_completions_logps = remove_outliers_iqr(ref_completions_ref_logps,
                                                                               ref_completions_logps,
                                                                               k=cfg.outlier_iqr_multiplier)

    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    ax[0].scatter(completions_ref_logps,
                  completions_logps,
                  alpha=cfg.scatter_alpha,
                  s=cfg.scatter_size)
    # add dashed y=x line
    min_val = min(min(completions_ref_logps), min(completions_logps))
    max_val = max(max(completions_ref_logps), max(completions_logps))
    ax[0].plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, label='y=x')
    ax[0].set_title(f'Log probabilities of {model_type} model completions')
    ax[0].set_xlabel('ref model')
    ax[0].set_ylabel(f'{model_type} model')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].scatter(ref_completions_ref_logps,
                  ref_completions_logps,
                  alpha=cfg.scatter_alpha,
                  s=cfg.scatter_size)
    # add dashed y=x line
    min_val = min(min(ref_completions_ref_logps), min(ref_completions_logps))
    max_val = max(max(ref_completions_ref_logps), max(ref_completions_logps))
    ax[1].plot([min_val, max_val], [min_val, max_val], 'k--', lw=1, label='y=x')
    ax[1].set_title(f'Log probabilities of ref model completions')
    ax[1].set_xlabel('ref model')
    ax[1].set_ylabel(f'{model_type} model')
    ax[1].legend()
    ax[1].grid(True)
    # Global title (applies to the whole figure)
    fig.suptitle(cfg.title.format(sample_idx=cfg.sample_idx), fontsize=16, weight="bold")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
