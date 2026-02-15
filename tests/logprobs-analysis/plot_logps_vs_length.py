import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk

import hydra
from omegaconf import DictConfig

# --- Outlier removal helper (IQR by default) ---
def remove_outliers_iqr(main_ar, ar1, ar2, k=1.5):
    """Return (main_ar_filtered, ar1_filtered, ar2_filtered) with joint outliers removed.
    k=1.5 is standard; use 3.0 for gentler trimming."""
    main_ar = np.asarray(main_ar, dtype=float)
    ar1 = np.asarray(ar1, dtype=float)
    ar2 = np.asarray(ar2, dtype=float)

    finite = np.isfinite(main_ar)
    main_ar_f, ar1_f, ar2_f = main_ar[finite], ar1[finite], ar2[finite]
    if main_ar_f.size == 0:
        return main_ar_f, ar1_f, ar2_f
    # IQR for main_ar
    q1_main_ar, q3_main_ar = np.percentile(main_ar_f, [25, 75]);
    iqr_main_ar = q3_main_ar - q1_main_ar
    if iqr_main_ar == 0:
        mask_main_ar = np.ones_like(main_ar_f, dtype=bool)
    else:
        lower_main_ar, upper_main_ar = q1_main_ar - k * iqr_main_ar, q3_main_ar + k * iqr_main_ar
        mask_main_ar = (main_ar_f >= lower_main_ar) & (main_ar_f <= upper_main_ar)
    # IQR for ar1
    q1_ar1, q3_ar1 = np.percentile(ar1, (25, 75))
    iqr_ar1 = q3_ar1 - q1_ar1
    if iqr_ar1 == 0:
        mask_ar1 = np.ones_like(ar1_f, dtype=bool)
    else:
        lower_ar1, upper_ar1 = q1_ar1 - k * iqr_ar1, q3_ar1 + k * iqr_ar1
        mask_ar1 = (ar1_f >= lower_ar1) & (ar1_f <= upper_ar1)
    # IQR for ar2
    q1_ar2, q3_ar2 = np.percentile(ar2, (25, 75))
    iqr_ar2 = q3_ar2 - q1_ar2
    if iqr_ar2 == 0:
        mask_ar2 = np.ones_like(ar2_f, dtype=bool)
    else:
        lower_ar2, upper_ar2 = q1_ar2 - k * iqr_ar2, q3_ar2 + k * iqr_ar2
        mask_ar2 = (ar2_f >= lower_ar2) & (ar2_f <= upper_ar2)

    mask = mask_main_ar & mask_ar1 & mask_ar2

    return main_ar_f[mask], ar1_f[mask], ar2_f[mask]

# --- Main plotting function ---
@hydra.main(version_base=None,
            config_path="./configs",
            config_name="plot-logps-vs-length")
def main(cfg: DictConfig):
    train_completions_train_logps = np.array(load_from_disk(cfg.datasets.trained_model_completions_trained_model_logps)[cfg.sample_id]["logprobs"])
    train_completions_ref_logps = np.array(load_from_disk(cfg.datasets.trained_model_completions_ref_model_logps)[cfg.sample_id]["logprobs"])
    train_completions_lengths = np.array(load_from_disk(cfg.datasets.trained_model_completions_trained_model_logps)[cfg.sample_id]["lengths"])
    ref_completions_train_logps = np.array(load_from_disk(cfg.datasets.ref_model_completions_trained_model_logps)[cfg.sample_id]["logprobs"])
    ref_completions_ref_logps = np.array(load_from_disk(cfg.datasets.ref_model_completions_ref_model_logps)[cfg.sample_id]["logprobs"])
    ref_completions_lengths = np.array(load_from_disk(cfg.datasets.ref_model_completions_ref_model_logps)[cfg.sample_id]["lengths"])

    if cfg.remove_outliers:
        train_completions_lengths, train_completions_ref_logps, train_completions_train_logps = remove_outliers_iqr(
            train_completions_lengths, train_completions_ref_logps, train_completions_train_logps, k=cfg.outlier_k)
        ref_completions_lengths, ref_completions_ref_logps, ref_completions_train_logps = remove_outliers_iqr(
            ref_completions_lengths, ref_completions_ref_logps, ref_completions_train_logps, k=cfg.outlier_k)

    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    # plot log probabilities vs length for both models and both datasets
    ax[0, 0].scatter(train_completions_lengths, train_completions_ref_logps,
                  label='Reference Model Log Probs',
                  alpha=cfg.scatter_alpha,
                  s=cfg.scatter_size)
    ax[0, 0].scatter(train_completions_lengths, train_completions_train_logps,
                  label=f'{cfg.model_type} Model Log Probs',
                  alpha=cfg.scatter_alpha,
                  s=cfg.scatter_size)
    ax[0, 0].legend()
    ax[0, 0].set_title(f'Completions from {cfg.model_type} Model')
    ax[0, 0].set_xlabel('Completion Length')
    ax[0, 0].set_ylabel('Log Probability')

    ax[0, 1].scatter(ref_completions_lengths, ref_completions_ref_logps,
                  label='Reference Model Log Probs',
                  alpha=cfg.scatter_alpha,
                  s=cfg.scatter_size)
    ax[0, 1].scatter(ref_completions_lengths, ref_completions_train_logps,
                  label=f'{cfg.model_type} Model Log Probs',
                  alpha=cfg.scatter_alpha,
                  s=cfg.scatter_size)
    ax[0, 1].legend()
    ax[0, 1].set_title('Completions from Reference Model')
    ax[0, 1].set_xlabel('Completion Length')
    ax[0, 1].set_ylabel('Log Probability')

    # plot length-normalized log probabilities vs length for both models and both datasets
    ax[1, 0].scatter(train_completions_lengths, train_completions_ref_logps / train_completions_lengths,
                  label='Reference Model Length-Normalized Log Probs',
                  alpha=cfg.scatter_alpha,
                  s=cfg.scatter_size)
    ax[1, 0].scatter(train_completions_lengths, train_completions_train_logps / train_completions_lengths,
                  label=f'{cfg.model_type} Model Length-Normalized Log Probs',
                  alpha=cfg.scatter_alpha,
                  s=cfg.scatter_size)
    ax[1, 0].legend()
    ax[1, 0].set_title(f'Completions from {cfg.model_type} Model')
    ax[1, 0].set_xlabel('Completion Length')
    ax[1, 0].set_ylabel('Length-Normalized Log Probability')

    ax[1, 1].scatter(ref_completions_lengths, ref_completions_ref_logps / ref_completions_lengths,
                  label='Reference Model Length-Normalized Log Probs',
                  alpha=cfg.scatter_alpha,
                  s=cfg.scatter_size)
    ax[1, 1].scatter(ref_completions_lengths, ref_completions_train_logps / ref_completions_lengths,
                  label=f'{cfg.model_type} Model Length-Normalized Log Probs',
                  alpha=cfg.scatter_alpha,
                  s=cfg.scatter_size)
    ax[1, 1].legend()
    ax[1, 1].set_title('Completions from Reference Model')
    ax[1, 1].set_xlabel('Completion Length')
    ax[1, 1].set_ylabel('Length-Normalized Log Probability')

    # overall bolded title
    fig.suptitle(f"Prompt sample {cfg.sample_id}", fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
