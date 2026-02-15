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

def plot_logps_mean_vs_var(cfg: DictConfig, ax):
    train_completions_train_logps = load_from_disk(cfg.datasets.trained_model_completions_trained_model_logps)["logprobs"]
    train_completions_ref_logps = load_from_disk(cfg.datasets.trained_model_completions_ref_model_logps)["logprobs"]
    ref_completions_train_logps = load_from_disk(cfg.datasets.ref_model_completions_trained_model_logps)["logprobs"]
    ref_completions_ref_logps = load_from_disk(cfg.datasets.ref_model_completions_ref_model_logps)["logprobs"]

    if cfg.remove_outliers:
        for sample_i in range(len(train_completions_train_logps)):
            train_completions_train_logps[sample_i] = remove_outliers_iqr(train_completions_train_logps[sample_i],
                                                                          k=cfg.outlier_iqr_multiplier)
            train_completions_ref_logps[sample_i] = remove_outliers_iqr(train_completions_ref_logps[sample_i],
                                                                        k=cfg.outlier_iqr_multiplier)
            ref_completions_train_logps[sample_i] = remove_outliers_iqr(ref_completions_train_logps[sample_i],
                                                                        k=cfg.outlier_iqr_multiplier)
            ref_completions_ref_logps[sample_i] = remove_outliers_iqr(ref_completions_ref_logps[sample_i],
                                                                      k=cfg.outlier_iqr_multiplier)

    train_completions_train_logps_mean = np.array([np.mean(x) for x in train_completions_train_logps])
    train_completions_train_logps_var = np.array([np.var(x) for x in train_completions_train_logps])
    ref_completions_train_logps_mean = np.array([np.mean(x) for x in ref_completions_train_logps])
    ref_completions_train_logps_var = np.array([np.var(x) for x in ref_completions_train_logps])


    ax[0].scatter(train_completions_train_logps_mean, train_completions_train_logps_var,
                  alpha=cfg.scatter_alpha, s=cfg.scatter_size, label=f"{cfg.model_type} Model, beta={cfg.beta}")
    # ax[0].scatter(train_completions_ref_logps_mean, train_completions_ref_logps_var,
    #               alpha=cfg.scatter_alpha, s=cfg.scatter_size, label="Reference Model")
    ax[0].set_xlabel('Mean Log Probability')
    ax[0].set_ylabel('Variance of Log Probability')
    ax[0].set_title(cfg.model_type + " completions")
    ax[0].legend()
    ax[1].scatter(ref_completions_train_logps_mean, ref_completions_train_logps_var,
                  alpha=cfg.scatter_alpha, s=cfg.scatter_size, label=f"{cfg.model_type} Model, beta={cfg.beta}")
    # ax[1].scatter(ref_completions_ref_logps_mean, ref_completions_ref_logps_var,
    #               alpha=cfg.scatter_alpha, s=cfg.scatter_size, label="Reference Model")
    ax[1].set_xlabel('Mean Log Probability')
    ax[1].set_ylabel('Variance of Log Probability')
    ax[1].set_title("Reference model completions")
    ax[1].legend()


def plot_logps_trained_vs_ref(cfg: DictConfig, ax):
    train_completions_train_logps = load_from_disk(cfg.datasets.trained_model_completions_trained_model_logps)["logprobs"]
    ref_completions_ref_logps = load_from_disk(cfg.datasets.ref_model_completions_ref_model_logps)["logprobs"]

    if cfg.remove_outliers:
        for sample_i in range(len(train_completions_train_logps)):
            train_completions_train_logps[sample_i] = remove_outliers_iqr(train_completions_train_logps[sample_i],
                                                                          k=cfg.outlier_iqr_multiplier)
            ref_completions_ref_logps[sample_i] = remove_outliers_iqr(ref_completions_ref_logps[sample_i],
                                                                      k=cfg.outlier_iqr_multiplier)

    train_completions_train_logps_mean = np.array([np.mean(x) for x in train_completions_train_logps])
    ref_completions_ref_logps_mean = np.array([np.mean(x) for x in ref_completions_ref_logps])


    ax[0].scatter(ref_completions_ref_logps_mean, train_completions_train_logps_mean,
                  alpha=cfg.scatter_alpha, s=cfg.scatter_size, label=f"beta={cfg.beta}")
    # Identity line
    min_val = min(np.min(ref_completions_ref_logps_mean), np.min(train_completions_train_logps_mean))
    max_val = max(np.max(ref_completions_ref_logps_mean), np.max(train_completions_train_logps_mean))
    ax[0].plot([min_val, max_val], [min_val, max_val], 'r--')
    ax[0].set_ylabel(f'Trained Mean Log Probability')
    ax[0].set_xlabel('Reference Mean Log Probability')
    ax[0].legend()


@hydra.main(version_base=None,
            config_path="./configs",
            config_name="plot-logps-for-betas")
def main(cfg: DictConfig):
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    for beta in cfg.betas:
        cfg.beta = beta
        cfg.datasets.trained_model_completions_trained_model_logps = cfg.datasets.trained_model_completions_trained_model_logps_template.format(beta=beta)
        cfg.datasets.trained_model_completions_ref_model_logps = cfg.datasets.trained_model_completions_ref_model_logps_template.format(beta=beta)
        cfg.datasets.ref_model_completions_trained_model_logps = cfg.datasets.ref_model_completions_trained_model_logps_template.format(beta=beta)
        cfg.datasets.ref_model_completions_ref_model_logps = cfg.datasets.ref_model_completions_ref_model_logps_template.format(beta=beta)
        # plot_logps_mean_vs_var(cfg, ax)
        plot_logps_trained_vs_ref(cfg, ax)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
