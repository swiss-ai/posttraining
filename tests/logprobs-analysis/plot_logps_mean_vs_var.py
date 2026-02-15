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

# --- Mean and variance computation with optional clipping ---
def compute_mean_and_var(logps_list, cfg):
    means = np.array([np.mean(x) for x in logps_list])
    if cfg.clip_logps:
        for i in range(len(logps_list)):
            logps_list[i] = np.array(logps_list[i])
            logps_list[i] = logps_list[i][logps_list[i] >= 2 * means[i]]
    vars = np.array([np.var(x) for x in logps_list])
    return means, vars


@hydra.main(version_base=None, 
            config_path="./configs", 
            config_name="plot-logps-mean-vs-var")
def main(cfg: DictConfig):
    train_completions_train_logps = load_from_disk(cfg.datasets.trained_model_completions_trained_model_logps)["logprobs"]
    train_completions_ref_logps = load_from_disk(cfg.datasets.trained_model_completions_ref_model_logps)["logprobs"]
    train_completions_lengths = load_from_disk(cfg.datasets.trained_model_completions_trained_model_logps)["lengths"]
    ref_completions_train_logps = load_from_disk(cfg.datasets.ref_model_completions_trained_model_logps)["logprobs"]
    ref_completions_ref_logps = load_from_disk(cfg.datasets.ref_model_completions_ref_model_logps)["logprobs"]
    ref_completions_lengths = load_from_disk(cfg.datasets.ref_model_completions_ref_model_logps)["lengths"]

    if cfg.remove_outliers:
        for sample_i in range(len(train_completions_train_logps)):
            train_completions_train_logps[sample_i] = remove_outliers_iqr(train_completions_train_logps[sample_i],
                                                                          k=cfg.outlier_iqr_multiplier)
            train_completions_ref_logps[sample_i] = remove_outliers_iqr(train_completions_ref_logps[sample_i],
                                                                        k=cfg.outlier_iqr_multiplier)
            train_completions_lengths[sample_i] = remove_outliers_iqr(train_completions_lengths[sample_i],
                                                                     k=cfg.outlier_iqr_multiplier)
            ref_completions_train_logps[sample_i] = remove_outliers_iqr(ref_completions_train_logps[sample_i],
                                                                        k=cfg.outlier_iqr_multiplier)
            ref_completions_ref_logps[sample_i] = remove_outliers_iqr(ref_completions_ref_logps[sample_i],
                                                                      k=cfg.outlier_iqr_multiplier)
            ref_completions_lengths[sample_i] = remove_outliers_iqr(ref_completions_lengths[sample_i],
                                                                   k=cfg.outlier_iqr_multiplier)

    # train_completions_train_logps_mean = np.array([np.mean(x) for x in train_completions_train_logps])
    # train_completions_train_logps_var = np.array([np.var(x) for x in train_completions_train_logps])
    # train_completions_ref_logps_mean = np.array([np.mean(x) for x in train_completions_ref_logps])
    # train_completions_ref_logps_var = np.array([np.var(x) for x in train_completions_ref_logps])
    train_completions_train_logps_mean, train_completions_train_logps_var = compute_mean_and_var(train_completions_train_logps, cfg)
    train_completions_ref_logps_mean, train_completions_ref_logps_var = compute_mean_and_var(train_completions_ref_logps, cfg)
    train_completions_lengths_mean = np.array([np.mean(x) for x in train_completions_lengths])
    train_completions_lengths_var = np.array([np.var(x) for x in train_completions_lengths])
    # ref_completions_train_logps_mean = np.array([np.mean(x) for x in ref_completions_train_logps])
    # ref_completions_train_logps_var = np.array([np.var(x) for x in ref_completions_train_logps])
    # ref_completions_ref_logps_mean = np.array([np.mean(x) for x in ref_completions_ref_logps])
    # ref_completions_ref_logps_var = np.array([np.var(x) for x in ref_completions_ref_logps])
    ref_completions_train_logps_mean, ref_completions_train_logps_var = compute_mean_and_var(ref_completions_train_logps, cfg)
    ref_completions_ref_logps_mean, ref_completions_ref_logps_var = compute_mean_and_var(ref_completions_ref_logps, cfg)
    ref_completions_lengths_mean = np.array([np.mean(x) for x in ref_completions_lengths])
    ref_completions_lengths_var = np.array([np.var(x) for x in ref_completions_lengths])
    
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    ax[0].scatter(train_completions_train_logps_mean, train_completions_train_logps_var,
                  alpha=cfg.scatter_alpha, s=cfg.scatter_size, label=f"{cfg.model_type} Model")
    # ax[0].scatter(train_completions_ref_logps_mean, train_completions_ref_logps_var,
    #               alpha=cfg.scatter_alpha, s=cfg.scatter_size, label="Reference Model")
    ax[0].set_xlabel('Mean Log Probability')
    ax[0].set_ylabel('Variance of Log Probability')
    ax[0].set_title(cfg.model_type + " completions")
    ax[0].legend()
    ax[1].scatter(ref_completions_train_logps_mean, ref_completions_train_logps_var,
                  alpha=cfg.scatter_alpha, s=cfg.scatter_size, label=f"{cfg.model_type} Model")
    # ax[1].scatter(ref_completions_ref_logps_mean, ref_completions_ref_logps_var,
    #               alpha=cfg.scatter_alpha, s=cfg.scatter_size, label="Reference Model")
    ax[1].set_xlabel('Mean Log Probability')
    ax[1].set_ylabel('Variance of Log Probability')
    ax[1].set_title("Reference model completions")
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(8, 6))
    # plt.scatter(train_completions_train_logps_mean, train_completions_train_logps_var,
    #               alpha=cfg.scatter_alpha, s=cfg.scatter_size, label=f"{cfg.model_type} completions")
    # plt.scatter(ref_completions_train_logps_mean, ref_completions_train_logps_var,
    #               alpha=cfg.scatter_alpha, s=cfg.scatter_size, label=f"ref completions")
    # plt.xlabel('Mean Log Probability')
    # plt.ylabel('Variance of Log Probability')
    # plt.title("Trained model Log Probability")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure(figsize=(8, 6))
    # plt.scatter(train_completions_ref_logps_mean, train_completions_ref_logps_var,
    #             alpha=cfg.scatter_alpha, s=cfg.scatter_size, label=f"{cfg.model_type} completions")
    # plt.scatter(ref_completions_ref_logps_mean, ref_completions_ref_logps_var,
    #             alpha=cfg.scatter_alpha, s=cfg.scatter_size, label=f"ref completions")
    # plt.xlabel('Mean Log Probability')
    # plt.ylabel('Variance of Log Probability')
    # plt.title("ref model Log Probability")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    # ax[0].scatter(train_completions_lengths_mean, train_completions_train_logps_var / 2 - train_completions_train_logps_mean,
    #               alpha=cfg.scatter_alpha, s=cfg.scatter_size, label=f"{cfg.model_type} Model")
    # ax[0].scatter(train_completions_lengths_mean, train_completions_ref_logps_var / 2 - train_completions_ref_logps_mean,
    #               alpha=cfg.scatter_alpha, s=cfg.scatter_size, label="Reference Model")
    # ax[0].set_xlabel('Mean Length')
    # ax[0].set_ylabel('Variance of Log Probability / 2 - Mean Length')
    # ax[0].set_title(cfg.model_type + " completions")
    # ax[0].legend()
    # # ax[0].set_ylim([0, 50000])
    # ax[1].scatter(ref_completions_lengths_mean, ref_completions_train_logps_var / 2 - ref_completions_train_logps_mean,
    #               alpha=cfg.scatter_alpha, s=cfg.scatter_size, label=f"{cfg.model_type} Model")
    # ax[1].scatter(ref_completions_lengths_mean, ref_completions_ref_logps_var / 2 - ref_completions_ref_logps_mean,
    #               alpha=cfg.scatter_alpha, s=cfg.scatter_size, label="Reference Model")
    # ax[1].set_xlabel('Mean Length')
    # ax[1].set_ylabel('Variance of Log Probability / 2 - Mean Length')
    # ax[1].set_title("Reference model completions")
    # ax[1].legend()
    # # ax[1].set_ylim([0, 50000])
    #
    # plt.tight_layout()
    # plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    ax[0].scatter(train_completions_lengths_mean,
                  train_completions_train_logps_mean,
                  alpha=cfg.scatter_alpha, s=cfg.scatter_size, label=f"{cfg.model_type} Model")
    ax[0].scatter(train_completions_lengths_mean,
                  train_completions_ref_logps_mean,
                  alpha=cfg.scatter_alpha, s=cfg.scatter_size, label="Reference Model")
    ax[0].set_xlabel('Mean Length')
    ax[0].set_ylabel('Mean Logp')
    ax[0].set_title(cfg.model_type + " completions")
    ax[0].legend()
    # ax[0].set_ylim([0, 50000])
    ax[1].scatter(ref_completions_lengths_mean, ref_completions_train_logps_mean,
                  alpha=cfg.scatter_alpha, s=cfg.scatter_size, label=f"{cfg.model_type} Model")
    ax[1].scatter(ref_completions_lengths_mean, ref_completions_ref_logps_mean,
                  alpha=cfg.scatter_alpha, s=cfg.scatter_size, label="Reference Model")
    ax[1].set_xlabel('Mean Length')
    ax[1].set_ylabel('Mean Logp')
    ax[1].set_title("Reference model completions")
    ax[1].legend()
    # ax[1].set_ylim([0, 50000])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    