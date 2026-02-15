import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
import hydra
from omegaconf import DictConfig


# ---------- helpers: robust KDE + Gaussian options ----------

def _iqr(x):
    q75, q25 = np.percentile(x, [75, 25])
    return q75 - q25

def _scott_bw(x):
    x = np.asarray(x)
    n = max(len(x), 1)
    s = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if s <= 0:
        i = _iqr(x)
        s = (i / 1.34) if i > 0 else 1.0
    return s * (n ** (-1.0 / 5.0))

def _kde_fixed(x, grid, bw):
    x = np.asarray(x, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n = len(x)
    if n == 0:
        return np.zeros_like(grid)
    bw = max(float(bw), 1e-8)
    u = (grid[:, None] - x[None, :]) / bw
    pdf = np.exp(-0.5 * u * u).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))
    area = np.trapz(pdf, grid)
    if area > 0:
        pdf /= area
    return pdf

def _kde_adaptive(x, grid, base_bw, alpha=0.5, eps=1e-12):
    # Abramson rule
    x = np.asarray(x, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n = len(x)
    if n == 0:
        return np.zeros_like(grid)

    f0 = _kde_fixed(x, x, base_bw) + eps
    g = np.exp(np.mean(np.log(f0)))
    h_i = base_bw * (g / f0) ** alpha
    U = (grid[:, None] - x[None, :]) / h_i[None, :]
    pdf = (np.exp(-0.5 * U * U) / (h_i[None, :] * np.sqrt(2 * np.pi))).mean(axis=1)
    area = np.trapz(pdf, grid)
    if area > 0:
        pdf /= area
    return pdf

def _ref_pdf_smooth(x_ref, x_grid, mode, bw_scale, adaptive_alpha):
    base = _scott_bw(x_ref)
    bw = max(base * float(bw_scale), 1e-8)
    if mode == "adaptive":
        return _kde_adaptive(x_ref, x_grid, base_bw=bw, alpha=float(adaptive_alpha))
    elif mode == "fixed":
        return _kde_fixed(x_ref, x_grid, bw=bw)
    elif mode == "gaussian":
        mu = float(np.mean(x_ref))
        sigma = float(np.std(x_ref, ddof=1)) if len(x_ref) > 1 else 1.0
        return _gaussian_pdf(x_grid, mu, sigma)
    else:
        raise ValueError("ref_density_mode must be one of: fixed | adaptive | gaussian")

def _gaussian_pdf(grid, mu, sigma):
    sigma = max(float(sigma), 1e-12)
    z = (np.asarray(grid) - mu) / sigma
    return np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * sigma)


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


# ---------- main ----------

@hydra.main(version_base=None, config_path="./configs", config_name="plot-dpo-rewards")
def main(cfg: DictConfig):
    model_type = cfg.model_type

    # Load datasets
    dataset_completions_logps = load_from_disk(cfg.datasets.trained_model_completions_trained_model_logps)
    dataset_completions_ref_logps = load_from_disk(cfg.datasets.trained_model_completions_ref_model_logps)
    dataset_ref_completions_logps = load_from_disk(cfg.datasets.ref_model_completions_trained_model_logps)
    dataset_ref_completions_ref_logps = load_from_disk(cfg.datasets.ref_model_completions_ref_model_logps)

    # Choose which logps to use
    if cfg.add_beta:
        if cfg.use_lengthnorm_beta:
            key = "lengthnorm_beta_scaled_logprobs"
        else:
            key = "beta_scaled_logprobs"
    else:
        key = "logprobs"

    completions_logps = np.array(dataset_completions_logps[cfg.sample_idx][key])
    completions_ref_logps = np.array(dataset_completions_ref_logps[cfg.sample_idx][key])
    ref_completions_logps = np.array(dataset_ref_completions_logps[cfg.sample_idx][key])
    ref_completions_ref_logps = np.array(dataset_ref_completions_ref_logps[cfg.sample_idx][key])

    if cfg.remove_outliers:
        completions_logps, completions_ref_logps = remove_outliers_iqr(completions_logps,
                                                                       completions_ref_logps,
                                                                       k=cfg.outlier_k)
        ref_completions_logps, ref_completions_ref_logps = remove_outliers_iqr(ref_completions_logps,
                                                                               ref_completions_ref_logps,
                                                                               k=cfg.outlier_k)

    # DPO rewards
    trained_model_dpo_rewards = completions_logps - completions_ref_logps
    ref_model_dpo_rewards = ref_completions_logps - ref_completions_ref_logps

    # Common bins across both distributions (your correction)
    bins = np.histogram_bin_edges(
        np.concatenate((ref_model_dpo_rewards, trained_model_dpo_rewards), axis=0),
        bins=cfg.num_bins,
    )
    bin_widths = np.diff(bins)
    mean_bin_width = float(np.mean(bin_widths)) if len(bin_widths) else 1.0

    # Plot histograms
    plt.figure(figsize=(10, 6))
    plt.hist(
        trained_model_dpo_rewards,
        bins=bins,
        alpha=cfg.alpha,
        label=f"{model_type} model",
        density=cfg.hist_density,
    )
    plt.hist(
        ref_model_dpo_rewards,
        bins=bins,
        alpha=cfg.alpha,
        label="Reference model",
        density=cfg.hist_density,
    )

    # Grid for smooth lines (pad a bit to avoid boundary artifacts)
    x_min, x_max = bins[0], bins[-1]
    pad = 0.05 * max(x_max - x_min, 1.0)
    x_grid = np.linspace(x_min - pad, x_max + pad, int(cfg.kde_grid_points))

    # ----- Reference smooth line -----
    ref_pdf = None
    if cfg.show_ref_density_line:
        ref_pdf = _ref_pdf_smooth(
            x_ref=ref_model_dpo_rewards,
            x_grid=x_grid,
            mode=cfg.ref_density_mode,           # fixed | adaptive | gaussian
            bw_scale=cfg.kde_bw_scale,           # <1 -> sharper / lighter tails
            adaptive_alpha=cfg.kde_adaptive_alpha,
        )
        if cfg.hist_density:
            y_ref_line = ref_pdf
        else:
            # scale to counts height (so line sits on top of bars)
            N_ref = len(ref_model_dpo_rewards)
            y_ref_line = ref_pdf * N_ref * mean_bin_width
        plt.plot(x_grid, y_ref_line, label="Reference model dist")

    # ----- Predicted trained line = ref_pdf * exp(tilt_sign * x / beta), renorm -----
    if cfg.show_predicted_trained_line:
        beta = float(cfg.beta)
        if beta == 0.0:
            raise ValueError("beta must be non-zero.")

        if ref_pdf is None:
            # build it even if green line is hidden
            ref_pdf = _ref_pdf_smooth(
                x_ref=ref_model_dpo_rewards,
                x_grid=x_grid,
                mode=cfg.ref_density_mode,
                bw_scale=cfg.kde_bw_scale,
                adaptive_alpha=cfg.kde_adaptive_alpha,
            )

        if cfg.predicted_mode == "kde_tilt":
            lam = float(cfg.tilt_sign) / beta
            s = x_grid * lam
            s = s - np.max(s)  # numerical stability
            pred_pdf_unnorm = ref_pdf * np.exp(s)
            Z = np.trapz(pred_pdf_unnorm, x_grid)
            pred_pdf = pred_pdf_unnorm / max(Z, 1e-12)
        elif cfg.predicted_mode == "gaussian_shift":
            mu = float(np.mean(ref_model_dpo_rewards))
            sigma = float(np.std(ref_model_dpo_rewards, ddof=1)) if len(ref_model_dpo_rewards) > 1 else 1.0
            lam = float(cfg.tilt_sign) / beta
            mu_shifted = mu + (sigma ** 2) * lam  # exact for Normal
            pred_pdf = _gaussian_pdf(x_grid, mu_shifted, sigma)
        else:
            raise ValueError("predicted_mode must be one of: kde_tilt | gaussian_shift")

        if cfg.hist_density:
            y_pred_line = pred_pdf
        else:
            N_ref = len(ref_model_dpo_rewards)
            y_pred_line = pred_pdf * N_ref * mean_bin_width

        plt.plot(x_grid, y_pred_line, linestyle="--", label="Predicted trained dist")

    # Cosmetics
    if cfg.xlim:
        plt.xlim(cfg.xlim)
    if cfg.ylim:
        plt.ylim(cfg.ylim)
    plt.xlabel(cfg.xlabel)
    plt.ylabel(cfg.ylabel if not cfg.hist_density else "Density")
    plt.title(cfg.title.format(sample_idx=cfg.sample_idx))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()