#!/usr/bin/env python3
"""Compute and plot per-token log-ratio (policy vs ref) from wandb runs.

Supports four metric formats:
  - "direct": runs that log `train/kl/policy_vs_ref/mean` (preference.py, spin_trainer.py)
  - "verl":   logs `actor/policy_chosen_logps` etc. — sequence-level, normalized by response_length/mean
  - "hf":     logs `train/no-beta-dpo-rewards/chosen` — already per-token
  - "trl":    logs `train/kl/overall` — sequence-level, normalized by train/loss/lens
"""

import argparse

import matplotlib.pyplot as plt
import wandb


KEYS_DIRECT = [
    "train/kl/policy_vs_ref/mean",
]

KEYS_VERL = [
    "actor/policy_chosen_logps",
    "actor/policy_rejected_logps",
    "actor/reference_chosen_logps",
    "actor/reference_rejected_logps",
    "response_length/mean",
]

KEYS_HF = [
    "train/no-beta-dpo-rewards/chosen",
    "train/no-beta-dpo-rewards/rejected",
]

KEYS_TRL = [
    "train/kl/chosen",
    "train/kl/rejected",
    "train/kl/overall",
    "train/loss/lens",
]


def detect_format(run):
    summary_keys = set(run.summary.keys())
    if "train/kl/policy_vs_ref/mean" in summary_keys:
        return "direct"
    if "actor/policy_chosen_logps" in summary_keys:
        return "verl"
    if "train/kl/overall" in summary_keys:
        return "trl"
    if "train/no-beta-dpo-rewards/chosen" in summary_keys:
        return "hf"
    return None


def fetch_kl(run, fmt):
    keys_map = {
        "direct": KEYS_DIRECT,
        "verl": KEYS_VERL,
        "hf": KEYS_HF,
        "trl": KEYS_TRL,
    }
    keys = keys_map[fmt]

    history = list(run.scan_history(keys=["_step"] + keys))
    steps, kl_mean = [], []

    for row in history:
        if fmt == "direct":
            km = row.get("train/kl/policy_vs_ref/mean")
            if km is None:
                continue
        elif fmt == "verl":
            pc = row.get("actor/policy_chosen_logps")
            rc = row.get("actor/reference_chosen_logps")
            if pc is None or rc is None:
                continue
            pr = row.get("actor/policy_rejected_logps")
            rr = row.get("actor/reference_rejected_logps")
            kc = pc - rc
            kr = (pr - rr) if (pr is not None and rr is not None) else None
            km = (kc + kr) / 2 if kr is not None else kc
            resp_len = row.get("response_length/mean")
            if resp_len is not None and resp_len > 0:
                km = km / resp_len
        elif fmt == "trl":
            km = row.get("train/kl/overall")
            if km is None:
                kc = row.get("train/kl/chosen")
                kr = row.get("train/kl/rejected")
                if kc is None and kr is None:
                    continue
                if kc is not None and kr is not None:
                    km = (kc + kr) / 2
                else:
                    km = kc if kc is not None else kr
            resp_len = row.get("train/loss/lens")
            if resp_len is not None and resp_len > 0:
                km = km / resp_len
        else:
            kc = row.get("train/no-beta-dpo-rewards/chosen")
            kr = row.get("train/no-beta-dpo-rewards/rejected")
            if kc is None and kr is None:
                continue
            if kc is not None and kr is not None:
                km = (kc + kr) / 2
            else:
                km = kc if kc is not None else kr

        steps.append(row.get("_step", 0))
        kl_mean.append(km)

    return steps, kl_mean


def main():
    parser = argparse.ArgumentParser(description="Plot per-token log-ratio from one or more wandb runs")
    parser.add_argument("run_paths", nargs="+", help="wandb run paths, e.g. entity/project/run_id")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path (default: kl_<run_id>.png or kl_comparison.png)")
    parser.add_argument("--labels", nargs="+", default=None, help="Custom labels for each run")
    args = parser.parse_args()

    api = wandb.Api()
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, run_path in enumerate(args.run_paths):
        run = api.run(run_path)
        fmt = detect_format(run)
        if fmt is None:
            print(f"WARNING: Could not detect format for {run_path}, skipping.")
            continue

        print(f"Fetching {run_path} (format={fmt})...")
        steps, kl_mean = fetch_kl(run, fmt)
        if not steps:
            print(f"  No KL data found, skipping.")
            continue

        label = args.labels[i] if (args.labels and i < len(args.labels)) else f"{run.name} ({fmt})"
        ax.plot(steps, kl_mean, linewidth=2, label=label, alpha=0.8)
        print(f"  {len(steps)} steps, KL range: [{min(kl_mean):.4f}, {max(kl_mean):.4f}]")

    ax.set_ylabel("Mean log-ratio per token (π − π_ref)")
    ax.set_xlabel("Step")
    ax.set_title("Policy Drift from Reference")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if args.output:
        out_path = args.output
    elif len(args.run_paths) == 1:
        run_id = args.run_paths[0].split("/")[-1]
        out_path = f"kl_{run_id}.png"
    else:
        out_path = "kl_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
