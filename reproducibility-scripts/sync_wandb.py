"""
To force sync a job from its jobid or its StdErr file.
"""


import logging
import os
import re
import subprocess
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)

os.environ["SLURM_JOB_ID"] = ""
os.environ[
    "SLURM_STDERR"
] = "/users/smoalla/projects/posttraining/dev/reproducibility-scripts/sft/fix-overfit/adam-2025-08-07-00-16/out/Apertus8B-tokens7.2T-it1728000-hotfix-apertus-sft-mixture-1-bs512-lr5e-06-epochs1-adam.err"


def try_sync_wandb(logger=_logger):
    def sync_runs_in_dir(parent_dir: Path):
        # find all run-* dirs
        run_dirs = [d for d in parent_dir.glob("run-*") if d.is_dir()]
        # sort by creation time (oldest first)
        run_dirs.sort(key=lambda d: d.stat().st_ctime)
        for run_dir in run_dirs:
            logger.info(f"Syncing wandb run directory: {run_dir}")
            try:
                subprocess.run(
                    ["wandb", "sync", str(run_dir)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to sync {run_dir}: {e.stderr.strip() or e}")
            else:
                logger.info(f"Successfully synced wandb run directory: {run_dir}")

    def extract_slurm_field(job_info: str, key: str) -> str | None:
        """Parses 'scontrol show job' output for KEY=value and returns the value."""
        m = re.search(rf"\b{re.escape(key)}=(\S+)", job_info)
        return m.group(1) if m else None

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is not None and job_id != "":
        logger.info(f"Attempting to sync wandb runs for SLURM job {job_id}")
        try:
            job_info = subprocess.check_output(
                ["scontrol", "show", "job", job_id],
                text=True,
            )
        except subprocess.SubprocessError as e:
            logger.error(f"Could not fetch SLURM info for job {job_id}: {e}")
            return

        out_path = extract_slurm_field(job_info, "StdOut")
        err_path = extract_slurm_field(job_info, "StdErr")
    else:
        logger.warning("No SLURM_JOB_ID not set.")
        # env vars for StdOut and StdErr
        out_path = os.environ.get("SLURM_STDOUT", "")
        err_path = os.environ.get("SLURM_STDERR", "")
        logger.info("Using environment variables for StdOut and StdErr.")

    if not out_path and not err_path:
        logger.error("No valid SLURM output or error paths found.")
        return

    logger.info(f"SLURM job {job_id} StdOut: {out_path}, StdErr: {err_path}")
    wandb_dir = None
    for path in (out_path, err_path):
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                for line in f:
                    if "wandb: Run data is saved locally in" in line:
                        wandb_dir = line.split(
                            "wandb: Run data is saved locally in", 1
                        )[1].strip()
                        break
        except Exception as e:
            logger.error(f"Could not read SLURM log {path}: {e}")
        if wandb_dir:
            wandb_dir = Path(wandb_dir).parent
            break

    if wandb_dir:
        logger.info(f"Found wandb run directory: {wandb_dir}")
        sync_runs_in_dir(wandb_dir)
    else:
        logger.warning("Couldn't locate a wandb run directory in SLURM logs.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try_sync_wandb()
    sys.exit(0)
