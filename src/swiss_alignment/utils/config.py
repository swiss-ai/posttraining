# Resolvers can be used in the config files.
# https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html
# They are useful when you want to make the default values of some config variables
# result from direct computation of other config variables.
# Only put variables meant to be edited by the user (as opposed to read-only variables described below)
# and avoid making them too complicated, the point is not to write code in the config file.
import logging
import os
import subprocess
import sys
from hashlib import blake2b
from pathlib import Path

import wandb
from omegaconf import DictConfig, OmegaConf, omegaconf

from swiss_alignment import utils

# Hydra sets up the logger automatically.
# https://hydra.cc/docs/tutorials/basic/running_your_app/logging/
_logger = logging.getLogger(__name__)


def register_resolvers():
    if not OmegaConf.has_resolver("eval"):
        # Useful to evaluate expressions in the config file.
        OmegaConf.register_new_resolver("eval", eval, use_cache=True)
    if not OmegaConf.has_resolver("generate_random_seed"):
        # Generate a random seed and record it in the config of the experiment.
        OmegaConf.register_new_resolver(
            "generate_random_seed", utils.seeding.generate_random_seed, use_cache=True
        )


def save_or_check_config(config: DictConfig, config_to_check_path: str) -> None:
    """
    Save if it doesn't exist; otherwise (in case of resuming) assert that the
    config is the same. If they differ, log the differing key(s).
    """
    config_to_check_path = Path(config_to_check_path)
    if not config_to_check_path.exists():
        OmegaConf.save(config, config_to_check_path)
        return

    new_config = config.copy()
    existing_config = OmegaConf.load(config_to_check_path)

    OmegaConf.resolve(new_config)
    OmegaConf.resolve(existing_config)

    remove_ignored_keys(new_config, config.resuming.ignore_keys)
    remove_ignored_keys(existing_config, config.resuming.ignore_keys)

    new_config_dict = OmegaConf.to_container(new_config, resolve=True)
    existing_config_dict = OmegaConf.to_container(existing_config, resolve=True)

    # Compare dictionaries
    differences = dictionary_diff(new_config_dict, existing_config_dict)
    if differences:
        diff_msg = "\n".join(differences)
        _logger.error(
            f"Config to resume is different from the one saved in {config_to_check_path}.\n"
            f"Differences:\n{diff_msg}"
        )
        raise AssertionError(
            f"Config differs from the existing config at {config_to_check_path}. See logs for details."
        )

    _logger.info(
        f"Configs match the one in {config_to_check_path}. Resuming with the same config."
    )


def remove_ignored_keys(config: DictConfig, ignore_keys: list[str]) -> None:
    """
    Remove keys from the config that are specified in ignore_keys.
    Exclude keys can be specified as dot-paths, e.g., "key1.key2.key3".
    """
    with omegaconf.open_dict(config):
        for key in ignore_keys:
            try:
                path_segments = key.split(".")
                node = config
                for segment in path_segments[:-1]:
                    node = node[segment]  # drill down
                del node[path_segments[-1]]  # remove the final key
            except KeyError:
                pass


def dictionary_diff(d1: dict, d2: dict, path: str = "") -> list[str]:
    """
    Recursively compare two dictionary (or scalar) structures and return a list
    of human-readable differences. `path` is carried along to show the nested key path.
    """
    differences = []

    # If both are dict-like, compare keys and recurse
    if isinstance(d1, dict) and isinstance(d2, dict):
        all_keys = set(d1.keys()).union(d2.keys())
        for key in all_keys:
            new_path = f"{path}.{key}" if path else key
            if key not in d1:
                differences.append(f"Missing in new config: {new_path}")
            elif key not in d2:
                differences.append(f"Missing in existing config: {new_path}")
            else:
                # Recurse
                differences.extend(dictionary_diff(d1[key], d2[key], new_path))
    else:
        # If they are not both dicts, compare values directly
        if d1 != d2:
            differences.append(
                f"Value mismatch at '{path}': new='{d1}' vs existing='{d2}'"
            )

    return differences


def setup_resuming_dir(config):
    """Create a unique identifier of the experiment used to specify a resuming/checkpoint directory.
    The identifier is a hash of the config, excluding keys specified in config.resuming.ignore_keys.
    If config.resuming.use_commit is True, the commit hash is appended to the identifier.
    I.e. the checkpoint directory is defined by: the config - the excluded config keys + the commit hash (if specified)
    """
    if config.resuming_dir is not None:
        return

    config_to_hash = config.copy()
    OmegaConf.resolve(config_to_hash)
    remove_ignored_keys(config_to_hash, config.resuming.ignore_keys)
    config_hash = blake2b(str(config_to_hash).encode(), digest_size=8).hexdigest()
    resuming_hash = config_hash
    if config.resuming.use_commit:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        resuming_hash += f"-{commit_hash[:8]}"

    resuming_dir = Path.cwd().parent.parent / "checkpoints" / resuming_hash
    resuming_dir.mkdir(parents=True, exist_ok=True)
    with omegaconf.open_dict(config):
        config.resuming_dir = str(resuming_dir)
        config.resuming_hash = resuming_hash
        if config.resuming.resume:
            if config.wandb.run_id is None:
                config.wandb.run_id = config.resuming_hash
            if config.wandb.run_name is None:
                config.wandb.run_name = config.resuming_hash


def setup_config_and_resuming(config, postprocess_func=None, logger=_logger):
    logger.info(f"Running command: {subprocess.list2cmdline(sys.argv)}")
    logger.info(f"Init directory: {Path.cwd()}")
    utils.config.setup_resuming_dir(config)
    logger.info(f"Run can be resumed from the directory: {config.resuming_dir}")
    if config.resuming.resume:
        os.chdir(config.resuming_dir)

    Path(f"config").mkdir(exist_ok=True, parents=True)
    utils.config.save_or_check_config(
        config,
        f"config/config-raw.yaml",
    )

    # Do some optional postprocessing to the config (e.g., checking division of batch size etc.)
    OmegaConf.resolve(config)
    if postprocess_func:
        config = postprocess_func(config)

    # Save the resolved config.
    utils.config.save_or_check_config(config, f"config/config-postprocessed.yaml")

    logger.info(f"Working directory: {Path.cwd()}")
    logger.info(f"Running with config: \n{OmegaConf.to_yaml(config)}")
    if config.resuming.resume:
        logger.info(f"Resuming from the directory: {Path.cwd()}")

    return config


def setup_wandb(config, logger=_logger):
    wandb_config = OmegaConf.to_container(config)
    wandb.init(
        id=config.wandb.run_id,
        name=config.wandb.run_name,
        resume="allow" if config.resuming.resume else "never",
        config=wandb_config,
        project=config.wandb.project,
        tags=config.wandb.tags,
        mode=config.wandb.mode,
        anonymous=config.wandb.anonymous,
        dir=Path.cwd() if not config.wandb.use_global_dir else config.wandb.global_dir,
    )

    # Re-log to capture log with wandb.
    logger.info(f"Running command: {subprocess.list2cmdline(sys.argv)}")
    logger.info(f"Init directory: {config.run_dir}")
    logger.info(f"Run can be resumed from the directory: {config.resuming_dir}")
    logger.info(f"Working directory: {Path.cwd()}")
    logger.info(f"Running with config: \n{OmegaConf.to_yaml(config)}")
    if config.resuming.resume:
        logger.info(f"Resuming from the directory: {Path.cwd()}")


def try_sync_wandb():
    # sync all runs in the wandb directory.
    run = wandb.run
    if run is not None:
        # The easy way.
        wandb_dir = Path(wandb.run.dir).parent.parent

        # Iterate over directories that start with run- and sync them in chronological order.
        for run_dir in sorted(wandb_dir.glob("run-*")):
            if not run_dir.is_dir():
                continue
            # Sync the run directory.
            logger = logging.getLogger(__name__)
            logger.info(f"Syncing wandb run directory: {run_dir}")
            try:
                subprocess.run(
                    ["wandb", "sync", str(run_dir)],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to sync {run_dir}: {e}")
                continue
            logger.info(f"Synced wandb run directory: {run_dir}")
    # else:
    #     # The SLURM way
    #     # extract the latest run from the SLURM logs
    #     """
    #     JOB_INFO=$(scontrol show job "$SLURM_JOB_ID")
    #     extract_field() {
    #       local key="$1"
    #     echo "$JOB_INFO" | tr ' ' '\n' | grep "^$key=" | cut -d= -f2-
    #     }
    #     OUTPUT_PATH=$(extract_field "StdOut")
    #     ERROR_PATH=$(extract_field "StdErr")
    #     """
    #     job_id = os.environ.get("SLURM_JOB_ID")
    #     if job_id is not None:
    #         try:
    #             job_info = subprocess.check_output(
    #                 ["scontrol", "show", "job", job_id],
    #                 text=True,
    #             )
    #             output_path = utils.slurm.extract_field(job_info, "StdOut")
    #             error_path = utils.slurm.extract_field(job_info, "StdErr")
    #
    #             # Find the wandb directory in the output or error files
    #             # wandb: Run data is saved locally in /iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/fix-overfit/Apertus8B-tokens7.2T-it1728000-hotfix-apertus-sft-mixture-1-bs512-lr5e-06-epochs1-adam/checkpoints/a0db78d43b9ebd41/wandb/run-20250807_002321-a0db78d43b9ebd41
    #             wandb_dir = None
    #             for line in (output_path, error_path):
    #                 if "wandb: Run data is saved locally in" in line:
    #                     wandb_dir = line.split("wandb: Run data is saved locally in")[-1].strip()
    #                     break
    #             if wandb_dir is not None:
    #                 wandb_dir = Path(wandb_dir)
    #                 if wandb_dir.exists():
    #                     # Iterate over directories that start with run- and sync them in chronological order.
    #                     for run_dir in sorted(wandb_dir.glob("run-*")):
    #                         if not run_dir.is_dir():
    #                             continue
    #                         # Sync the run directory.
    #                         logger = logging.getLogger(__name__)
    #                         logger.info(f"Syncing wandb run directory: {run_dir}")
    #                         try:
    #                             subprocess.run(
    #                                 ["wandb", "sync", str(run_dir)],
    #                                 check=True,
    #                                 capture_output=True,
    #                             )
    #                         except subprocess.CalledProcessError as e:
    #                             logger.error(f"Failed to sync {run_dir}: {e}")
    #                             continue
    #                         logger.info(f"Synced wandb run directory: {run_dir}")
