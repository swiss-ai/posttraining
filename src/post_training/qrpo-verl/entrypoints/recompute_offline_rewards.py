from __future__ import annotations

import json
import math
import os
import shutil
import socket
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from pprint import pprint
from typing import Any

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from verl.experimental.reward_loop import RewardLoopManager
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils import hf_tokenizer
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local

from data.dataset_adapter import load_hf_dataset_from_config, row_to_prompt_record
from offline_rewards import (
    OfflineRewardChunkStore,
    OfflineRewardRequest,
    apply_offline_reward_rows_to_dataset,
    offline_reward_requests_from_prompt_records,
    score_offline_reward_requests,
)


@hydra.main(
    config_path="../configs",
    config_name="qrpo",
    version_base=None,
)
def main(config: DictConfig) -> None:
    auto_set_device(config)
    recompute_offline_rewards(config)


def recompute_offline_rewards(config: DictConfig) -> None:
    """Recompute dataset offline rewards with the configured reward function."""

    print(f"Offline reward recompute hostname: {socket.gethostname()}")
    pprint(OmegaConf.to_container(config, resolve=True))

    OmegaConf.resolve(config)
    _init_ray(config)

    offline_reward_config = config.get("offline_rewards", None)
    if offline_reward_config is None:
        raise ValueError("offline_rewards config is required.")

    output_path = offline_reward_config.get("output_path", None)
    if output_path is None:
        raise ValueError("offline_rewards.output_path must be set.")
    output_path = Path(str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overwrite_output = bool(offline_reward_config.get("overwrite_output", False))
    if output_path.exists():
        if not overwrite_output:
            raise FileExistsError(
                f"offline_rewards.output_path already exists: {output_path}. "
                "Set offline_rewards.overwrite_output=true to replace it."
            )

    work_dir = offline_reward_config.get("work_dir", None)
    if work_dir is None:
        work_dir = f"{output_path}.work"
    chunk_store = OfflineRewardChunkStore(work_dir)

    reward_num_workers = int(config.reward.get("num_workers", 0))
    if reward_num_workers <= 0:
        raise ValueError("Offline reward recomputation requires reward.num_workers > 0.")
    if bool(config.reward.reward_model.get("enable", False)):
        raise NotImplementedError(
            "entrypoints.recompute_offline_rewards currently supports custom "
            "reward functions through RewardLoopManager. Standalone reward-model "
            "resource pools are not wired here yet."
        )

    chunk_size_completions = int(
        offline_reward_config.get(
            "chunk_size_completions",
            config.data.get("train_batch_size", reward_num_workers),
        )
    )
    if chunk_size_completions <= 0:
        raise ValueError(
            "offline_rewards.chunk_size_completions must be positive, got "
            f"{chunk_size_completions}."
        )
    if chunk_size_completions % reward_num_workers != 0:
        raise ValueError(
            "offline_rewards.chunk_size_completions must be divisible by "
            f"reward.num_workers ({reward_num_workers}), got "
            f"{chunk_size_completions}."
        )

    data_source = offline_reward_config.get("data_source", None)
    if not data_source:
        raise ValueError("offline_rewards.data_source must be set.")

    local_model_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )
    tokenizer = hf_tokenizer(
        local_model_path,
        trust_remote_code=bool(config.data.get("trust_remote_code", False)),
    )
    _ensure_tokenizer_has_pad_token(tokenizer, config)

    dataset = load_hf_dataset_from_config(config.data)
    dataset_config = OmegaConf.to_container(config.data, resolve=True)
    dataset_config["ref_rewards_key"] = None
    dataset_config["offline_rewards_key"] = None

    total_offline_completions = _count_offline_completions(
        dataset=dataset,
        dataset_config=dataset_config,
    )
    manifest = _offline_reward_manifest(
        config=config,
        dataset=dataset,
        dataset_size=len(dataset),
        total_offline_completions=total_offline_completions,
        chunk_size_completions=chunk_size_completions,
        data_source=str(data_source),
    )
    _write_manifest(Path(work_dir), manifest)

    offline_reward_tokenization_config = OmegaConf.to_container(
        offline_reward_config.get("tokenization", config.offline_tokenization),
        resolve=True,
    )
    reward_loop_manager = RewardLoopManager(config=config)
    reuse_existing = bool(offline_reward_config.get("reuse_existing", True))

    all_rows: list[dict[str, Any]] = []
    chunks = _iter_offline_reward_request_chunks(
        dataset=dataset,
        dataset_config=dataset_config,
        chunk_size_completions=chunk_size_completions,
    )
    total_chunks = math.ceil(total_offline_completions / chunk_size_completions)

    for chunk_index, requests in tqdm(
        chunks,
        total=total_chunks,
        desc="Offline Reward Chunks",
    ):
        if reuse_existing and chunk_store.chunk_path(chunk_index).exists():
            rows = chunk_store.load_chunk(
                chunk_index=chunk_index,
                manifest=manifest,
            )
        else:
            rows = _score_request_chunk_by_tools(
                reward_loop_manager=reward_loop_manager,
                requests=requests,
                tokenizer=tokenizer,
                data_source=str(data_source),
                offline_tokenization_config=offline_reward_tokenization_config,
                chunk_size_completions=chunk_size_completions,
                meta_info={"offline_reward_chunk_index": int(chunk_index)},
            )
            chunk_store.save_chunk(
                chunk_index=chunk_index,
                rows=rows,
                manifest=manifest,
                overwrite=not reuse_existing,
            )
        all_rows.extend(rows)

    updated_dataset = apply_offline_reward_rows_to_dataset(
        dataset=dataset,
        rows=all_rows,
        offline_trajectories_key=str(
            config.data.get("offline_trajectories_key", "offline_trajectories")
        ),
        offline_rewards_key=str(
            config.data.get("offline_rewards_key", "offline_rewards")
        ),
        reward_extra_info_key=offline_reward_config.get(
            "reward_extra_info_key",
            "offline_reward_extra_info",
        ),
        require_all=True,
    )
    if output_path.exists():
        shutil.rmtree(output_path)
    updated_dataset.save_to_disk(str(output_path))


def _iter_offline_reward_request_chunks(
    *,
    dataset: Any,
    dataset_config: Mapping[str, Any],
    chunk_size_completions: int,
) -> Iterator[tuple[int, list[OfflineRewardRequest]]]:
    buffer: list[OfflineRewardRequest] = []
    chunk_index = 0

    for dataset_index in range(len(dataset)):
        prompt_record = row_to_prompt_record(
            dataset[int(dataset_index)],
            dataset_config,
            row_index=int(dataset_index),
        )
        buffer.extend(
            offline_reward_requests_from_prompt_records(
                prompt_records=[prompt_record],
                dataset_indices=[int(dataset_index)],
            )
        )

        while len(buffer) >= chunk_size_completions:
            yield chunk_index, buffer[:chunk_size_completions]
            buffer = buffer[chunk_size_completions:]
            chunk_index += 1

    if buffer:
        yield chunk_index, buffer


def _score_request_chunk_by_tools(
    *,
    reward_loop_manager: Any,
    requests: list[OfflineRewardRequest],
    tokenizer: Any,
    data_source: str,
    offline_tokenization_config: Mapping[str, Any],
    chunk_size_completions: int,
    meta_info: dict[str, Any],
) -> list[dict[str, Any]]:
    """Score a flat chunk, grouping internally by tools for tokenizer calls."""

    rows: list[dict[str, Any]] = []
    for group_index, group_requests in enumerate(_group_requests_by_tools(requests)):
        group_meta_info = dict(meta_info)
        group_meta_info["offline_reward_tool_group_index"] = int(group_index)
        rows.extend(
            score_offline_reward_requests(
                reward_loop_manager=reward_loop_manager,
                requests=group_requests,
                tokenizer=tokenizer,
                data_source=data_source,
                offline_tokenization_config=offline_tokenization_config,
                chunk_size_completions=chunk_size_completions,
                meta_info=group_meta_info,
                description=None,
            )
        )
    rows.sort(
        key=lambda row: (
            int(row["dataset_index"]),
            int(row["offline_completion_index"]),
        )
    )
    return rows


def _group_requests_by_tools(
    requests: Sequence[OfflineRewardRequest],
) -> list[list[OfflineRewardRequest]]:
    groups: dict[str, list[OfflineRewardRequest]] = {}
    for request in requests:
        groups.setdefault(_tools_signature(request.tools), []).append(request)
    return list(groups.values())


def _tools_signature(tools: Any) -> str:
    if tools is None:
        return "null"
    return json.dumps(tools, sort_keys=True, default=str)


def _count_offline_completions(
    *,
    dataset: Any,
    dataset_config: Mapping[str, Any],
) -> int:
    offline_trajectories_key = str(
        dataset_config.get("offline_trajectories_key", "offline_trajectories")
    )
    total = 0
    for dataset_index in tqdm(
        range(len(dataset)),
        desc="Counting Offline Completions",
    ):
        row = dataset[int(dataset_index)]
        total += len(row[offline_trajectories_key])
    return total


def _offline_reward_manifest(
    *,
    config: DictConfig,
    dataset: Any,
    dataset_size: int,
    total_offline_completions: int,
    chunk_size_completions: int,
    data_source: str,
) -> dict[str, Any]:
    offline_reward_config = config.offline_rewards
    env_keys = list(offline_reward_config.get("manifest_env_keys", []))
    reward_function_path = config.reward.custom_reward_function.get("path", None)
    manifest = {
        "source": "offline_reward_recompute",
        "data_path": str(config.data.get("path")),
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "dataset_size": int(dataset_size),
        "total_offline_completions": int(total_offline_completions),
        "chunk_size_completions": int(chunk_size_completions),
        "data_source": str(data_source),
        "actor_model_path": str(config.actor_rollout_ref.model.path),
        "reward_function_path": reward_function_path,
        "reward_function_sha256": _file_sha256(reward_function_path),
        "reward_function_name": config.reward.custom_reward_function.get(
            "name",
            None,
        ),
        "reward_manager": OmegaConf.to_container(
            config.reward.reward_manager,
            resolve=True,
        ),
        "offline_reward_tokenization": OmegaConf.to_container(
            offline_reward_config.get("tokenization", config.offline_tokenization),
            resolve=True,
        ),
        "offline_rewards_key": str(
            config.data.get("offline_rewards_key", "offline_rewards")
        ),
        "reward_extra_info_key": offline_reward_config.get(
            "reward_extra_info_key",
            "offline_reward_extra_info",
        ),
    }
    for env_key in env_keys:
        env_key = str(env_key)
        env_value = os.environ.get(env_key)
        manifest[f"env/{env_key}"] = env_value
        manifest[f"env/{env_key}_sha256"] = _file_sha256(env_value)
    return manifest


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _file_sha256(path: Any | None) -> str | None:
    if not path:
        return None

    path = os.fspath(path)
    if not os.path.isfile(path):
        return None

    import hashlib

    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _init_ray(config: DictConfig) -> None:
    if ray.is_initialized():
        return

    default_runtime_env = get_ppo_ray_runtime_env()
    ray_init_kwargs = config.get("ray_kwargs", {}).get("ray_init", {})
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
    ray_init_kwargs = OmegaConf.create(
        {
            **ray_init_kwargs,
            "runtime_env": runtime_env,
        }
    )
    print(f"ray init kwargs: {ray_init_kwargs}")
    ray.init(**OmegaConf.to_container(ray_init_kwargs, resolve=True))


def _ensure_tokenizer_has_pad_token(tokenizer: Any, config: DictConfig) -> None:
    if tokenizer.pad_token_id is not None:
        return

    use_eos_as_pad = bool(config.get("tokenizer", {}).get("use_eos_as_pad", False))
    if not use_eos_as_pad:
        raise ValueError(
            "tokenizer.pad_token_id is None. Set tokenizer.use_eos_as_pad=true "
            "explicitly if you want to use eos_token as pad_token."
        )
    if tokenizer.eos_token is None:
        raise ValueError(
            "Cannot use eos_token as pad_token because tokenizer.eos_token is None."
        )
    tokenizer.pad_token = tokenizer.eos_token


if __name__ == "__main__":
    main()
