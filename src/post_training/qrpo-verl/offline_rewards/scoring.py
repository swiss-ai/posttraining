from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm import tqdm
from verl.protocol import DataProto

from batch import keys as K
from batch.offline_tokenization import tokenize_prompt_trajectory_batch
from data.schemas import ChatMessage, PromptRecord
from verl_adapters.rewards import sequence_rewards_from_verl_output


_OFFLINE_REWARD_METADATA_KEYS = (
    K.PROMPT_ID,
    K.TRAJECTORY_ID,
    K.DATASET_INDEX,
    K.OFFLINE_COMPLETION_INDEX,
)


@dataclass(frozen=True)
class OfflineRewardRequest:
    """One existing offline trajectory to score with VERL's reward loop."""

    prompt_id: str
    trajectory_id: str
    dataset_index: int
    offline_completion_index: int
    prompt_messages: tuple[ChatMessage, ...]
    trajectory_messages: tuple[ChatMessage, ...]
    tools: tuple[Mapping[str, Any], ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def offline_reward_requests_from_prompt_records(
    *,
    prompt_records: Sequence[PromptRecord],
    dataset_indices: Sequence[int],
) -> list[OfflineRewardRequest]:
    """Flatten prompt records into completion-level offline reward requests."""

    prompt_records = tuple(prompt_records)
    dataset_indices = tuple(int(index) for index in dataset_indices)

    if len(prompt_records) != len(dataset_indices):
        raise ValueError(
            f"Got {len(dataset_indices)} dataset_indices for "
            f"{len(prompt_records)} prompt_records."
        )

    requests: list[OfflineRewardRequest] = []
    for prompt, dataset_index in zip(
        prompt_records,
        dataset_indices,
        strict=True,
    ):
        for offline_completion_index, trajectory_messages in enumerate(
            prompt.offline_trajectories
        ):
            requests.append(
                OfflineRewardRequest(
                    prompt_id=prompt.prompt_id,
                    trajectory_id=_offline_reward_trajectory_id(
                        prompt_id=prompt.prompt_id,
                        offline_completion_index=offline_completion_index,
                    ),
                    dataset_index=int(dataset_index),
                    offline_completion_index=int(offline_completion_index),
                    prompt_messages=tuple(prompt.prompt_messages),
                    trajectory_messages=tuple(trajectory_messages),
                    tools=prompt.tools,
                    metadata=dict(prompt.metadata),
                )
            )

    return requests


def offline_reward_requests_to_dataproto(
    *,
    requests: Sequence[OfflineRewardRequest],
    tokenizer: Any,
    data_source: str,
    config: Mapping[str, Any] | None = None,
    meta_info: dict[str, Any] | None = None,
) -> DataProto:
    """Build the DataProto expected by VERL RewardLoopManager."""

    cfg = config or {}
    requests = tuple(requests)

    if not requests:
        raise ValueError("Cannot build offline reward input for zero requests.")
    if not data_source:
        raise ValueError("offline reward scoring requires a non-empty data_source.")

    tensors = tokenize_prompt_trajectory_batch(
        items=requests,
        tokenizer=tokenizer,
        config=cfg,
    )

    reward_models = np.empty(len(requests), dtype=object)
    extra_infos = np.empty(len(requests), dtype=object)

    for index, request in enumerate(requests):
        reward_models[index] = {
            "ground_truth": {
                "prompt": request.prompt_messages,
                "prompt_id": request.prompt_id,
                "trajectory_id": request.trajectory_id,
                K.DATASET_INDEX: int(request.dataset_index),
                K.OFFLINE_COMPLETION_INDEX: int(
                    request.offline_completion_index
                ),
            }
        }
        extra_infos[index] = {
            "prompt": request.prompt_messages,
            "prompt_id": request.prompt_id,
            "trajectory_id": request.trajectory_id,
            K.DATASET_INDEX: int(request.dataset_index),
            K.OFFLINE_COMPLETION_INDEX: int(request.offline_completion_index),
            K.SOURCE: K.SOURCE_OFFLINE,
            "metadata": dict(request.metadata),
        }

    non_tensors = {
        K.PROMPT_ID: np.asarray(
            [request.prompt_id for request in requests],
            dtype=object,
        ),
        K.TRAJECTORY_ID: np.asarray(
            [request.trajectory_id for request in requests],
            dtype=object,
        ),
        K.DATASET_INDEX: np.asarray(
            [request.dataset_index for request in requests],
            dtype=np.int64,
        ),
        K.OFFLINE_COMPLETION_INDEX: np.asarray(
            [request.offline_completion_index for request in requests],
            dtype=np.int64,
        ),
        K.DATA_SOURCE: np.asarray([data_source] * len(requests), dtype=object),
        K.REWARD_MODEL: reward_models,
        K.EXTRA_INFO: extra_infos,
    }

    merged_meta_info = {
        "qrpo_batch_format": "offline_reward_requests",
        "validate": bool(cfg.get("validate", False)),
    }
    if meta_info is not None:
        merged_meta_info.update(meta_info)

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info=merged_meta_info,
    )


def score_offline_reward_requests(
    *,
    reward_loop_manager: Any,
    requests: Sequence[OfflineRewardRequest],
    tokenizer: Any,
    data_source: str,
    offline_tokenization_config: Mapping[str, Any] | None = None,
    chunk_size_completions: int | None = None,
    meta_info: dict[str, Any] | None = None,
    description: str | None = "Scoring Offline Rewards",
) -> list[dict[str, Any]]:
    """Score offline requests through VERL RewardLoopManager."""

    num_reward_workers = len(getattr(reward_loop_manager, "reward_loop_workers", []))
    if num_reward_workers <= 0:
        raise ValueError(
            "Offline reward scoring requires at least one reward loop worker. "
            "Set reward.num_workers > 0."
        )

    requests = tuple(requests)
    if not requests:
        return []

    if chunk_size_completions is None:
        chunk_size_completions = len(requests)
    chunk_size_completions = int(chunk_size_completions)
    if chunk_size_completions <= 0:
        raise ValueError(
            "chunk_size_completions must be positive, got "
            f"{chunk_size_completions}."
        )
    if chunk_size_completions % num_reward_workers != 0:
        raise ValueError(
            "chunk_size_completions must be divisible by the number of reward "
            f"workers ({num_reward_workers}) so full chunks do not need padding. "
            f"Got {chunk_size_completions}."
        )

    rows: list[dict[str, Any]] = []
    starts = range(0, len(requests), chunk_size_completions)
    if description is not None:
        starts = tqdm(starts, desc=description)

    for start in starts:
        chunk_requests = requests[start:start + chunk_size_completions]
        reward_input = offline_reward_requests_to_dataproto(
            requests=chunk_requests,
            tokenizer=tokenizer,
            data_source=data_source,
            config=offline_tokenization_config,
            meta_info=meta_info,
        )
        metadata = extract_offline_reward_metadata(reward_input)
        reward_input, original_size = pad_offline_reward_input_for_reward_workers(
            reward_input,
            num_workers=num_reward_workers,
        )
        reward_output = reward_loop_manager.compute_rm_score(reward_input)
        reward_output = truncate_offline_reward_output(
            reward_output,
            size=original_size,
        )
        attach_offline_reward_metadata(
            reward_output=reward_output,
            metadata=metadata,
        )
        rows.extend(offline_reward_output_to_rows(reward_output))

    rows.sort(
        key=lambda row: (
            int(row[K.DATASET_INDEX]),
            int(row[K.OFFLINE_COMPLETION_INDEX]),
        )
    )
    return rows


def extract_offline_reward_metadata(request_batch: DataProto) -> dict[str, np.ndarray]:
    missing_keys = [
        key for key in _OFFLINE_REWARD_METADATA_KEYS
        if key not in request_batch.non_tensor_batch
    ]
    if missing_keys:
        raise KeyError(
            "offline reward request batch is missing metadata keys: "
            f"{missing_keys}."
        )

    return {
        key: request_batch.non_tensor_batch[key]
        for key in _OFFLINE_REWARD_METADATA_KEYS
    }


def pad_offline_reward_input_for_reward_workers(
    request_batch: DataProto,
    *,
    num_workers: int,
) -> tuple[DataProto, int]:
    """Pad reward requests so VERL RewardLoopManager can split them equally."""

    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}.")

    original_size = len(request_batch)
    if original_size == 0:
        raise ValueError("Cannot pad an empty offline reward request batch.")

    remainder = original_size % num_workers
    if remainder == 0:
        return request_batch, original_size

    pad_size = num_workers - remainder
    pad_indices = np.arange(pad_size) % original_size
    selected_indices = np.concatenate(
        [np.arange(original_size, dtype=np.int64), pad_indices.astype(np.int64)],
        axis=0,
    )
    return request_batch.select_idxs(selected_indices), original_size


def truncate_offline_reward_output(
    reward_output: DataProto,
    *,
    size: int,
) -> DataProto:
    """Drop padded reward outputs before persistence."""

    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}.")
    if len(reward_output) < size:
        raise ValueError(
            f"Cannot truncate reward output of size {len(reward_output)} "
            f"to larger size {size}."
        )
    if len(reward_output) == size:
        return reward_output

    return reward_output.select_idxs(np.arange(size, dtype=np.int64))


def attach_offline_reward_metadata(
    *,
    reward_output: DataProto,
    metadata: Mapping[str, np.ndarray],
) -> DataProto:
    """Attach request metadata to reward output in-place."""

    batch_size = len(reward_output)
    output_non_tensors = reward_output.non_tensor_batch

    missing_keys = [
        key for key in _OFFLINE_REWARD_METADATA_KEYS
        if key not in metadata
    ]
    if missing_keys:
        raise KeyError(
            "offline reward metadata is missing required keys: "
            f"{missing_keys}."
        )

    for key in _OFFLINE_REWARD_METADATA_KEYS:
        values = metadata[key]
        if len(values) != batch_size:
            raise ValueError(
                f"offline reward metadata[{key!r}] has length {len(values)}, "
                f"expected {batch_size}."
            )

        if key not in output_non_tensors:
            output_non_tensors[key] = values
            continue

        existing = output_non_tensors[key]
        if len(existing) != batch_size:
            raise ValueError(
                f"reward_output.non_tensor_batch[{key!r}] has length "
                f"{len(existing)}, expected {batch_size}."
            )
        if existing.tolist() != values.tolist():
            raise ValueError(
                f"reward_output.non_tensor_batch[{key!r}] does not match "
                "the corresponding offline reward request metadata."
            )

    return reward_output


def offline_reward_output_to_rows(
    reward_output: DataProto,
) -> list[dict[str, Any]]:
    """Convert a scored reward DataProto into flat offline reward rows."""

    if reward_output.batch is None:
        raise ValueError("reward_output.batch is required.")

    for key in _OFFLINE_REWARD_METADATA_KEYS:
        if key not in reward_output.non_tensor_batch:
            raise KeyError(f"reward_output.non_tensor_batch is missing {key!r}.")

    rewards = sequence_rewards_from_verl_output(
        reward_output,
        batch_size=len(reward_output),
        reward_key="rm_scores",
    ).detach().cpu().float()
    reward_extra_keys = (reward_output.meta_info or {}).get("reward_extra_keys", [])

    rows: list[dict[str, Any]] = []
    for index in range(len(reward_output)):
        reward_extra_info = {
            key: _to_debug_value(reward_output.non_tensor_batch[key][index])
            for key in reward_extra_keys
            if key in reward_output.non_tensor_batch
        }
        rows.append(
            {
                "prompt_id": str(reward_output.non_tensor_batch[K.PROMPT_ID][index]),
                K.DATASET_INDEX: int(
                    reward_output.non_tensor_batch[K.DATASET_INDEX][index]
                ),
                K.OFFLINE_COMPLETION_INDEX: int(
                    reward_output.non_tensor_batch[K.OFFLINE_COMPLETION_INDEX][index]
                ),
                K.TRAJECTORY_ID: str(
                    reward_output.non_tensor_batch[K.TRAJECTORY_ID][index]
                ),
                "reward": float(rewards[index].item()),
                "reward_extra_info": reward_extra_info or None,
            }
        )

    rows.sort(
        key=lambda row: (
            int(row[K.DATASET_INDEX]),
            int(row[K.OFFLINE_COMPLETION_INDEX]),
        )
    )
    return rows


def _offline_reward_trajectory_id(
    *,
    prompt_id: str,
    offline_completion_index: int,
) -> str:
    return f"{prompt_id}::{K.SOURCE_OFFLINE}::idx_{offline_completion_index}"


def _to_debug_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, tuple):
        return [_to_debug_value(item) for item in value]

    if isinstance(value, list):
        return [_to_debug_value(item) for item in value]

    if isinstance(value, Mapping):
        return {
            str(key): _to_debug_value(item)
            for key, item in value.items()
        }

    return value
