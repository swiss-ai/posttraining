from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from verl.protocol import DataProto

from batch import keys as K
from data.schemas import PromptRecord

_REF_ROLLOUT_METADATA_KEYS = (
    K.PROMPT_ID,
    K.TRAJECTORY_ID,
    K.DATASET_INDEX,
    K.REF_COMPLETION_INDEX,
)


def ref_reward_rollout_input_from_prompt_records(
    *,
    prompt_records: Sequence[PromptRecord],
    ref_version: str,
    num_ref_completions: int,
    config: Mapping[str, Any],
    dataset_indices: Sequence[int],
    meta_info: dict[str, Any] | None = None,
) -> DataProto:
    prompt_records = tuple(prompt_records)
    if not prompt_records:
        raise ValueError("Cannot build ref reward rollout input for zero prompts.")

    if num_ref_completions <= 0:
        raise ValueError(
            f"num_ref_completions must be positive, got {num_ref_completions}."
        )

    dataset_indices = tuple(int(index) for index in dataset_indices)

    if len(dataset_indices) != len(prompt_records):
        raise ValueError(
            f"Got {len(dataset_indices)} dataset_indices for "
            f"{len(prompt_records)} prompt_records."
        )

    data_source = config.get("data_source")
    if not data_source:
        raise ValueError(
            "ref reward rollout config requires data_source for reward computation."
        )

    agent_name = config.get("agent_name", None)
    size = len(prompt_records) * int(num_ref_completions)

    raw_prompt = np.empty(size, dtype=object)
    reward_models = np.empty(size, dtype=object)
    extra_infos = np.empty(size, dtype=object)
    prompt_ids: list[str] = []
    trajectory_ids: list[str] = []
    expanded_dataset_indices: list[int] = []
    ref_completion_indices: list[int] = []

    row_index = 0
    for prompt, dataset_index in zip(
        prompt_records,
        dataset_indices,
        strict=True,
    ):
        prompt_messages = tuple(prompt.prompt_messages)
        for ref_completion_index in range(num_ref_completions):
            trajectory_id = _ref_trajectory_id(
                prompt_id=prompt.prompt_id,
                ref_version=ref_version,
                ref_completion_index=ref_completion_index,
            )

            raw_prompt[row_index] = prompt_messages
            reward_models[row_index] = {
                "ground_truth": {
                    "prompt": prompt_messages,
                    "prompt_id": prompt.prompt_id,
                    "trajectory_id": trajectory_id,
                    K.REF_VERSION: ref_version,
                }
            }
            extra_infos[row_index] = {
                "prompt": prompt_messages,
                "prompt_id": prompt.prompt_id,
                "trajectory_id": trajectory_id,
                K.DATASET_INDEX: int(dataset_index),
                K.REF_VERSION: ref_version,
                K.REF_COMPLETION_INDEX: int(ref_completion_index),
            }

            prompt_ids.append(prompt.prompt_id)
            trajectory_ids.append(trajectory_id)
            expanded_dataset_indices.append(int(dataset_index))
            ref_completion_indices.append(int(ref_completion_index))
            row_index += 1

    non_tensor_batch: dict[str, np.ndarray] = {
        K.RAW_PROMPT: raw_prompt,
        K.PROMPT_ID: np.asarray(prompt_ids, dtype=object),
        K.TRAJECTORY_ID: np.asarray(trajectory_ids, dtype=object),
        K.DATASET_INDEX: np.asarray(expanded_dataset_indices, dtype=np.int64),
        K.REF_COMPLETION_INDEX: np.asarray(ref_completion_indices, dtype=np.int64),
        K.DATA_SOURCE: np.asarray([data_source] * size, dtype=object),
        K.REWARD_MODEL: reward_models,
        K.EXTRA_INFO: extra_infos,
    }

    if agent_name is not None:
        non_tensor_batch[K.AGENT_NAME] = np.asarray([agent_name] * size, dtype=object)

    merged_meta_info = {
        "qrpo_batch_format": "ref_reward_rollout_requests",
        K.REF_VERSION: str(ref_version),
        "num_ref_completions": int(num_ref_completions),
        "validate": bool(config.get("validate", False)),
    }
    if meta_info is not None:
        merged_meta_info.update(meta_info)

    return DataProto(
        non_tensor_batch=non_tensor_batch,
        meta_info=merged_meta_info,
    )


def extract_ref_rollout_metadata(
    request_batch: DataProto,
) -> dict[str, np.ndarray]:
    missing_keys = [
        key for key in _REF_ROLLOUT_METADATA_KEYS
        if key not in request_batch.non_tensor_batch
    ]
    if missing_keys:
        raise KeyError(
            "ref reward rollout request batch is missing metadata keys: "
            f"{missing_keys}."
        )

    return {
        key: request_batch.non_tensor_batch[key]
        for key in _REF_ROLLOUT_METADATA_KEYS
    }


def pad_ref_rollout_input_for_agent_workers(
    request_batch: DataProto,
    *,
    num_workers: int,
) -> tuple[DataProto, int]:
    """Pad ref rollout requests so VERL AgentLoop can split them equally."""

    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}.")

    original_size = len(request_batch)
    if original_size == 0:
        raise ValueError("Cannot pad an empty ref rollout request batch.")

    remainder = original_size % num_workers
    if remainder == 0:
        return request_batch, original_size

    pad_size = num_workers - remainder
    pad_indices = np.arange(pad_size) % original_size
    padded_non_tensors = {
        key: np.concatenate([values, values[pad_indices]], axis=0)
        for key, values in request_batch.non_tensor_batch.items()
    }

    return (
        DataProto(
            non_tensor_batch=padded_non_tensors,
            meta_info=dict(request_batch.meta_info),
        ),
        original_size,
    )


def truncate_ref_rollout_output(
    rollout_output: DataProto,
    *,
    size: int,
) -> DataProto:
    """Drop padded ref rollout outputs before grouping and persistence."""

    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}.")
    if len(rollout_output) < size:
        raise ValueError(
            f"Cannot truncate rollout output of size {len(rollout_output)} "
            f"to larger size {size}."
        )
    if len(rollout_output) == size:
        return rollout_output

    batch = rollout_output.batch[:size] if rollout_output.batch is not None else None
    non_tensor_batch = {
        key: values[:size]
        for key, values in rollout_output.non_tensor_batch.items()
    }
    return DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        meta_info=dict(rollout_output.meta_info),
    )


def attach_ref_rollout_metadata(
    *,
    rollout_output: DataProto,
    metadata: Mapping[str, np.ndarray],
) -> DataProto:
    if rollout_output.batch is None:
        raise ValueError("rollout_output.batch is required.")

    batch_size = len(rollout_output)
    output_non_tensors = rollout_output.non_tensor_batch

    missing_keys = [
        key for key in _REF_ROLLOUT_METADATA_KEYS
        if key not in metadata
    ]
    if missing_keys:
        raise KeyError(
            f"ref reward rollout metadata is missing required keys: {missing_keys}."
        )

    for key in _REF_ROLLOUT_METADATA_KEYS:
        values = metadata[key]
        if len(values) != batch_size:
            raise ValueError(
                f"ref reward rollout metadata[{key!r}] has length {len(values)}, "
                f"expected {batch_size}."
            )

        if key not in output_non_tensors:
            output_non_tensors[key] = values
            continue

        existing = output_non_tensors[key]
        if len(existing) != batch_size:
            raise ValueError(
                f"rollout_output.non_tensor_batch[{key!r}] has length "
                f"{len(existing)}, expected {batch_size}."
            )
        if existing.tolist() != values.tolist():
            raise ValueError(
                f"rollout_output.non_tensor_batch[{key!r}] does not match "
                "the corresponding ref rollout request metadata."
            )

    return rollout_output


def ref_rollout_output_to_store_rows(
    *,
    rollout_output: DataProto,
    tokenizer: Any,
    ref_version: str,
    num_ref_completions: int,
) -> list[dict[str, Any]]:
    if rollout_output.batch is None:
        raise ValueError("rollout_output.batch is required.")

    if num_ref_completions <= 0:
        raise ValueError(
            f"num_ref_completions must be positive, got {num_ref_completions}."
        )

    required_batch_keys = [
        K.PROMPTS,
        K.RESPONSES,
        K.ATTENTION_MASK,
        "rm_scores",
    ]
    for key in required_batch_keys:
        if key not in rollout_output.batch:
            raise KeyError(f"rollout_output.batch is missing required key {key!r}.")

    required_non_tensor_keys = [
        K.PROMPT_ID,
        K.DATASET_INDEX,
        K.REF_COMPLETION_INDEX,
    ]
    for key in required_non_tensor_keys:
        if key not in rollout_output.non_tensor_batch:
            raise KeyError(f"rollout_output.non_tensor_batch is missing {key!r}.")

    rm_scores = rollout_output.batch["rm_scores"].float()
    responses = rollout_output.batch[K.RESPONSES]
    if rm_scores.shape != responses.shape:
        raise ValueError(
            f"rm_scores shape {tuple(rm_scores.shape)} does not match responses "
            f"shape {tuple(responses.shape)}."
        )

    rewards = rm_scores.sum(dim=-1).detach().cpu().float()
    reward_extra_keys = (rollout_output.meta_info or {}).get("reward_extra_keys", [])

    grouped: dict[str, dict[str, Any]] = {}
    for index in range(len(rollout_output)):
        prompt_id = str(rollout_output.non_tensor_batch[K.PROMPT_ID][index])
        dataset_index = int(rollout_output.non_tensor_batch[K.DATASET_INDEX][index])
        ref_completion_index = int(
            rollout_output.non_tensor_batch[K.REF_COMPLETION_INDEX][index]
        )
        completion = _decode_response(
            rollout_output=rollout_output,
            index=index,
            tokenizer=tokenizer,
        )
        reward_extra_info = {
            key: _to_debug_value(rollout_output.non_tensor_batch[key][index])
            for key in reward_extra_keys
            if key in rollout_output.non_tensor_batch
        }

        group = grouped.setdefault(
            prompt_id,
            {
                "prompt_id": prompt_id,
                K.DATASET_INDEX: dataset_index,
                K.REF_VERSION: str(ref_version),
                "_items": {},
            },
        )
        if group[K.DATASET_INDEX] != dataset_index:
            raise ValueError(
                f"prompt_id={prompt_id!r} has inconsistent dataset_index values: "
                f"{group[K.DATASET_INDEX]} and {dataset_index}."
            )
        if ref_completion_index in group["_items"]:
            raise ValueError(
                f"prompt_id={prompt_id!r} has duplicate ref_completion_index="
                f"{ref_completion_index}."
            )

        group["_items"][ref_completion_index] = {
            "completion": completion,
            "reward": float(rewards[index].item()),
            "reward_extra_info": reward_extra_info or None,
        }

    rows: list[dict[str, Any]] = []
    for group in grouped.values():
        items = group.pop("_items")
        expected_indices = set(range(num_ref_completions))
        if set(items) != expected_indices:
            raise ValueError(
                f"prompt_id={group['prompt_id']!r} has ref completion indices "
                f"{sorted(items)}, expected {sorted(expected_indices)}."
            )

        ordered_items = [items[index] for index in range(num_ref_completions)]
        rows.append(
            {
                "prompt_id": group["prompt_id"],
                K.DATASET_INDEX: group[K.DATASET_INDEX],
                K.REF_VERSION: group[K.REF_VERSION],
                "ref_completions": [
                    item["completion"]
                    for item in ordered_items
                ],
                "ref_rewards": [
                    item["reward"]
                    for item in ordered_items
                ],
                "reward_extra_info": [
                    item["reward_extra_info"]
                    for item in ordered_items
                ],
            }
        )

    rows.sort(key=lambda row: int(row[K.DATASET_INDEX]))
    return rows


def _ref_trajectory_id(
    *,
    prompt_id: str,
    ref_version: str,
    ref_completion_index: int,
) -> str:
    return f"{prompt_id}::ref::{ref_version}::{ref_completion_index}"


def _decode_response(
    *,
    rollout_output: DataProto,
    index: int,
    tokenizer: Any,
) -> str:
    batch = rollout_output.batch
    responses = batch[K.RESPONSES][index]
    response_len = int(responses.shape[0])
    prompt_len = int(batch[K.PROMPTS].shape[1])
    response_attention = batch[K.ATTENTION_MASK][
        index,
        prompt_len:prompt_len + response_len,
    ]
    valid_response_len = int(
        response_attention.detach().cpu().long().sum().item()
    )
    valid_response_len = max(0, min(valid_response_len, response_len))

    token_ids = responses[:valid_response_len].detach().cpu().tolist()
    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        return " ".join(str(token_id) for token_id in token_ids)

    return str(decode(token_ids, skip_special_tokens=True))


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
