from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from verl.protocol import DataProto

from batch import keys as K
from batch.candidate_plan import OnlineRolloutRequest

_ONLINE_ROLLOUT_TRAIN_METADATA_KEYS = (
    K.PROMPT_ID,
    K.TRAJECTORY_ID,
    K.REF_REWARDS,
)


@dataclass(frozen=True)
class OnlineRolloutCandidatePlan:
    """Mapping from online train slots to rollout candidate requests."""

    training_requests: tuple[OnlineRolloutRequest, ...]
    candidate_requests: tuple[OnlineRolloutRequest, ...]
    candidate_counts_per_train_slot: tuple[int, ...]
    candidates_per_train_sample: int
    actual_probability: float
    selection: str

    @property
    def has_extra_candidates(self) -> bool:
        return any(count > 1 for count in self.candidate_counts_per_train_slot)


def online_rollout_requests_to_dataproto(
    *,
    requests: Sequence[OnlineRolloutRequest],
    config: Mapping[str, Any],
    meta_info: dict[str, Any] | None = None,
) -> DataProto:
    requests = tuple(requests)

    if not requests:
        raise ValueError("Cannot build rollout DataProto from an empty request list.")

    agent_name = config.get("agent_name", None)
    data_source = config.get("data_source")
    if not data_source:
        raise ValueError(
            "online_rollout.data_source is required for online reward computation."
        )

    raw_prompt = np.empty(len(requests), dtype=object)
    raw_prompt[:] = [tuple(request.prompt_messages) for request in requests]

    ref_rewards = np.empty(len(requests), dtype=object)
    ref_rewards[:] = [tuple(request.ref_rewards) for request in requests]

    reward_models = np.empty(len(requests), dtype=object)
    extra_infos = np.empty(len(requests), dtype=object)

    for i, request in enumerate(requests):
        prompt_messages = tuple(request.prompt_messages)
        train_trajectory_id = request.train_trajectory_id or request.trajectory_id
        candidate_index = (
            0
            if request.candidate_index is None
            else int(request.candidate_index)
        )
        request_candidates_per_train_sample = (
            1
            if request.candidates_per_train_sample is None
            else int(request.candidates_per_train_sample)
        )

        # VERL's naive reward-manager adapter requires reward_model["ground_truth"].
        # Our active UltraFeedback reward will mostly use extra_info["prompt"],
        # but we keep ground_truth populated for compatibility and validation.
        reward_models[i] = {
            "ground_truth": {
                "prompt": prompt_messages,
                "prompt_id": request.prompt_id,
                "trajectory_id": request.trajectory_id,
                "train_trajectory_id": train_trajectory_id,
                "candidate_index": candidate_index,
                "candidates_per_train_sample": request_candidates_per_train_sample,
            }
        }

        extra_infos[i] = {
            "prompt": prompt_messages,
            "prompt_id": request.prompt_id,
            "trajectory_id": request.trajectory_id,
            "train_trajectory_id": train_trajectory_id,
            "candidate_index": candidate_index,
            "candidates_per_train_sample": request_candidates_per_train_sample,
            "online_index": request.online_index,
            "ref_rewards": tuple(request.ref_rewards),
        }

    non_tensor_batch: dict[str, np.ndarray] = {
        K.RAW_PROMPT: raw_prompt,
        K.PROMPT_ID: np.asarray([request.prompt_id for request in requests], dtype=object),
        K.TRAJECTORY_ID: np.asarray([request.trajectory_id for request in requests], dtype=object),
        K.SOURCE: np.asarray([K.SOURCE_ONLINE] * len(requests), dtype=object),
        K.DATA_SOURCE: np.asarray([data_source] * len(requests), dtype=object),
        K.REWARD_MODEL: reward_models,
        K.EXTRA_INFO: extra_infos,
        K.REF_REWARDS: ref_rewards,
    }

    if agent_name is not None:
        non_tensor_batch[K.AGENT_NAME] = np.asarray([agent_name] * len(requests), dtype=object)

    merged_meta_info = {
        "qrpo_batch_format": "online_rollout_requests",
        "validate": bool(config.get("validate", False)),
    }
    if meta_info is not None:
        merged_meta_info.update(meta_info)

    return DataProto(
        non_tensor_batch=non_tensor_batch,
        meta_info=merged_meta_info,
    )


def expand_online_rollout_candidate_requests(
    *,
    requests: Sequence[OnlineRolloutRequest],
    config: Mapping[str, Any],
    meta_info: Mapping[str, Any] | None = None,
    divisible_by: int = 1,
) -> OnlineRolloutCandidatePlan:
    """Expand online train slots into rollout candidates for best-of-k."""

    training_requests = tuple(requests)
    selection_config = config.get("candidate_selection", {}) or {}
    enabled = bool(selection_config.get("enabled", False))
    selection = str(selection_config.get("selection", "best_reward"))
    candidates_per_train_sample = _resolve_candidates_per_train_sample(
        selection_config
    )
    probability = float(selection_config.get("probability", 1.0))
    seed = int(selection_config.get("seed", 0))

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "online_rollout.candidate_selection.probability must be in [0, 1]."
        )
    if selection != "best_reward":
        raise ValueError(
            "online_rollout.candidate_selection.selection must be 'best_reward'."
        )
    if divisible_by <= 0:
        raise ValueError(f"divisible_by must be positive, got {divisible_by}.")

    if not enabled or candidates_per_train_sample == 1 or probability == 0.0:
        return OnlineRolloutCandidatePlan(
            training_requests=training_requests,
            candidate_requests=training_requests,
            candidate_counts_per_train_slot=tuple([1] * len(training_requests)),
            candidates_per_train_sample=1,
            actual_probability=0.0,
            selection=selection,
        )

    step = 0 if meta_info is None else int(meta_info.get("global_steps", 0) or 0)
    rng = random.Random(seed + step * 1_000_003)
    scores = [rng.random() for _ in training_requests]
    selected_count = sum(score < probability for score in scores)
    selected_count = _adjust_best_of_k_count_for_divisibility(
        base_count=len(training_requests),
        selected_count=selected_count,
        max_selected_count=len(training_requests),
        extra_candidates_per_selected=candidates_per_train_sample - 1,
        divisible_by=divisible_by,
    )
    selected_indices = set(
        sorted(range(len(training_requests)), key=lambda index: scores[index])[
            :selected_count
        ]
    )

    candidate_requests: list[OnlineRolloutRequest] = []
    candidate_counts_per_train_slot: list[int] = []

    for index, request in enumerate(training_requests):
        count = candidates_per_train_sample if index in selected_indices else 1
        candidate_counts_per_train_slot.append(count)

        if count == 1:
            candidate_requests.append(request)
            continue

        for candidate_index in range(count):
            candidate_requests.append(
                replace(
                    request,
                    trajectory_id=(
                        f"{request.trajectory_id}::candidate_{candidate_index}"
                    ),
                    train_trajectory_id=request.trajectory_id,
                    candidate_index=candidate_index,
                    candidates_per_train_sample=count,
                )
            )

    return OnlineRolloutCandidatePlan(
        training_requests=training_requests,
        candidate_requests=tuple(candidate_requests),
        candidate_counts_per_train_slot=tuple(candidate_counts_per_train_slot),
        candidates_per_train_sample=candidates_per_train_sample,
        actual_probability=selected_count / max(1, len(training_requests)),
        selection=selection,
    )


def select_online_rollout_candidates(
    *,
    rollout_output: DataProto,
    rewards: torch.Tensor,
    candidate_plan: OnlineRolloutCandidatePlan,
) -> tuple[DataProto, torch.Tensor, dict[str, float]]:
    """Select one rollout output per online train slot."""

    if not candidate_plan.has_extra_candidates:
        return rollout_output, rewards, {}

    if rewards.ndim != 1 or rewards.shape[0] != len(candidate_plan.candidate_requests):
        raise ValueError(
            "candidate rewards must have one value per candidate request, got "
            f"{tuple(rewards.shape)} for {len(candidate_plan.candidate_requests)} requests."
        )
    if len(rollout_output) != len(candidate_plan.candidate_requests):
        raise ValueError(
            "rollout_output length must match candidate request count, got "
            f"{len(rollout_output)} and {len(candidate_plan.candidate_requests)}."
        )

    selected_indices: list[int] = []
    selected_rewards: list[torch.Tensor] = []
    best_minus_first: list[float] = []
    offset = 0

    for count in candidate_plan.candidate_counts_per_train_slot:
        if count == 1:
            selected_index = offset
        else:
            group_rewards = rewards[offset : offset + count]
            relative_index = int(torch.argmax(group_rewards).detach().cpu().item())
            selected_index = offset + relative_index
            best_minus_first.append(
                float(
                    (group_rewards[relative_index] - group_rewards[0])
                    .detach()
                    .cpu()
                    .item()
                )
            )

        selected_indices.append(selected_index)
        selected_rewards.append(rewards[selected_index])
        offset += count

    if offset != len(candidate_plan.candidate_requests):
        raise ValueError(
            "candidate_counts_per_train_slot does not cover all candidate "
            f"requests: covered {offset}, expected "
            f"{len(candidate_plan.candidate_requests)}."
        )

    selected_output = _index_dataproto_rows(rollout_output, selected_indices)
    selected_output.non_tensor_batch[K.TRAJECTORY_ID] = np.asarray(
        [request.trajectory_id for request in candidate_plan.training_requests],
        dtype=object,
    )
    selected_rewards_tensor = torch.stack(selected_rewards).to(device=rewards.device)

    metrics = {
        "online_candidate_selection/train_slots": float(
            len(candidate_plan.training_requests)
        ),
        "online_candidate_selection/rollout_candidates": float(
            len(candidate_plan.candidate_requests)
        ),
        "online_candidate_selection/best_of_k_slots": float(
            sum(count > 1 for count in candidate_plan.candidate_counts_per_train_slot)
        ),
        "online_candidate_selection/actual_probability": float(
            candidate_plan.actual_probability
        ),
        "online_candidate_selection/candidates_per_train_sample": float(
            candidate_plan.candidates_per_train_sample
        ),
    }
    if best_minus_first:
        metrics["online_candidate_selection/best_minus_first_reward_mean"] = float(
            sum(best_minus_first) / len(best_minus_first)
        )

    return selected_output, selected_rewards_tensor, metrics


def _resolve_candidates_per_train_sample(
    selection_config: Mapping[str, Any],
) -> int:
    value = int(selection_config.get("candidates_per_train_sample", 1))

    if value <= 0:
        raise ValueError(
            "online_rollout.candidate_selection.candidates_per_train_sample "
            "must be positive."
        )

    return value


def extract_online_rollout_train_metadata(
    request_batch: DataProto,
) -> dict[str, np.ndarray]:
    """Extract request metadata that must survive into QRPO training.

    When VERL AgentLoop computes reward during rollout, it does not preserve the
    original request non-tensor batch on its output. QRPO still needs a narrow
    set of per-sample identifiers to build the online train batch.
    """

    missing_keys = [
        key for key in _ONLINE_ROLLOUT_TRAIN_METADATA_KEYS
        if key not in request_batch.non_tensor_batch
    ]
    if missing_keys:
        raise KeyError(
            "online rollout request batch is missing required QRPO training "
            f"metadata keys: {missing_keys}."
        )

    return {
        key: request_batch.non_tensor_batch[key]
        for key in _ONLINE_ROLLOUT_TRAIN_METADATA_KEYS
    }


def attach_online_rollout_train_metadata(
    *,
    rollout_output: DataProto,
    metadata: Mapping[str, np.ndarray],
) -> DataProto:
    """Attach QRPO training metadata to rollout output in-place."""

    if rollout_output.batch is None:
        raise ValueError("rollout_output.batch is required.")

    batch_size = len(rollout_output)
    output_non_tensors = rollout_output.non_tensor_batch

    missing_keys = [
        key for key in _ONLINE_ROLLOUT_TRAIN_METADATA_KEYS
        if key not in metadata
    ]
    if missing_keys:
        raise KeyError(
            "online rollout training metadata is missing required keys: "
            f"{missing_keys}."
        )

    for key in _ONLINE_ROLLOUT_TRAIN_METADATA_KEYS:
        values = metadata[key]
        if len(values) != batch_size:
            raise ValueError(
                f"online rollout training metadata[{key!r}] has length "
                f"{len(values)}, expected {batch_size}."
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
                f"rollout_output.non_tensor_batch[{key!r}] does not match the "
                "corresponding online rollout request metadata."
            )

    return rollout_output


def _adjust_best_of_k_count_for_divisibility(
    *,
    base_count: int,
    selected_count: int,
    max_selected_count: int,
    extra_candidates_per_selected: int,
    divisible_by: int,
) -> int:
    if divisible_by == 1:
        return selected_count

    valid_counts = [
        count
        for count in range(max_selected_count + 1)
        if (base_count + count * extra_candidates_per_selected) % divisible_by == 0
    ]
    if not valid_counts:
        raise ValueError(
            "Cannot make online rollout candidate count divisible by "
            f"{divisible_by}: base_count={base_count}, "
            f"extra_candidates_per_selected={extra_candidates_per_selected}, "
            f"max_selected_count={max_selected_count}."
        )

    return min(
        valid_counts,
        key=lambda count: (abs(count - selected_count), count > selected_count),
    )


def _index_dataproto_rows(data: DataProto, indices: Sequence[int]) -> DataProto:
    if data.batch is None:
        raise ValueError("DataProto.batch is required.")

    index_list = [int(index) for index in indices]
    tensors = {}
    for key, value in data.batch.items():
        index_tensor = torch.tensor(index_list, dtype=torch.long, device=value.device)
        tensors[key] = value.index_select(0, index_tensor)

    numpy_indices = np.asarray(index_list, dtype=np.int64)
    non_tensors = {
        key: values[numpy_indices]
        for key, values in data.non_tensor_batch.items()
    }

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info=dict(data.meta_info or {}),
    )


def online_rollout_output_to_train_dataproto(
    *,
    rollout_output: DataProto,
    rewards: torch.Tensor,
    temperature: float | None = None,
    meta_info: dict[str, Any] | None = None,
) -> DataProto:
    """Convert VERL AgentLoop output to QRPO online training DataProto.

    Expected VERL AgentLoop tensor keys:
      prompts
      responses
      response_mask
      input_ids
      attention_mask
      position_ids

    Output QRPO training tensor keys:
      prompts
      responses
      response_mask
      input_ids
      attention_mask
      position_ids
      trajectory_reward
      ref_rewards

    The response_mask is kept in VERL's native response-aligned layout:
      1 for LLM-generated response tokens
      0 for tool/environment response tokens and padding
    """

    if rollout_output.batch is None:
        raise ValueError("rollout_output.batch is required.")

    batch = rollout_output.batch
    required_batch_keys = [
        K.PROMPTS,
        K.RESPONSES,
        K.RESPONSE_MASK,
        K.INPUT_IDS,
        K.ATTENTION_MASK,
        K.POSITION_IDS,
    ]
    for key in required_batch_keys:
        if key not in batch:
            raise KeyError(f"rollout_output.batch is missing required key {key!r}.")

    required_non_tensor_keys = [
        K.REF_REWARDS,
        K.PROMPT_ID,
        K.TRAJECTORY_ID,
    ]
    for key in required_non_tensor_keys:
        if key not in rollout_output.non_tensor_batch:
            raise KeyError(f"rollout_output.non_tensor_batch is missing {key!r}.")

    prompts = batch[K.PROMPTS]
    responses = batch[K.RESPONSES]
    response_mask = batch[K.RESPONSE_MASK].bool()
    input_ids = batch[K.INPUT_IDS]
    attention_mask = batch[K.ATTENTION_MASK]
    position_ids = batch[K.POSITION_IDS]

    if rewards.ndim != 1 or rewards.shape[0] != len(rollout_output):
        raise ValueError(
            f"rewards must have shape ({len(rollout_output)},), got {tuple(rewards.shape)}."
        )

    if response_mask.shape != responses.shape:
        raise ValueError(
            f"response_mask shape {tuple(response_mask.shape)} does not match "
            f"responses shape {tuple(responses.shape)}."
        )

    expected_seq_len = prompts.shape[1] + responses.shape[1]
    if input_ids.shape[1] != expected_seq_len:
        raise ValueError(
            "input_ids sequence length must equal prompt_len + response_len: "
            f"{input_ids.shape[1]} != {prompts.shape[1]} + {responses.shape[1]}."
        )

    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"attention_mask shape {tuple(attention_mask.shape)} does not match "
            f"input_ids shape {tuple(input_ids.shape)}."
        )

    if position_ids.shape != input_ids.shape:
        raise ValueError(
            f"position_ids shape {tuple(position_ids.shape)} does not match "
            f"input_ids shape {tuple(input_ids.shape)}."
        )

    ref_rewards = torch.tensor(
        [tuple(x) for x in rollout_output.non_tensor_batch[K.REF_REWARDS]],
        dtype=torch.float32,
    )

    tensors = {
        K.PROMPTS: prompts,
        K.RESPONSES: responses,
        K.RESPONSE_MASK: response_mask,
        K.INPUT_IDS: input_ids,
        K.ATTENTION_MASK: attention_mask,
        K.POSITION_IDS: position_ids,
        K.TRAJECTORY_REWARD: rewards.float(),
        K.REF_REWARDS: ref_rewards,
    }

    non_tensors = {
        K.PROMPT_ID: rollout_output.non_tensor_batch[K.PROMPT_ID],
        K.TRAJECTORY_ID: rollout_output.non_tensor_batch[K.TRAJECTORY_ID],
        K.SOURCE: np.asarray([K.SOURCE_ONLINE] * len(rollout_output), dtype=object),
    }

    merged_meta_info = {
        "qrpo_batch_format": "verl_prompt_response",
    }

    if temperature is not None:
        merged_meta_info[K.TEMPERATURE] = float(temperature)
    elif (
        rollout_output.meta_info is not None
        and K.TEMPERATURE in rollout_output.meta_info
    ):
        merged_meta_info[K.TEMPERATURE] = float(
            rollout_output.meta_info[K.TEMPERATURE]
        )

    if meta_info is not None:
        merged_meta_info.update(meta_info)

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info=merged_meta_info,
    )
