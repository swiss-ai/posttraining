from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from verl.protocol import DataProto

from batch import keys as K
from batch.candidate_plan import OnlineRolloutRequest


def online_rollout_requests_to_dataproto(
    *,
    requests: Sequence[OnlineRolloutRequest],
    config: Mapping[str, Any],
    meta_info: dict[str, Any] | None = None,
) -> DataProto:
    """Convert online rollout requests to VERL AgentLoop input DataProto.

    This adapter intentionally does not tokenize prompts. VERL AgentLoop handles
    chat-template application, rollout execution, tool calls, and trajectory
    postprocessing.

    Required config:
      agent_name: name of the registered VERL agent loop to use

    Optional config:
      data_source: string stored in non_tensor_batch if not None
      reward_model: object/dict stored per sample if your reward loop needs it
      validate: bool stored in meta_info
    """

    requests = tuple(requests)

    if not requests:
        raise ValueError("Cannot build rollout DataProto from an empty request list.")

    agent_name = config.get("agent_name")
    if not agent_name:
        raise ValueError("rollout config must contain a non-empty 'agent_name'.")

    data_source = config.get("data_source", None)
    reward_model = config.get("reward_model", None)

    raw_prompt = np.empty(len(requests), dtype=object)
    raw_prompt[:] = [tuple(request.prompt_messages) for request in requests]

    ref_rewards = np.empty(len(requests), dtype=object)
    ref_rewards[:] = [tuple(request.ref_rewards) for request in requests]

    non_tensor_batch: dict[str, np.ndarray] = {
        K.RAW_PROMPT: raw_prompt,
        K.AGENT_NAME: np.asarray([agent_name] * len(requests), dtype=object),
        K.PROMPT_ID: np.asarray([request.prompt_id for request in requests], dtype=object),
        K.TRAJECTORY_ID: np.asarray([request.trajectory_id for request in requests], dtype=object),
        K.SOURCE: np.asarray([K.SOURCE_ONLINE] * len(requests), dtype=object),
        K.REF_REWARDS: ref_rewards,
    }

    if data_source is not None:
        non_tensor_batch[K.DATA_SOURCE] = np.asarray([data_source] * len(requests), dtype=object)

    if reward_model is not None:
        reward_models = np.empty(len(requests), dtype=object)
        reward_models[:] = [reward_model] * len(requests)
        non_tensor_batch[K.REWARD_MODEL] = reward_models

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


def online_rollout_output_to_train_dataproto(
    *,
    rollout_output: DataProto,
    rewards: torch.Tensor,
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

    # Prefer explicit meta_info passed by the trainer, then rollout_output.meta_info
    if rollout_output.meta_info is not None and "temperature" in rollout_output.meta_info:
        merged_meta_info["temperature"] = float(rollout_output.meta_info["temperature"])

    if meta_info is not None:
        merged_meta_info.update(meta_info)

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info=merged_meta_info,
    )
