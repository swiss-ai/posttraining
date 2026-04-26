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
      data_source: string stored in non_tensor_batch
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
        "source": K.SOURCE_ONLINE,
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

    Expected VERL AgentLoop batch keys:
      prompts
      responses
      response_mask
      input_ids
      attention_mask
      position_ids

    Output QRPO training keys:
      input_ids
      attention_mask
      position_ids
      loss_mask
      trajectory_reward
      ref_rewards

    The loss_mask is the response_mask placed into the full input sequence:
      prompt tokens: 0
      LLM-generated response tokens: 1
      tool/environment response tokens: 0
      padding: 0
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

    if K.REF_REWARDS not in rollout_output.non_tensor_batch:
        raise KeyError(f"rollout_output.non_tensor_batch is missing {K.REF_REWARDS!r}.")
    if K.PROMPT_ID not in rollout_output.non_tensor_batch:
        raise KeyError(f"rollout_output.non_tensor_batch is missing {K.PROMPT_ID!r}.")
    if K.TRAJECTORY_ID not in rollout_output.non_tensor_batch:
        raise KeyError(f"rollout_output.non_tensor_batch is missing {K.TRAJECTORY_ID!r}.")

    input_ids = batch[K.INPUT_IDS]
    attention_mask = batch[K.ATTENTION_MASK]
    position_ids = batch[K.POSITION_IDS]
    prompts = batch[K.PROMPTS]
    response_mask = batch[K.RESPONSE_MASK]

    if rewards.ndim != 1 or rewards.shape[0] != len(rollout_output):
        raise ValueError(
            f"rewards must have shape ({len(rollout_output)},), got {tuple(rewards.shape)}."
        )

    prompt_len = prompts.shape[1]
    response_len = response_mask.shape[1]

    if response_mask.shape != batch[K.RESPONSES].shape:
        raise ValueError(
            f"response_mask shape {tuple(response_mask.shape)} does not match "
            f"responses shape {tuple(batch[K.RESPONSES].shape)}."
        )

    if input_ids.shape[1] != prompt_len + response_len:
        raise ValueError(
            "input_ids sequence length must equal prompt_len + response_len: "
            f"{input_ids.shape[1]} != {prompt_len} + {response_len}."
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

    loss_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    loss_mask[:, prompt_len : prompt_len + response_len] = response_mask.bool()

    ref_rewards = torch.tensor(
        [tuple(x) for x in rollout_output.non_tensor_batch[K.REF_REWARDS]],
        dtype=torch.float32,
    )

    tensors = {
        K.INPUT_IDS: input_ids,
        K.ATTENTION_MASK: attention_mask,
        K.POSITION_IDS: position_ids,
        K.LOSS_MASK: loss_mask,
        K.TRAJECTORY_REWARD: rewards.float(),
        K.REF_REWARDS: ref_rewards,
    }

    non_tensors = {
        K.PROMPT_ID: rollout_output.non_tensor_batch[K.PROMPT_ID],
        K.TRAJECTORY_ID: rollout_output.non_tensor_batch[K.TRAJECTORY_ID],
        K.SOURCE: np.asarray([K.SOURCE_ONLINE] * len(rollout_output), dtype=object),
    }

    merged_meta_info = {
        "qrpo_batch_format": "full_sequence_with_loss_mask",
        "source": K.SOURCE_ONLINE,
    }
    if meta_info is not None:
        merged_meta_info.update(meta_info)

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info=merged_meta_info,
    )