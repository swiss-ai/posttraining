from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from verl.protocol import DataProto

from batch import keys as K


def sequence_rewards_from_verl_output(
    reward_output: DataProto | torch.Tensor | Mapping[str, Any],
    *,
    batch_size: int | None = None,
    reward_key: str = K.REWARD_TENSOR,
) -> torch.Tensor:
    """Extract QRPO sequence rewards from VERL reward output.

    QRPO uses one scalar reward per trajectory: shape [B].

    Supported inputs:
      1. torch.Tensor [B]
         Already sequence-level rewards.

      2. torch.Tensor [B, T]
         VERL-style reward tensor. For sequence-level tasks, VERL often stores
         the scalar reward at the final valid response token. Summing over T
         recovers the sequence reward.

      3. DataProto with batch[reward_key]
         Same tensor rules as above.

      4. Mapping with reward_key
         For reward managers that return {"reward_tensor": tensor, ...}.

    The output is always float32.
    """

    reward_tensor = _extract_reward_tensor(
        reward_output,
        reward_key=reward_key,
    ).float()

    if reward_tensor.ndim == 1:
        rewards = reward_tensor
    elif reward_tensor.ndim == 2:
        rewards = reward_tensor.sum(dim=-1)
    else:
        raise ValueError(
            f"Reward tensor must have shape [B] or [B, T], got {tuple(reward_tensor.shape)}."
        )

    if batch_size is not None and rewards.shape[0] != batch_size:
        raise ValueError(
            f"Sequence rewards must have shape ({batch_size},), got {tuple(rewards.shape)}."
        )

    return rewards


def attach_sequence_rewards(
    data: DataProto,
    rewards: torch.Tensor,
) -> DataProto:
    """Attach QRPO sequence rewards to a DataProto.

    This mutates and returns `data`, following VERL's common pattern of adding
    fields to DataProto as the training pipeline progresses.
    """

    if data.batch is None:
        raise ValueError("DataProto.batch is required.")

    rewards = rewards.float()

    if rewards.ndim != 1 or rewards.shape[0] != len(data):
        raise ValueError(
            f"rewards must have shape ({len(data)},), got {tuple(rewards.shape)}."
        )

    data.batch[K.TRAJECTORY_REWARD] = rewards
    return data


def extract_and_attach_sequence_rewards(
    data: DataProto,
    reward_output: DataProto | torch.Tensor | Mapping[str, Any],
    *,
    reward_key: str = K.REWARD_TENSOR,
) -> DataProto:
    """Extract sequence rewards from VERL output and attach them to `data`."""

    rewards = sequence_rewards_from_verl_output(
        reward_output,
        batch_size=len(data),
        reward_key=reward_key,
    )
    return attach_sequence_rewards(data, rewards)


def _extract_reward_tensor(
    reward_output: DataProto | torch.Tensor | Mapping[str, Any],
    *,
    reward_key: str,
) -> torch.Tensor:
    if isinstance(reward_output, torch.Tensor):
        return reward_output

    if isinstance(reward_output, DataProto):
        if reward_output.batch is None:
            raise ValueError("Reward DataProto has no tensor batch.")

        if K.TRAJECTORY_REWARD in reward_output.batch:
            return reward_output.batch[K.TRAJECTORY_REWARD]

        if reward_key in reward_output.batch:
            return reward_output.batch[reward_key]

        raise KeyError(
            f"Reward DataProto.batch must contain {K.TRAJECTORY_REWARD!r} "
            f"or {reward_key!r}."
        )

    if isinstance(reward_output, Mapping):
        if K.TRAJECTORY_REWARD in reward_output:
            value = reward_output[K.TRAJECTORY_REWARD]
        elif reward_key in reward_output:
            value = reward_output[reward_key]
        else:
            raise KeyError(
                f"Reward mapping must contain {K.TRAJECTORY_REWARD!r} or {reward_key!r}."
            )

        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Reward value must be a torch.Tensor, got {type(value).__name__}."
            )

        return value

    raise TypeError(
        "reward_output must be a DataProto, torch.Tensor, or mapping, "
        f"got {type(reward_output).__name__}."
    )
