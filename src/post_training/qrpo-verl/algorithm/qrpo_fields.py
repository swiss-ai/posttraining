from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from verl.protocol import DataProto

from batch import keys as K


def compute_ref_quantiles(
    *,
    trajectory_rewards: torch.Tensor,
    ref_rewards: torch.Tensor,
) -> torch.Tensor:
    """Compute empirical reference quantiles F_ref(r).

    Uses the inclusive empirical CDF:

        F_ref(r) = mean(ref_rewards <= r)

    Shapes:
      trajectory_rewards: [B]
      ref_rewards:        [B, M]
      output:             [B]

    No clipping is applied. For the identity quantile transform, exact 0 and 1
    are valid values.
    """

    if trajectory_rewards.ndim != 1:
        raise ValueError(
            f"trajectory_rewards must have shape [B], got {tuple(trajectory_rewards.shape)}."
        )

    if ref_rewards.ndim != 2:
        raise ValueError(
            f"ref_rewards must have shape [B, M], got {tuple(ref_rewards.shape)}."
        )

    if ref_rewards.shape[0] != trajectory_rewards.shape[0]:
        raise ValueError(
            "trajectory_rewards and ref_rewards must have the same batch size, got "
            f"{trajectory_rewards.shape[0]} and {ref_rewards.shape[0]}."
        )

    return (ref_rewards <= trajectory_rewards[:, None]).float().mean(dim=-1)


def compute_response_lengths(response_mask: torch.Tensor) -> torch.Tensor:
    """Count trainable model-output tokens in the VERL response block.

    For tool-call and multi-turn trajectories:

      assistant/model tokens: 1
      tool/env/user/pad tokens: 0

    Shapes:
      response_mask: [B, response_len]
      output:        [B]
    """

    if response_mask.ndim != 2:
        raise ValueError(f"response_mask must have shape [B, T], got {tuple(response_mask.shape)}.")

    lengths = response_mask.bool().sum(dim=-1).float()

    if not torch.all(lengths > 0):
        raise ValueError("All trajectories must have at least one trainable response token.")

    return lengths


def identity_quantile_log_partition(beta: torch.Tensor | float) -> torch.Tensor:
    """Compute log Z for identity-transformed quantile reward.

    For q ~ Uniform(0, 1):

        Z = E[exp(q / beta)]
          = beta * (exp(1 / beta) - 1)

    Therefore:

        log Z = log(beta) + log(expm1(1 / beta))

    This function is mainly useful for diagnostics. The actual QRPO residual
    should use identity_quantile_beta_log_partition(...) directly.
    """

    beta_tensor = _as_float_tensor(beta)

    if not torch.all(beta_tensor > 0):
        raise ValueError("beta must be positive.")

    t = 1.0 / beta_tensor

    flat_t = t.reshape(-1)
    flat_beta = beta_tensor.reshape(-1)
    flat_out = torch.empty_like(flat_t)

    small = flat_t <= math.log(10.0)

    if small.any():
        flat_out[small] = (
            torch.log(flat_beta[small])
            + torch.log(torch.expm1(flat_t[small]))
        )

    if (~small).any():
        large_t = flat_t[~small]
        large_beta = flat_beta[~small]

        # log Z = log(beta) + t + log1p(-exp(-t))
        # This avoids computing exp(t).
        flat_out[~small] = (
            torch.log(large_beta)
            + large_t
            + torch.log1p(-torch.exp(-large_t))
        )

    return flat_out.reshape_as(beta_tensor)


def identity_quantile_beta_log_partition(beta: torch.Tensor | float) -> torch.Tensor:
    """Compute beta * log Z for identity-transformed quantile reward.

    This is the term that appears directly in the stable QRPO residual:

        residual = R_q - beta * log Z - beta * log(pi / pi_ref)

    For q ~ Uniform(0, 1):

        Z = beta * (exp(1 / beta) - 1)

    so:

        beta * log Z
        = beta * log(beta) + beta * log(expm1(1 / beta))

    For small beta, we use:

        log(expm1(1 / beta))
        = 1 / beta + log1p(-exp(-1 / beta))

    therefore:

        beta * log Z
        = beta * log(beta) + 1 + beta * log1p(-exp(-1 / beta))

    This avoids forming a very large log Z and then multiplying it back by beta.
    """

    beta_tensor = _as_float_tensor(beta)

    if not torch.all(beta_tensor > 0):
        raise ValueError("beta must be positive.")

    t = 1.0 / beta_tensor

    flat_beta = beta_tensor.reshape(-1)
    flat_t = t.reshape(-1)
    flat_out = torch.empty_like(flat_t)

    small = flat_t <= math.log(10.0)

    if small.any():
        # beta * (log(beta) + log(expm1(1 / beta)))
        flat_out[small] = flat_beta[small] * (
            torch.log(flat_beta[small])
            + torch.log(torch.expm1(flat_t[small]))
        )

    if (~small).any():
        large_beta = flat_beta[~small]
        large_t = flat_t[~small]

        # beta * log(beta) + beta * (t + log1p(-exp(-t)))
        # = beta * log(beta) + 1 + beta * log1p(-exp(-t))
        flat_out[~small] = (
            large_beta * torch.log(large_beta)
            + 1.0
            + large_beta * torch.log1p(-torch.exp(-large_t))
        )

    return flat_out.reshape_as(beta_tensor)


def add_qrpo_fields(
    data: DataProto,
    *,
    config: Mapping[str, Any],
) -> DataProto:
    """Attach QRPO algorithm fields to a training DataProto.

    Required tensor keys:
      trajectory_reward: [B]
      ref_rewards:       [B, M]

    Additionally required if length_normalization=true:
      response_mask:     [B, response_len]

    Added tensor keys:
      ref_quantile:        [B]
      transformed_reward:  [B]
      trajectory_length:   [B]
      effective_beta:      [B]
      log_partition:       [B]  # diagnostic
      beta_log_partition:  [B]  # use this in the loss

    Currently implemented transform:
      identity:
        transformed_reward = ref_quantile

    Length normalization:
      false:
        effective_beta = beta

      true:
        effective_beta = beta / response_mask.sum(-1)

    Stable actor-loss residual:

        log_ratio = log pi_theta(y|x) - log pi_ref(y|x)

        residual = (
            transformed_reward
            - beta_log_partition
            - effective_beta * log_ratio
        )

        loss = residual.square()
    """

    if data.batch is None:
        raise ValueError("DataProto.batch is required.")

    beta = config.get("beta")
    if beta is None:
        raise ValueError("QRPO config must contain 'beta'.")

    beta = float(beta)
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}.")

    transform = config.get("transform", "identity")
    if transform != "identity":
        raise NotImplementedError(
            f"Unsupported QRPO transform {transform!r}. Only 'identity' is implemented."
        )

    length_normalization = config.get("length_normalization", True)
    if not isinstance(length_normalization, bool):
        raise ValueError(
            "QRPO config field 'length_normalization' must be a boolean."
        )

    if K.TRAJECTORY_REWARD not in data.batch:
        raise KeyError(f"DataProto.batch is missing {K.TRAJECTORY_REWARD!r}.")

    if K.REF_REWARDS not in data.batch:
        raise KeyError(f"DataProto.batch is missing {K.REF_REWARDS!r}.")

    trajectory_rewards = data.batch[K.TRAJECTORY_REWARD].float()
    ref_rewards = data.batch[K.REF_REWARDS].float()

    ref_quantile = compute_ref_quantiles(
        trajectory_rewards=trajectory_rewards,
        ref_rewards=ref_rewards,
    )

    transformed_reward = ref_quantile

    if length_normalization:
        if K.RESPONSE_MASK not in data.batch:
            raise KeyError(
                f"Length-normalized QRPO requires DataProto.batch[{K.RESPONSE_MASK!r}]."
            )

        trajectory_length = compute_response_lengths(data.batch[K.RESPONSE_MASK]).to(
            device=trajectory_rewards.device
        )
        effective_beta = torch.full_like(trajectory_length, beta) / trajectory_length
    else:
        # We still expose trajectory_length for logging/debugging when available,
        # but it does not affect effective_beta.
        if K.RESPONSE_MASK in data.batch:
            trajectory_length = compute_response_lengths(data.batch[K.RESPONSE_MASK]).to(
                device=trajectory_rewards.device
            )
        else:
            trajectory_length = torch.ones_like(trajectory_rewards)

        effective_beta = torch.full_like(trajectory_rewards, beta)

    log_partition = identity_quantile_log_partition(effective_beta).to(
        device=trajectory_rewards.device
    )
    beta_log_partition = identity_quantile_beta_log_partition(effective_beta).to(
        device=trajectory_rewards.device
    )

    data.batch[K.REF_QUANTILE] = ref_quantile
    data.batch[K.TRANSFORMED_REWARD] = transformed_reward
    data.batch[K.TRAJECTORY_LENGTH] = trajectory_length
    data.batch[K.EFFECTIVE_BETA] = effective_beta
    data.batch[K.LOG_PARTITION] = log_partition
    data.batch[K.BETA_LOG_PARTITION] = beta_log_partition

    return data


def _as_float_tensor(value: torch.Tensor | float) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.float()

    return torch.tensor(float(value), dtype=torch.float32)
