from __future__ import annotations

from typing import Any, Mapping

import torch
from verl.protocol import DataProto

from batch import keys as K


def compute_sequence_log_ratio(
    *,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute sequence-level log πθ/πref over trainable tokens.

    Shapes:
      log_probs:      [B, T]
      ref_log_probs:  [B, T]
      loss_mask:      [B, T]

    The mask must mark exactly the model-output tokens included in the QRPO
    objective. Prompt tokens, tool outputs, environment messages, and padding
    should be zero.
    """

    if log_probs.shape != ref_log_probs.shape:
        raise ValueError(
            f"log_probs shape {tuple(log_probs.shape)} does not match "
            f"ref_log_probs shape {tuple(ref_log_probs.shape)}."
        )

    if loss_mask.shape != log_probs.shape:
        raise ValueError(
            f"loss_mask shape {tuple(loss_mask.shape)} does not match "
            f"log_probs shape {tuple(log_probs.shape)}."
        )

    if log_probs.ndim != 2:
        raise ValueError(f"log_probs must have shape [B, T], got {tuple(log_probs.shape)}.")

    mask = loss_mask.bool()
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("Every sample must have at least one trainable token.")

    return ((log_probs.float() - ref_log_probs.float()) * mask.float()).sum(dim=-1)


def compute_qrpo_residual(
    *,
    transformed_reward: torch.Tensor,
    beta_log_partition: torch.Tensor,
    effective_beta: torch.Tensor,
    sequence_log_ratio: torch.Tensor,
) -> torch.Tensor:
    """Compute stable QRPO residual.

    residual = R_q - beta_eff * log Z - beta_eff * log(pi / pi_ref)
    """

    expected_shape = transformed_reward.shape

    for name, tensor in {
        "beta_log_partition": beta_log_partition,
        "effective_beta": effective_beta,
        "sequence_log_ratio": sequence_log_ratio,
    }.items():
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} shape {tuple(tensor.shape)} does not match "
                f"transformed_reward shape {tuple(expected_shape)}."
            )

    if transformed_reward.ndim != 1:
        raise ValueError(
            f"transformed_reward must have shape [B], got {tuple(transformed_reward.shape)}."
        )

    return (
        transformed_reward.float()
        - beta_log_partition.float()
        - effective_beta.float() * sequence_log_ratio.float()
    )


def compute_qrpo_loss_from_fields(
    *,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    transformed_reward: torch.Tensor,
    beta_log_partition: torch.Tensor,
    effective_beta: torch.Tensor,
    reduction: str = "mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute scalar QRPO loss from tensors already attached to a batch."""

    sequence_log_ratio = compute_sequence_log_ratio(
        log_probs=log_probs,
        ref_log_probs=ref_log_probs,
        loss_mask=loss_mask,
    )

    residual = compute_qrpo_residual(
        transformed_reward=transformed_reward,
        beta_log_partition=beta_log_partition,
        effective_beta=effective_beta,
        sequence_log_ratio=sequence_log_ratio,
    )

    per_sample_loss = residual.square()

    if reduction == "mean":
        loss = per_sample_loss.mean()
    elif reduction == "sum":
        loss = per_sample_loss.sum()
    elif reduction == "none":
        loss = per_sample_loss
    else:
        raise ValueError(f"Unsupported reduction={reduction!r}.")

    metrics = {
        K.SEQUENCE_LOG_RATIO: sequence_log_ratio.detach(),
        K.QRPO_RESIDUAL: residual.detach(),
        K.QRPO_LOSS: per_sample_loss.detach(),
    }

    return loss, metrics


def add_qrpo_loss_fields(
    data: DataProto,
    *,
    config: Mapping[str, Any] | None = None,
) -> DataProto:
    """Attach QRPO diagnostic loss fields to a DataProto.

    This does not backpropagate by itself. It is useful for tests, logging, and
    trainer plumbing. The actual actor worker loss should call
    compute_qrpo_loss_from_fields(...) inside the model training step.
    """

    cfg = config or {}
    reduction = cfg.get("reduction", "none")

    if data.batch is None:
        raise ValueError("DataProto.batch is required.")

    required = [
        K.LOG_PROBS,
        K.REF_LOG_PROBS,
        K.LOSS_MASK,
        K.TRANSFORMED_REWARD,
        K.BETA_LOG_PARTITION,
        K.EFFECTIVE_BETA,
    ]

    for key in required:
        if key not in data.batch:
            raise KeyError(f"DataProto.batch is missing {key!r}.")

    loss, metrics = compute_qrpo_loss_from_fields(
        log_probs=data.batch[K.LOG_PROBS],
        ref_log_probs=data.batch[K.REF_LOG_PROBS],
        loss_mask=data.batch[K.LOSS_MASK],
        transformed_reward=data.batch[K.TRANSFORMED_REWARD],
        beta_log_partition=data.batch[K.BETA_LOG_PARTITION],
        effective_beta=data.batch[K.EFFECTIVE_BETA],
        reduction=reduction,
    )

    data.batch[K.SEQUENCE_LOG_RATIO] = metrics[K.SEQUENCE_LOG_RATIO]
    data.batch[K.QRPO_RESIDUAL] = metrics[K.QRPO_RESIDUAL]
    data.batch[K.QRPO_LOSS] = metrics[K.QRPO_LOSS]

    if reduction != "none":
        data.meta_info["qrpo_loss"] = float(loss.detach().cpu().item())

    return data
