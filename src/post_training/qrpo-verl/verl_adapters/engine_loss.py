from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from algorithm.qrpo_loss import compute_qrpo_loss_from_fields
from batch import keys as K

from verl.utils.metric import AggregationType, Metric
from verl.workers.utils.padding import no_padding_2_padding


def qrpo_engine_loss(
    config: Any,
    model_output: TensorDict,
    data: TensorDict,
    dp_group=None,
):
    """QRPO actor loss for VERL's new model-engine TrainingWorker.

    Signature matches VERL engine losses:

        loss_fn(config, model_output, data, dp_group=None)

    Inputs:
      model_output["log_probs"]:
        current actor logprobs produced by the trainable actor forward pass.
        May be no-padding/nested depending on backend; we convert it with
        VERL's no_padding_2_padding(...).

      data["ref_log_prob"]:
        reference-policy logprobs from compute_ref_log_prob(...), response-aligned.

      data["response_mask"]:
        response-aligned trainable-token mask.

      data["transformed_reward"], data["effective_beta"], data["beta_log_partition"]:
        QRPO fields precomputed on the driver side.

    Stable residual:

        residual = R_q - beta_eff * log Z - beta_eff * log(pi_theta / pi_ref)
    """

    log_probs = no_padding_2_padding(model_output["log_probs"], data)

    loss, qrpo_metrics = compute_qrpo_loss_from_fields(
        log_probs=log_probs,
        ref_log_probs=data[K.REF_LOG_PROBS],
        response_mask=data[K.RESPONSE_MASK],
        transformed_reward=data[K.TRANSFORMED_REWARD],
        beta_log_partition=data[K.BETA_LOG_PARTITION],
        effective_beta=data[K.EFFECTIVE_BETA],
        reduction="mean",
    )

    metrics = {
        "actor/qrpo_loss": Metric(
            value=loss.detach(),
            aggregation=AggregationType.MEAN,
        ),
        "actor/sequence_log_ratio_mean": Metric(
            value=qrpo_metrics[K.SEQUENCE_LOG_RATIO].mean(),
            aggregation=AggregationType.MEAN,
        ),
        "actor/sequence_log_ratio_min": Metric(
            value=qrpo_metrics[K.SEQUENCE_LOG_RATIO].min(),
            aggregation=AggregationType.MEAN,
        ),
        "actor/sequence_log_ratio_max": Metric(
            value=qrpo_metrics[K.SEQUENCE_LOG_RATIO].max(),
            aggregation=AggregationType.MEAN,
        ),
        "actor/qrpo_residual_mean": Metric(
            value=qrpo_metrics[K.QRPO_RESIDUAL].mean(),
            aggregation=AggregationType.MEAN,
        ),
        "actor/qrpo_residual_abs_mean": Metric(
            value=qrpo_metrics[K.QRPO_RESIDUAL].abs().mean(),
            aggregation=AggregationType.MEAN,
        ),
        "actor/qrpo_per_sample_loss_mean": Metric(
            value=qrpo_metrics[K.QRPO_LOSS].mean(),
            aggregation=AggregationType.MEAN,
        ),
    }

    return loss, metrics