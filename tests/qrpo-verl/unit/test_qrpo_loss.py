import pytest
import torch
from verl.protocol import DataProto

from algorithm.qrpo_loss import (
    add_qrpo_loss_fields,
    compute_qrpo_loss_from_fields,
    compute_qrpo_residual,
    compute_sequence_log_ratio,
)
from batch import keys as K


def test_compute_sequence_log_ratio_masks_tool_and_padding_response_tokens() -> None:
    log_probs = torch.tensor(
        [
            [-1.0, -2.0, -3.0, -4.0],
            [-1.0, -2.0, -3.0, -4.0],
        ],
        dtype=torch.float32,
    )
    ref_log_probs = torch.tensor(
        [
            [-1.5, -2.5, -3.5, -4.5],
            [-0.5, -2.5, -2.0, -5.0],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ],
        dtype=torch.bool,
    )

    log_ratio = compute_sequence_log_ratio(
        log_probs=log_probs,
        ref_log_probs=ref_log_probs,
        response_mask=response_mask,
    )

    expected = torch.tensor(
        [
            (-2.0 + 2.5) + (-3.0 + 3.5),
            (-1.0 + 0.5) + (-3.0 + 2.0) + (-4.0 + 5.0),
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(log_ratio, expected)


def test_compute_sequence_log_ratio_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="ref_log_probs"):
        compute_sequence_log_ratio(
            log_probs=torch.zeros(2, 3),
            ref_log_probs=torch.zeros(2, 4),
            response_mask=torch.ones(2, 3),
        )

    with pytest.raises(ValueError, match="response_mask"):
        compute_sequence_log_ratio(
            log_probs=torch.zeros(2, 3),
            ref_log_probs=torch.zeros(2, 3),
            response_mask=torch.ones(2, 4),
        )


def test_compute_sequence_log_ratio_rejects_empty_trainable_tokens() -> None:
    with pytest.raises(ValueError, match="at least one trainable"):
        compute_sequence_log_ratio(
            log_probs=torch.zeros(2, 3),
            ref_log_probs=torch.zeros(2, 3),
            response_mask=torch.tensor(
                [
                    [1, 0, 0],
                    [0, 0, 0],
                ],
                dtype=torch.bool,
            ),
        )


def test_compute_qrpo_residual_uses_stable_form() -> None:
    transformed_reward = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    beta_log_partition = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32)
    effective_beta = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)
    sequence_log_ratio = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)

    residual = compute_qrpo_residual(
        transformed_reward=transformed_reward,
        beta_log_partition=beta_log_partition,
        effective_beta=effective_beta,
        sequence_log_ratio=sequence_log_ratio,
    )

    expected = transformed_reward - beta_log_partition - effective_beta * sequence_log_ratio
    assert torch.allclose(residual, expected)


def test_compute_qrpo_residual_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="effective_beta"):
        compute_qrpo_residual(
            transformed_reward=torch.zeros(2),
            beta_log_partition=torch.zeros(2),
            effective_beta=torch.zeros(3),
            sequence_log_ratio=torch.zeros(2),
        )


def test_compute_qrpo_loss_from_fields_mean_reduction() -> None:
    log_probs = torch.tensor(
        [
            [-1.0, -2.0],
            [-3.0, -4.0],
        ],
        dtype=torch.float32,
    )
    ref_log_probs = torch.tensor(
        [
            [-1.5, -1.0],
            [-2.0, -5.0],
        ],
        dtype=torch.float32,
    )
    response_mask = torch.tensor(
        [
            [1, 0],
            [1, 1],
        ],
        dtype=torch.bool,
    )

    transformed_reward = torch.tensor([0.5, 1.0], dtype=torch.float32)
    beta_log_partition = torch.tensor([0.1, 0.2], dtype=torch.float32)
    effective_beta = torch.tensor([2.0, 3.0], dtype=torch.float32)

    loss, metrics = compute_qrpo_loss_from_fields(
        log_probs=log_probs,
        ref_log_probs=ref_log_probs,
        response_mask=response_mask,
        transformed_reward=transformed_reward,
        beta_log_partition=beta_log_partition,
        effective_beta=effective_beta,
        reduction="mean",
    )

    sequence_log_ratio = torch.tensor(
        [
            -1.0 + 1.5,
            (-3.0 + 2.0) + (-4.0 + 5.0),
        ],
        dtype=torch.float32,
    )
    residual = transformed_reward - beta_log_partition - effective_beta * sequence_log_ratio
    expected_per_sample = residual.square()

    assert torch.allclose(loss, expected_per_sample.mean())
    assert torch.allclose(metrics[K.SEQUENCE_LOG_RATIO], sequence_log_ratio)
    assert torch.allclose(metrics[K.QRPO_RESIDUAL], residual)
    assert torch.allclose(metrics[K.QRPO_LOSS], expected_per_sample)


def test_compute_qrpo_loss_from_fields_none_reduction() -> None:
    loss, _ = compute_qrpo_loss_from_fields(
        log_probs=torch.zeros(2, 3),
        ref_log_probs=torch.zeros(2, 3),
        response_mask=torch.ones(2, 3, dtype=torch.bool),
        transformed_reward=torch.tensor([1.0, 2.0]),
        beta_log_partition=torch.tensor([0.1, 0.2]),
        effective_beta=torch.tensor([1.0, 1.0]),
        reduction="none",
    )

    assert loss.shape == (2,)


def test_compute_qrpo_loss_from_fields_rejects_unknown_reduction() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        compute_qrpo_loss_from_fields(
            log_probs=torch.zeros(1, 2),
            ref_log_probs=torch.zeros(1, 2),
            response_mask=torch.ones(1, 2, dtype=torch.bool),
            transformed_reward=torch.ones(1),
            beta_log_partition=torch.zeros(1),
            effective_beta=torch.ones(1),
            reduction="median",
        )


def test_add_qrpo_loss_fields_attaches_diagnostics() -> None:
    data = DataProto.from_dict(
        tensors={
            K.LOG_PROBS: torch.tensor([[-1.0, -2.0]], dtype=torch.float32),
            K.REF_LOG_PROBS: torch.tensor([[-1.5, -2.5]], dtype=torch.float32),
            K.RESPONSE_MASK: torch.tensor([[1, 0]], dtype=torch.bool),
            K.TRANSFORMED_REWARD: torch.tensor([0.5], dtype=torch.float32),
            K.BETA_LOG_PARTITION: torch.tensor([0.1], dtype=torch.float32),
            K.EFFECTIVE_BETA: torch.tensor([2.0], dtype=torch.float32),
        }
    )

    result = add_qrpo_loss_fields(data, config={"reduction": "mean"})

    assert result is data
    assert K.SEQUENCE_LOG_RATIO in data.batch
    assert K.QRPO_RESIDUAL in data.batch
    assert K.QRPO_LOSS in data.batch
    assert "qrpo_loss" in data.meta_info

    expected_log_ratio = torch.tensor([0.5])
    expected_residual = (
        torch.tensor([0.5])
        - torch.tensor([0.1])
        - torch.tensor([2.0]) * expected_log_ratio
    )
    expected_loss = expected_residual.square()

    assert torch.allclose(data.batch[K.SEQUENCE_LOG_RATIO], expected_log_ratio)
    assert torch.allclose(data.batch[K.QRPO_RESIDUAL], expected_residual)
    assert torch.allclose(data.batch[K.QRPO_LOSS], expected_loss)


def test_add_qrpo_loss_fields_requires_keys() -> None:
    data = DataProto.from_dict(
        tensors={
            K.LOG_PROBS: torch.zeros(1, 2),
        }
    )

    with pytest.raises(KeyError, match=K.REF_LOG_PROBS):
        add_qrpo_loss_fields(data)
