import math

import pytest
import torch
from verl.protocol import DataProto

from algorithm.qrpo_fields import (
    add_qrpo_fields,
    compute_ref_quantiles,
    compute_response_lengths,
    identity_quantile_beta_log_partition,
    identity_quantile_log_partition,
)
from batch import keys as K


def test_compute_ref_quantiles_uses_inclusive_empirical_cdf() -> None:
    trajectory_rewards = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float32)
    ref_rewards = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=torch.float32,
    )

    quantiles = compute_ref_quantiles(
        trajectory_rewards=trajectory_rewards,
        ref_rewards=ref_rewards,
    )

    assert torch.equal(
        quantiles,
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32),
    )


def test_compute_ref_quantiles_allows_exact_zero_and_one() -> None:
    trajectory_rewards = torch.tensor([-1.0, 10.0], dtype=torch.float32)
    ref_rewards = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=torch.float32,
    )

    quantiles = compute_ref_quantiles(
        trajectory_rewards=trajectory_rewards,
        ref_rewards=ref_rewards,
    )

    assert torch.equal(
        quantiles,
        torch.tensor([0.0, 1.0], dtype=torch.float32),
    )


def test_compute_ref_quantiles_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="trajectory_rewards"):
        compute_ref_quantiles(
            trajectory_rewards=torch.zeros(2, 1),
            ref_rewards=torch.zeros(2, 3),
        )

    with pytest.raises(ValueError, match="ref_rewards"):
        compute_ref_quantiles(
            trajectory_rewards=torch.zeros(2),
            ref_rewards=torch.zeros(2),
        )

    with pytest.raises(ValueError, match="same batch size"):
        compute_ref_quantiles(
            trajectory_rewards=torch.zeros(2),
            ref_rewards=torch.zeros(3, 4),
        )


def test_compute_response_lengths_counts_response_mask_tokens() -> None:
    response_mask = torch.tensor(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ],
        dtype=torch.bool,
    )

    lengths = compute_response_lengths(response_mask)

    assert torch.equal(lengths, torch.tensor([2.0, 3.0], dtype=torch.float32))


def test_compute_response_lengths_rejects_zero_trainable_tokens() -> None:
    response_mask = torch.tensor(
        [
            [0, 0],
            [1, 0],
        ],
        dtype=torch.bool,
    )

    with pytest.raises(ValueError, match="at least one trainable"):
        compute_response_lengths(response_mask)


def test_compute_response_lengths_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="response_mask"):
        compute_response_lengths(torch.ones(2))


def test_identity_quantile_log_partition_matches_closed_form_scalar() -> None:
    beta = 2.0

    log_z = identity_quantile_log_partition(beta)

    expected = math.log(beta * (math.exp(1.0 / beta) - 1.0))
    assert torch.allclose(log_z, torch.tensor(expected, dtype=torch.float32))


def test_identity_quantile_log_partition_matches_closed_form_vector() -> None:
    beta = torch.tensor([2.0, 1.0], dtype=torch.float32)

    log_z = identity_quantile_log_partition(beta)

    expected = torch.tensor(
        [
            math.log(2.0 * (math.exp(0.5) - 1.0)),
            math.log(math.exp(1.0) - 1.0),
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(log_z, expected)


def test_identity_quantile_beta_log_partition_matches_beta_times_log_partition() -> None:
    beta = torch.tensor([6.0, 3.0, 2.0, 0.25], dtype=torch.float32)

    beta_log_z = identity_quantile_beta_log_partition(beta)
    log_z = identity_quantile_log_partition(beta)

    assert torch.allclose(beta_log_z, beta * log_z, rtol=1e-5, atol=1e-6)


def test_identity_quantile_beta_log_partition_matches_closed_form_scalar() -> None:
    beta = 2.0

    beta_log_z = identity_quantile_beta_log_partition(beta)

    expected_log_z = math.log(beta * (math.exp(1.0 / beta) - 1.0))
    expected = beta * expected_log_z

    assert torch.allclose(beta_log_z, torch.tensor(expected, dtype=torch.float32))


def test_identity_quantile_partitions_handle_small_beta_stably() -> None:
    beta = torch.tensor([1.0, 0.1, 0.01], dtype=torch.float32)

    log_z = identity_quantile_log_partition(beta)
    beta_log_z = identity_quantile_beta_log_partition(beta)

    assert torch.isfinite(log_z).all()
    assert torch.isfinite(beta_log_z).all()

    expected_small = 1.0 + 0.01 * math.log(0.01)
    assert torch.allclose(
        beta_log_z[-1],
        torch.tensor(expected_small, dtype=torch.float32),
        rtol=1e-4,
        atol=1e-4,
    )


def test_identity_quantile_partitions_reject_non_positive_beta() -> None:
    with pytest.raises(ValueError, match="positive"):
        identity_quantile_log_partition(0.0)

    with pytest.raises(ValueError, match="positive"):
        identity_quantile_log_partition(torch.tensor([1.0, -1.0]))

    with pytest.raises(ValueError, match="positive"):
        identity_quantile_beta_log_partition(0.0)

    with pytest.raises(ValueError, match="positive"):
        identity_quantile_beta_log_partition(torch.tensor([1.0, -1.0]))


def make_qrpo_data() -> DataProto:
    return DataProto.from_dict(
        tensors={
            K.TRAJECTORY_REWARD: torch.tensor([0.0, 2.0, 4.0], dtype=torch.float32),
            K.REF_REWARDS: torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [1.0, 2.0, 3.0, 4.0],
                    [1.0, 2.0, 3.0, 4.0],
                ],
                dtype=torch.float32,
            ),
            K.RESPONSE_MASK: torch.tensor(
                [
                    [1, 0, 0],
                    [1, 1, 0],
                    [1, 1, 1],
                ],
                dtype=torch.bool,
            ),
        }
    )


def test_add_qrpo_fields_defaults_to_length_normalized_identity_transform() -> None:
    data = make_qrpo_data()

    result = add_qrpo_fields(
        data,
        config={
            "beta": 6.0,
            "transform": "identity",
        },
    )

    assert result is data

    expected_quantile = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    expected_lengths = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    expected_effective_beta = torch.tensor([6.0, 3.0, 2.0], dtype=torch.float32)

    expected_log_z = identity_quantile_log_partition(expected_effective_beta)
    expected_beta_log_z = identity_quantile_beta_log_partition(expected_effective_beta)

    assert torch.equal(data.batch[K.REF_QUANTILE], expected_quantile)
    assert torch.equal(data.batch[K.TRANSFORMED_REWARD], expected_quantile)
    assert torch.equal(data.batch[K.TRAJECTORY_LENGTH], expected_lengths)
    assert torch.equal(data.batch[K.EFFECTIVE_BETA], expected_effective_beta)

    assert torch.allclose(data.batch[K.LOG_PARTITION], expected_log_z)
    assert torch.allclose(data.batch[K.BETA_LOG_PARTITION], expected_beta_log_z)


def test_add_qrpo_fields_supports_non_length_normalized_identity_transform() -> None:
    data = make_qrpo_data()

    add_qrpo_fields(
        data,
        config={
            "beta": 2.0,
            "transform": "identity",
            "length_normalization": False,
        },
    )

    expected_quantile = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    expected_lengths = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    expected_effective_beta = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)

    expected_log_z = identity_quantile_log_partition(expected_effective_beta)
    expected_beta_log_z = identity_quantile_beta_log_partition(expected_effective_beta)

    assert torch.equal(data.batch[K.REF_QUANTILE], expected_quantile)
    assert torch.equal(data.batch[K.TRANSFORMED_REWARD], expected_quantile)
    assert torch.equal(data.batch[K.TRAJECTORY_LENGTH], expected_lengths)
    assert torch.equal(data.batch[K.EFFECTIVE_BETA], expected_effective_beta)

    assert torch.allclose(data.batch[K.LOG_PARTITION], expected_log_z)
    assert torch.allclose(data.batch[K.BETA_LOG_PARTITION], expected_beta_log_z)


def test_add_qrpo_fields_boolean_length_normalization_config() -> None:
    data = make_qrpo_data()

    add_qrpo_fields(
        data,
        config={
            "beta": 6.0,
            "length_normalization": True,
        },
    )

    assert torch.equal(
        data.batch[K.EFFECTIVE_BETA],
        torch.tensor([6.0, 3.0, 2.0], dtype=torch.float32),
    )

    data = make_qrpo_data()

    add_qrpo_fields(
        data,
        config={
            "beta": 6.0,
            "length_normalization": False,
        },
    )

    assert torch.equal(
        data.batch[K.EFFECTIVE_BETA],
        torch.tensor([6.0, 6.0, 6.0], dtype=torch.float32),
    )


def test_add_qrpo_fields_stores_terms_for_stable_residual() -> None:
    data = make_qrpo_data()

    add_qrpo_fields(
        data,
        config={
            "beta": 6.0,
            "length_normalization": True,
        },
    )

    log_ratio = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)

    residual = (
        data.batch[K.TRANSFORMED_REWARD]
        - data.batch[K.BETA_LOG_PARTITION]
        - data.batch[K.EFFECTIVE_BETA] * log_ratio
    )

    expected_quantile = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    expected_beta = torch.tensor([6.0, 3.0, 2.0], dtype=torch.float32)
    expected_beta_log_z = identity_quantile_beta_log_partition(expected_beta)
    expected_residual = expected_quantile - expected_beta_log_z - expected_beta * log_ratio

    assert torch.allclose(residual, expected_residual)


def test_add_qrpo_fields_requires_response_mask_for_length_normalization() -> None:
    data = DataProto.from_dict(
        tensors={
            K.TRAJECTORY_REWARD: torch.tensor([1.0], dtype=torch.float32),
            K.REF_REWARDS: torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        }
    )

    with pytest.raises(KeyError, match=K.RESPONSE_MASK):
        add_qrpo_fields(
            data,
            config={
                "beta": 1.0,
                "length_normalization": True,
            },
        )


def test_add_qrpo_fields_does_not_require_response_mask_without_length_normalization() -> None:
    data = DataProto.from_dict(
        tensors={
            K.TRAJECTORY_REWARD: torch.tensor([1.0], dtype=torch.float32),
            K.REF_REWARDS: torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        }
    )

    add_qrpo_fields(
        data,
        config={
            "beta": 1.0,
            "length_normalization": False,
        },
    )

    assert torch.equal(data.batch[K.TRAJECTORY_LENGTH], torch.tensor([1.0]))
    assert torch.equal(data.batch[K.EFFECTIVE_BETA], torch.tensor([1.0]))


def test_add_qrpo_fields_requires_beta() -> None:
    data = make_qrpo_data()

    with pytest.raises(ValueError, match="beta"):
        add_qrpo_fields(data, config={})


def test_add_qrpo_fields_rejects_non_positive_beta() -> None:
    data = make_qrpo_data()

    with pytest.raises(ValueError, match="positive"):
        add_qrpo_fields(data, config={"beta": 0.0})


def test_add_qrpo_fields_rejects_unknown_transform() -> None:
    data = make_qrpo_data()

    with pytest.raises(NotImplementedError, match="Unsupported"):
        add_qrpo_fields(
            data,
            config={
                "beta": 1.0,
                "transform": "log",
            },
        )


def test_add_qrpo_fields_rejects_non_boolean_length_normalization() -> None:
    data = make_qrpo_data()

    with pytest.raises(ValueError, match="length_normalization"):
        add_qrpo_fields(
            data,
            config={
                "beta": 1.0,
                "length_normalization": "trainable_tokens",
            },
        )


def test_add_qrpo_fields_requires_reward_keys() -> None:
    data = DataProto.from_dict(
        tensors={
            K.TRAJECTORY_REWARD: torch.tensor([1.0], dtype=torch.float32),
        }
    )

    with pytest.raises(KeyError, match=K.REF_REWARDS):
        add_qrpo_fields(
            data,
            config={
                "beta": 1.0,
                "length_normalization": False,
            },
        )
