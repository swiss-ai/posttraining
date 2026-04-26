import pytest
import torch
from verl.protocol import DataProto

from batch import keys as K
from verl_adapters.rewards import (
    attach_sequence_rewards,
    extract_and_attach_sequence_rewards,
    sequence_rewards_from_verl_output,
)


def test_sequence_rewards_from_tensor_vector() -> None:
    rewards = sequence_rewards_from_verl_output(
        torch.tensor([1.0, 2.5], dtype=torch.float32),
        batch_size=2,
    )

    assert torch.equal(rewards, torch.tensor([1.0, 2.5], dtype=torch.float32))


def test_sequence_rewards_from_verl_reward_tensor_matrix() -> None:
    reward_tensor = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 2.5, 0.0],
        ],
        dtype=torch.float32,
    )

    rewards = sequence_rewards_from_verl_output(
        reward_tensor,
        batch_size=2,
    )

    assert torch.equal(rewards, torch.tensor([1.0, 2.5], dtype=torch.float32))


def test_sequence_rewards_from_mapping_reward_tensor() -> None:
    reward_output = {
        K.REWARD_TENSOR: torch.tensor(
            [
                [0.0, 1.0],
                [2.0, 0.0],
            ],
            dtype=torch.float32,
        )
    }

    rewards = sequence_rewards_from_verl_output(
        reward_output,
        batch_size=2,
    )

    assert torch.equal(rewards, torch.tensor([1.0, 2.0], dtype=torch.float32))


def test_sequence_rewards_from_mapping_trajectory_reward_takes_precedence() -> None:
    reward_output = {
        K.TRAJECTORY_REWARD: torch.tensor([3.0, 4.0], dtype=torch.float32),
        K.REWARD_TENSOR: torch.tensor(
            [
                [100.0, 100.0],
                [100.0, 100.0],
            ],
            dtype=torch.float32,
        ),
    }

    rewards = sequence_rewards_from_verl_output(
        reward_output,
        batch_size=2,
    )

    assert torch.equal(rewards, torch.tensor([3.0, 4.0], dtype=torch.float32))


def test_sequence_rewards_from_dataproto_reward_tensor() -> None:
    reward_data = DataProto.from_dict(
        tensors={
            K.REWARD_TENSOR: torch.tensor(
                [
                    [0.0, 0.0, 1.5],
                    [0.0, 2.5, 0.0],
                ],
                dtype=torch.float32,
            )
        }
    )

    rewards = sequence_rewards_from_verl_output(
        reward_data,
        batch_size=2,
    )

    assert torch.equal(rewards, torch.tensor([1.5, 2.5], dtype=torch.float32))


def test_sequence_rewards_from_dataproto_trajectory_reward_takes_precedence() -> None:
    reward_data = DataProto.from_dict(
        tensors={
            K.TRAJECTORY_REWARD: torch.tensor([5.0, 6.0], dtype=torch.float32),
            K.REWARD_TENSOR: torch.tensor(
                [
                    [100.0],
                    [100.0],
                ],
                dtype=torch.float32,
            ),
        }
    )

    rewards = sequence_rewards_from_verl_output(
        reward_data,
        batch_size=2,
    )

    assert torch.equal(rewards, torch.tensor([5.0, 6.0], dtype=torch.float32))


def test_sequence_rewards_casts_to_float32() -> None:
    rewards = sequence_rewards_from_verl_output(
        torch.tensor([1, 2], dtype=torch.int64),
        batch_size=2,
    )

    assert rewards.dtype == torch.float32
    assert torch.equal(rewards, torch.tensor([1.0, 2.0], dtype=torch.float32))


def test_sequence_rewards_rejects_wrong_batch_size() -> None:
    with pytest.raises(ValueError, match="shape"):
        sequence_rewards_from_verl_output(
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            batch_size=2,
        )


def test_sequence_rewards_rejects_bad_rank() -> None:
    with pytest.raises(ValueError, match="shape"):
        sequence_rewards_from_verl_output(
            torch.zeros(2, 3, 4),
            batch_size=2,
        )


def test_sequence_rewards_rejects_missing_reward_key_in_mapping() -> None:
    with pytest.raises(KeyError, match=K.REWARD_TENSOR):
        sequence_rewards_from_verl_output(
            {"other": torch.tensor([1.0])},
            batch_size=1,
        )


def test_sequence_rewards_rejects_non_tensor_reward_value() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        sequence_rewards_from_verl_output(
            {K.REWARD_TENSOR: [1.0, 2.0]},
            batch_size=2,
        )


def test_attach_sequence_rewards_mutates_dataproto() -> None:
    data = DataProto.from_dict(
        tensors={
            K.INPUT_IDS: torch.ones(2, 3, dtype=torch.long),
        }
    )

    result = attach_sequence_rewards(
        data,
        torch.tensor([1.0, 2.0], dtype=torch.float32),
    )

    assert result is data
    assert K.TRAJECTORY_REWARD in data.batch
    assert torch.equal(data.batch[K.TRAJECTORY_REWARD], torch.tensor([1.0, 2.0]))


def test_attach_sequence_rewards_rejects_wrong_shape() -> None:
    data = DataProto.from_dict(
        tensors={
            K.INPUT_IDS: torch.ones(2, 3, dtype=torch.long),
        }
    )

    with pytest.raises(ValueError, match="rewards must have shape"):
        attach_sequence_rewards(
            data,
            torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        )


def test_extract_and_attach_sequence_rewards() -> None:
    data = DataProto.from_dict(
        tensors={
            K.INPUT_IDS: torch.ones(2, 3, dtype=torch.long),
        }
    )
    reward_output = {
        K.REWARD_TENSOR: torch.tensor(
            [
                [0.0, 1.0],
                [2.0, 0.0],
            ],
            dtype=torch.float32,
        )
    }

    result = extract_and_attach_sequence_rewards(
        data,
        reward_output,
    )

    assert result is data
    assert torch.equal(data.batch[K.TRAJECTORY_REWARD], torch.tensor([1.0, 2.0]))
