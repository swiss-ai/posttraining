import numpy as np
import pytest
import torch
from verl.protocol import DataProto

from batch import keys as K
from verl_adapters.rollout import online_rollout_output_to_train_dataproto


def make_ref_rewards(values):
    arr = np.empty(len(values), dtype=object)
    arr[:] = [tuple(v) for v in values]
    return arr


def make_rollout_output() -> DataProto:
    prompts = torch.tensor(
        [
            [0, 101, 102],
            [201, 202, 203],
        ],
        dtype=torch.long,
    )

    responses = torch.tensor(
        [
            [301, 302, 303, 0],
            [401, 402, 403, 404],
        ],
        dtype=torch.long,
    )

    # First row: two model tokens, one tool/env token, then padding.
    # Second row: model, tool/env, model, model.
    response_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 1],
        ],
        dtype=torch.long,
    )

    input_ids = torch.cat([prompts, responses], dim=-1)

    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    position_ids = torch.tensor(
        [
            [0, 0, 1, 2, 3, 4, 4],
            [0, 1, 2, 3, 4, 5, 6],
        ],
        dtype=torch.long,
    )

    return DataProto.from_dict(
        tensors={
            K.PROMPTS: prompts,
            K.RESPONSES: responses,
            K.RESPONSE_MASK: response_mask,
            K.INPUT_IDS: input_ids,
            K.ATTENTION_MASK: attention_mask,
            K.POSITION_IDS: position_ids,
        },
        non_tensors={
            K.PROMPT_ID: np.asarray(["p0", "p1"], dtype=object),
            K.TRAJECTORY_ID: np.asarray(
                ["p0::online::actor_10::0", "p1::online::actor_10::0"],
                dtype=object,
            ),
            K.REF_REWARDS: make_ref_rewards(
                [
                    (0.1, 0.2, 0.3),
                    (0.4, 0.5, 0.6),
                ]
            ),
        },
    )


def test_online_rollout_output_to_train_dataproto_builds_qrpo_training_batch() -> None:
    rollout = make_rollout_output()
    rewards = torch.tensor([1.0, 2.0], dtype=torch.float32)

    data = online_rollout_output_to_train_dataproto(
        rollout_output=rollout,
        rewards=rewards,
    )

    assert len(data) == 2

    for key in (
        K.PROMPTS,
        K.RESPONSES,
        K.RESPONSE_MASK,
        K.INPUT_IDS,
        K.ATTENTION_MASK,
        K.POSITION_IDS,
        K.TRAJECTORY_REWARD,
        K.REF_REWARDS,
    ):
        assert key in data.batch

    assert torch.equal(data.batch[K.PROMPTS], rollout.batch[K.PROMPTS])
    assert torch.equal(data.batch[K.RESPONSES], rollout.batch[K.RESPONSES])
    assert torch.equal(data.batch[K.RESPONSE_MASK], rollout.batch[K.RESPONSE_MASK].bool())
    assert torch.equal(data.batch[K.INPUT_IDS], rollout.batch[K.INPUT_IDS])
    assert torch.equal(data.batch[K.ATTENTION_MASK], rollout.batch[K.ATTENTION_MASK])
    assert torch.equal(data.batch[K.POSITION_IDS], rollout.batch[K.POSITION_IDS])

    assert data.batch[K.TRAJECTORY_REWARD].tolist() == [1.0, 2.0]
    assert data.batch[K.REF_REWARDS].shape == (2, 3)

    assert data.non_tensor_batch[K.PROMPT_ID].tolist() == ["p0", "p1"]
    assert data.non_tensor_batch[K.TRAJECTORY_ID].tolist() == [
        "p0::online::actor_10::0",
        "p1::online::actor_10::0",
    ]
    assert data.non_tensor_batch[K.SOURCE].tolist() == ["online", "online"]

    assert data.meta_info["qrpo_batch_format"] == "verl_prompt_response"
    assert data.meta_info["source"] == "online"


def test_online_rollout_output_to_train_dataproto_preserves_response_mask() -> None:
    rollout = make_rollout_output()
    rewards = torch.tensor([1.0, 2.0], dtype=torch.float32)

    data = online_rollout_output_to_train_dataproto(
        rollout_output=rollout,
        rewards=rewards,
    )

    expected_response_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 1],
        ],
        dtype=torch.bool,
    )

    assert torch.equal(data.batch[K.RESPONSE_MASK], expected_response_mask)


def test_online_rollout_output_to_train_dataproto_keeps_tool_tokens_non_trainable() -> None:
    rollout = make_rollout_output()
    rewards = torch.tensor([1.0, 2.0], dtype=torch.float32)

    data = online_rollout_output_to_train_dataproto(
        rollout_output=rollout,
        rewards=rewards,
    )

    # Row 0 response token 303 is a tool/env token.
    assert rollout.batch[K.RESPONSES][0, 2].item() == 303
    assert data.batch[K.RESPONSE_MASK][0, 2].item() is False

    # Row 1 response token 402 is a tool/env token.
    assert rollout.batch[K.RESPONSES][1, 1].item() == 402
    assert data.batch[K.RESPONSE_MASK][1, 1].item() is False


def test_online_rollout_output_to_train_dataproto_merges_meta_info() -> None:
    rollout = make_rollout_output()
    rewards = torch.tensor([1.0, 2.0], dtype=torch.float32)

    data = online_rollout_output_to_train_dataproto(
        rollout_output=rollout,
        rewards=rewards,
        meta_info={"global_steps": 9},
    )

    assert data.meta_info["global_steps"] == 9
    assert data.meta_info["source"] == "online"


def test_online_rollout_output_to_train_dataproto_rejects_wrong_reward_shape() -> None:
    rollout = make_rollout_output()

    with pytest.raises(ValueError, match="rewards must have shape"):
        online_rollout_output_to_train_dataproto(
            rollout_output=rollout,
            rewards=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        )


def test_online_rollout_output_to_train_dataproto_requires_rollout_batch() -> None:
    rollout = DataProto(
        non_tensor_batch={
            K.PROMPT_ID: np.asarray(["p0"], dtype=object),
            K.TRAJECTORY_ID: np.asarray(["t0"], dtype=object),
            K.REF_REWARDS: make_ref_rewards([(0.1, 0.2)]),
        }
    )

    with pytest.raises(ValueError, match="batch is required"):
        online_rollout_output_to_train_dataproto(
            rollout_output=rollout,
            rewards=torch.tensor([1.0], dtype=torch.float32),
        )


def test_online_rollout_output_to_train_dataproto_requires_ref_rewards_metadata() -> None:
    rollout = make_rollout_output()
    rollout.non_tensor_batch.pop(K.REF_REWARDS)

    with pytest.raises(KeyError, match=K.REF_REWARDS):
        online_rollout_output_to_train_dataproto(
            rollout_output=rollout,
            rewards=torch.tensor([1.0, 2.0], dtype=torch.float32),
        )


def test_online_rollout_output_to_train_dataproto_rejects_bad_response_mask_shape() -> None:
    rollout = make_rollout_output()
    rollout.batch[K.RESPONSE_MASK] = torch.ones(2, 3, dtype=torch.long)

    with pytest.raises(ValueError, match="response_mask shape"):
        online_rollout_output_to_train_dataproto(
            rollout_output=rollout,
            rewards=torch.tensor([1.0, 2.0], dtype=torch.float32),
        )
