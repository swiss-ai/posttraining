import numpy as np
import pytest
import torch
from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask

from batch import keys as K
from verl_adapters.train_batch import concat_qrpo_training_dataprotos


def make_dataproto(
    *,
    source: str,
    prefix: str,
    batch_size: int,
    seq_len: int,
    num_ref_rewards: int = 3,
) -> DataProto:
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Last token in the first row is padding if seq_len > 3, just to test masks.
    if seq_len > 3:
        attention_mask[0, -1] = 0
        input_ids[0, -1] = 0

    loss_mask = attention_mask.bool().clone()
    loss_mask[:, :2] = False

    rewards = torch.arange(batch_size, dtype=torch.float32) + 1.0
    ref_rewards = torch.ones(batch_size, num_ref_rewards, dtype=torch.float32)

    return DataProto.from_dict(
        tensors={
            K.INPUT_IDS: input_ids,
            K.ATTENTION_MASK: attention_mask,
            K.POSITION_IDS: compute_position_id_with_mask(attention_mask),
            K.LOSS_MASK: loss_mask,
            K.TRAJECTORY_REWARD: rewards,
            K.REF_REWARDS: ref_rewards,
        },
        non_tensors={
            K.PROMPT_ID: np.asarray([f"{prefix}_p{i}" for i in range(batch_size)], dtype=object),
            K.TRAJECTORY_ID: np.asarray([f"{prefix}_t{i}" for i in range(batch_size)], dtype=object),
            K.SOURCE: np.asarray([source] * batch_size, dtype=object),
        },
        meta_info={
            "qrpo_batch_format": "full_sequence_with_loss_mask",
            "source": source,
        },
    )


def test_concat_qrpo_training_dataprotos_pads_and_concats() -> None:
    offline = make_dataproto(source="offline", prefix="off", batch_size=2, seq_len=5)
    online = make_dataproto(source="online", prefix="on", batch_size=3, seq_len=7)

    mixed = concat_qrpo_training_dataprotos(
        [offline, online],
        pad_token_id=0,
    )

    assert len(mixed) == 5

    assert mixed.batch[K.INPUT_IDS].shape == (5, 7)
    assert mixed.batch[K.ATTENTION_MASK].shape == (5, 7)
    assert mixed.batch[K.POSITION_IDS].shape == (5, 7)
    assert mixed.batch[K.LOSS_MASK].shape == (5, 7)

    assert mixed.batch[K.TRAJECTORY_REWARD].shape == (5,)
    assert mixed.batch[K.REF_REWARDS].shape == (5, 3)

    assert mixed.non_tensor_batch[K.SOURCE].tolist() == [
        "offline",
        "offline",
        "online",
        "online",
        "online",
    ]

    assert mixed.meta_info["qrpo_batch_format"] == "full_sequence_with_loss_mask"
    assert mixed.meta_info["source"] == "mixed"


def test_concat_qrpo_training_dataprotos_right_pads_shorter_batch() -> None:
    offline = make_dataproto(source="offline", prefix="off", batch_size=1, seq_len=4)
    online = make_dataproto(source="online", prefix="on", batch_size=1, seq_len=6)

    mixed = concat_qrpo_training_dataprotos(
        [offline, online],
        pad_token_id=99,
    )

    # First row came from seq_len=4 and should be padded to 6.
    assert mixed.batch[K.INPUT_IDS][0, -2:].tolist() == [99, 99]
    assert mixed.batch[K.ATTENTION_MASK][0, -2:].tolist() == [0, 0]
    assert mixed.batch[K.POSITION_IDS][0, -2:].tolist() == [0, 0]
    assert mixed.batch[K.LOSS_MASK][0, -2:].tolist() == [False, False]


def test_concat_qrpo_training_dataprotos_preserves_order() -> None:
    offline = make_dataproto(source="offline", prefix="off", batch_size=2, seq_len=4)
    online = make_dataproto(source="online", prefix="on", batch_size=2, seq_len=4)

    mixed = concat_qrpo_training_dataprotos(
        [offline, online],
        pad_token_id=0,
    )

    assert mixed.non_tensor_batch[K.PROMPT_ID].tolist() == [
        "off_p0",
        "off_p1",
        "on_p0",
        "on_p1",
    ]
    assert mixed.non_tensor_batch[K.TRAJECTORY_ID].tolist() == [
        "off_t0",
        "off_t1",
        "on_t0",
        "on_t1",
    ]


def test_concat_qrpo_training_dataprotos_merges_custom_meta_info() -> None:
    offline = make_dataproto(source="offline", prefix="off", batch_size=1, seq_len=4)
    online = make_dataproto(source="online", prefix="on", batch_size=1, seq_len=4)

    mixed = concat_qrpo_training_dataprotos(
        [offline, online],
        pad_token_id=0,
        meta_info={"global_steps": 12},
    )

    assert mixed.meta_info["global_steps"] == 12


def test_concat_qrpo_training_dataprotos_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        concat_qrpo_training_dataprotos([], pad_token_id=0)


def test_concat_qrpo_training_dataprotos_single_input_returns_same_batch_with_meta_update() -> None:
    offline = make_dataproto(source="offline", prefix="off", batch_size=1, seq_len=4)

    mixed = concat_qrpo_training_dataprotos(
        [offline],
        pad_token_id=0,
        meta_info={"global_steps": 3},
    )

    assert mixed is offline
    assert mixed.meta_info["global_steps"] == 3


def test_concat_qrpo_training_dataprotos_rejects_missing_required_tensor_key() -> None:
    offline = make_dataproto(source="offline", prefix="off", batch_size=1, seq_len=4)
    online = make_dataproto(source="online", prefix="on", batch_size=1, seq_len=4)

    del online.batch[K.LOSS_MASK]

    with pytest.raises(KeyError, match=K.LOSS_MASK):
        concat_qrpo_training_dataprotos([offline, online], pad_token_id=0)


def test_concat_qrpo_training_dataprotos_rejects_mismatched_non_tensor_keys() -> None:
    offline = make_dataproto(source="offline", prefix="off", batch_size=1, seq_len=4)
    online = make_dataproto(source="online", prefix="on", batch_size=1, seq_len=4)

    online.non_tensor_batch["extra"] = np.asarray(["x"], dtype=object)

    with pytest.raises(ValueError, match="non-tensor keys"):
        concat_qrpo_training_dataprotos([offline, online], pad_token_id=0)
