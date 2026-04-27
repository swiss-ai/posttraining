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
    prompt_len: int,
    response_len: int,
    num_ref_rewards: int = 3,
) -> DataProto:
    prompts = torch.arange(batch_size * prompt_len, dtype=torch.long).reshape(
        batch_size, prompt_len
    )
    responses = (
        torch.arange(batch_size * response_len, dtype=torch.long).reshape(
            batch_size, response_len
        )
        + 1000
    )

    prompt_attention_mask = torch.ones(batch_size, prompt_len, dtype=torch.long)
    response_attention_mask = torch.ones(batch_size, response_len, dtype=torch.long)

    # Simulate left-padded prompt and right-padded response in first row.
    if prompt_len > 2:
        prompts[0, 0] = 0
        prompt_attention_mask[0, 0] = 0

    if response_len > 2:
        responses[0, -1] = 0
        response_attention_mask[0, -1] = 0

    response_mask = response_attention_mask.bool().clone()
    response_mask[:, 0] = True
    if response_len > 2:
        response_mask[0, -1] = False

    input_ids = torch.cat([prompts, responses], dim=-1)
    attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)

    rewards = torch.arange(batch_size, dtype=torch.float32) + 1.0
    ref_rewards = torch.ones(batch_size, num_ref_rewards, dtype=torch.float32)

    return DataProto.from_dict(
        tensors={
            K.PROMPTS: prompts,
            K.RESPONSES: responses,
            K.RESPONSE_MASK: response_mask,
            K.INPUT_IDS: input_ids,
            K.ATTENTION_MASK: attention_mask,
            K.POSITION_IDS: compute_position_id_with_mask(attention_mask),
            K.TRAJECTORY_REWARD: rewards,
            K.REF_REWARDS: ref_rewards,
        },
        non_tensors={
            K.PROMPT_ID: np.asarray([f"{prefix}_p{i}" for i in range(batch_size)], dtype=object),
            K.TRAJECTORY_ID: np.asarray([f"{prefix}_t{i}" for i in range(batch_size)], dtype=object),
            K.SOURCE: np.asarray([source] * batch_size, dtype=object),
        },
        meta_info={
            "qrpo_batch_format": "verl_prompt_response",
            "source": source,
        },
    )


def test_concat_qrpo_training_dataprotos_pads_and_concats() -> None:
    offline = make_dataproto(
        source="offline",
        prefix="off",
        batch_size=2,
        prompt_len=4,
        response_len=5,
    )
    online = make_dataproto(
        source="online",
        prefix="on",
        batch_size=3,
        prompt_len=6,
        response_len=7,
    )

    mixed = concat_qrpo_training_dataprotos(
        [offline, online],
        pad_token_id=0,
    )

    assert len(mixed) == 5

    assert mixed.batch[K.PROMPTS].shape == (5, 6)
    assert mixed.batch[K.RESPONSES].shape == (5, 7)
    assert mixed.batch[K.RESPONSE_MASK].shape == (5, 7)

    assert mixed.batch[K.INPUT_IDS].shape == (5, 13)
    assert mixed.batch[K.ATTENTION_MASK].shape == (5, 13)
    assert mixed.batch[K.POSITION_IDS].shape == (5, 13)

    assert mixed.batch[K.TRAJECTORY_REWARD].shape == (5,)
    assert mixed.batch[K.REF_REWARDS].shape == (5, 3)

    assert torch.equal(
        mixed.batch[K.INPUT_IDS],
        torch.cat([mixed.batch[K.PROMPTS], mixed.batch[K.RESPONSES]], dim=-1),
    )

    assert mixed.non_tensor_batch[K.SOURCE].tolist() == [
        "offline",
        "offline",
        "online",
        "online",
        "online",
    ]

    assert mixed.meta_info["qrpo_batch_format"] == "verl_prompt_response"
    assert mixed.meta_info["source"] == "mixed"


def test_concat_qrpo_training_dataprotos_left_pads_prompts_and_right_pads_responses() -> None:
    offline = make_dataproto(
        source="offline",
        prefix="off",
        batch_size=1,
        prompt_len=4,
        response_len=4,
    )
    online = make_dataproto(
        source="online",
        prefix="on",
        batch_size=1,
        prompt_len=6,
        response_len=6,
    )

    mixed = concat_qrpo_training_dataprotos(
        [offline, online],
        pad_token_id=99,
    )

    # Offline prompt was length 4, padded to length 6 on the left.
    assert mixed.batch[K.PROMPTS][0, :2].tolist() == [99, 99]

    # Offline response was length 4, padded to length 6 on the right.
    assert mixed.batch[K.RESPONSES][0, -2:].tolist() == [99, 99]

    # Response mask padding is false.
    assert mixed.batch[K.RESPONSE_MASK][0, -2:].tolist() == [False, False]

    # Full attention mask should reflect left prompt padding and right response padding.
    assert mixed.batch[K.ATTENTION_MASK][0, :2].tolist() == [0, 0]
    assert mixed.batch[K.ATTENTION_MASK][0, -2:].tolist() == [0, 0]


def test_concat_qrpo_training_dataprotos_preserves_order() -> None:
    offline = make_dataproto(
        source="offline",
        prefix="off",
        batch_size=2,
        prompt_len=4,
        response_len=4,
    )
    online = make_dataproto(
        source="online",
        prefix="on",
        batch_size=2,
        prompt_len=4,
        response_len=4,
    )

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
    offline = make_dataproto(source="offline", prefix="off", batch_size=1, prompt_len=4, response_len=4)
    online = make_dataproto(source="online", prefix="on", batch_size=1, prompt_len=4, response_len=4)

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
    offline = make_dataproto(source="offline", prefix="off", batch_size=1, prompt_len=4, response_len=4)

    mixed = concat_qrpo_training_dataprotos(
        [offline],
        pad_token_id=0,
        meta_info={"global_steps": 3},
    )

    assert mixed is offline
    assert mixed.meta_info["global_steps"] == 3


def test_concat_qrpo_training_dataprotos_rejects_missing_required_tensor_key() -> None:
    offline = make_dataproto(source="offline", prefix="off", batch_size=1, prompt_len=4, response_len=4)
    online = make_dataproto(source="online", prefix="on", batch_size=1, prompt_len=4, response_len=4)

    del online.batch[K.RESPONSE_MASK]

    with pytest.raises(KeyError, match=K.RESPONSE_MASK):
        concat_qrpo_training_dataprotos([offline, online], pad_token_id=0)


def test_concat_qrpo_training_dataprotos_rejects_mismatched_non_tensor_keys() -> None:
    offline = make_dataproto(source="offline", prefix="off", batch_size=1, prompt_len=4, response_len=4)
    online = make_dataproto(source="online", prefix="on", batch_size=1, prompt_len=4, response_len=4)

    online.non_tensor_batch["extra"] = np.asarray(["x"], dtype=object)

    with pytest.raises(ValueError, match="non-tensor keys"):
        concat_qrpo_training_dataprotos([offline, online], pad_token_id=0)


def test_concat_qrpo_training_dataprotos_rejects_bad_response_mask_shape() -> None:
    offline = make_dataproto(source="offline", prefix="off", batch_size=1, prompt_len=4, response_len=4)
    online = make_dataproto(source="online", prefix="on", batch_size=1, prompt_len=4, response_len=4)

    online.batch[K.RESPONSE_MASK] = torch.ones(1, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="response_mask shape"):
        concat_qrpo_training_dataprotos([offline, online], pad_token_id=0)
