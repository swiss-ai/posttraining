from __future__ import annotations

import numpy as np
import pytest
import torch
from verl.protocol import DataProto

from batch import keys as K
from data.schemas import PromptRecord
from ref_rewards.generation import (
    attach_ref_rollout_metadata,
    extract_ref_rollout_metadata,
    pad_ref_rollout_input_for_agent_workers,
    ref_reward_rollout_input_from_prompt_records,
    ref_rollout_output_to_store_rows,
    truncate_ref_rollout_output,
)


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in token_ids if token_id != 0)


def _prompt(prompt_id: str) -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        prompt_messages=({"role": "user", "content": prompt_id},),
        ref_rewards=(0.0,),
        offline_trajectories=(),
        offline_rewards=(),
    )


def _rollout_output_without_metadata() -> DataProto:
    prompts = torch.tensor(
        [
            [101, 102],
            [101, 102],
            [201, 202],
            [201, 202],
        ],
        dtype=torch.long,
    )
    responses = torch.tensor(
        [
            [11, 12, 0],
            [13, 0, 0],
            [21, 22, 0],
            [23, 0, 0],
        ],
        dtype=torch.long,
    )
    input_ids = torch.cat([prompts, responses], dim=-1)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    return DataProto.from_dict(
        tensors={
            K.PROMPTS: prompts,
            K.RESPONSES: responses,
            K.INPUT_IDS: input_ids,
            K.ATTENTION_MASK: attention_mask,
            "rm_scores": torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [4.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        },
        non_tensors={},
        meta_info={},
    )


def test_ref_reward_rollout_input_expands_prompts():
    batch = ref_reward_rollout_input_from_prompt_records(
        prompt_records=[_prompt("p0"), _prompt("p1")],
        ref_version="ref_v1",
        num_ref_completions=2,
        config={"data_source": "activeultrafeedback", "agent_name": None},
        dataset_indices=[10, 11],
        meta_info={"global_steps": 3},
    )

    assert batch.meta_info[K.REF_VERSION] == "ref_v1"
    assert batch.meta_info["num_ref_completions"] == 2
    assert batch.meta_info["global_steps"] == 3
    assert batch.non_tensor_batch[K.PROMPT_ID].tolist() == ["p0", "p0", "p1", "p1"]
    assert batch.non_tensor_batch[K.DATASET_INDEX].tolist() == [10, 10, 11, 11]
    assert batch.non_tensor_batch[K.REF_COMPLETION_INDEX].tolist() == [0, 1, 0, 1]
    assert batch.non_tensor_batch[K.DATA_SOURCE].tolist() == [
        "activeultrafeedback",
        "activeultrafeedback",
        "activeultrafeedback",
        "activeultrafeedback",
    ]
    assert K.REWARD_MODEL in batch.non_tensor_batch
    assert K.EXTRA_INFO in batch.non_tensor_batch


def test_pad_ref_rollout_input_for_agent_workers_duplicates_trailing_padding():
    batch = ref_reward_rollout_input_from_prompt_records(
        prompt_records=[_prompt("p0"), _prompt("p1"), _prompt("p2")],
        ref_version="ref_v1",
        num_ref_completions=1,
        config={"data_source": "activeultrafeedback"},
        dataset_indices=[10, 11, 12],
    )

    padded, original_size = pad_ref_rollout_input_for_agent_workers(
        batch,
        num_workers=4,
    )

    assert original_size == 3
    assert len(padded) == 4
    assert padded.non_tensor_batch[K.PROMPT_ID].tolist() == [
        "p0",
        "p1",
        "p2",
        "p0",
    ]
    assert padded.non_tensor_batch[K.DATASET_INDEX].tolist() == [10, 11, 12, 10]
    assert padded.meta_info == batch.meta_info


def test_truncate_ref_rollout_output_drops_padded_rows():
    rollout_output = _rollout_output_without_metadata()
    rollout_output.non_tensor_batch[K.PROMPT_ID] = np.asarray(
        ["p0", "p0", "p1", "pad"],
        dtype=object,
    )

    truncated = truncate_ref_rollout_output(rollout_output, size=3)

    assert len(truncated) == 3
    assert truncated.batch[K.RESPONSES].shape == (3, 3)
    assert truncated.non_tensor_batch[K.PROMPT_ID].tolist() == ["p0", "p0", "p1"]


def test_ref_rollout_output_to_store_rows_groups_rewards_and_completions():
    request_batch = ref_reward_rollout_input_from_prompt_records(
        prompt_records=[_prompt("p0"), _prompt("p1")],
        ref_version="ref_v1",
        num_ref_completions=2,
        config={"data_source": "activeultrafeedback"},
        dataset_indices=[10, 11],
    )
    rollout_output = _rollout_output_without_metadata()
    rollout_output.non_tensor_batch["reward/helpfulness_score"] = np.asarray(
        [1.5, 2.5, 3.5, 4.5],
        dtype=object,
    )
    rollout_output.meta_info["reward_extra_keys"] = ["reward/helpfulness_score"]

    attach_ref_rollout_metadata(
        rollout_output=rollout_output,
        metadata=extract_ref_rollout_metadata(request_batch),
    )

    rows = ref_rollout_output_to_store_rows(
        rollout_output=rollout_output,
        tokenizer=FakeTokenizer(),
        ref_version="ref_v1",
        num_ref_completions=2,
    )

    assert rows == [
        {
            "prompt_id": "p0",
            K.DATASET_INDEX: 10,
            K.REF_VERSION: "ref_v1",
            "ref_completions": ["11 12", "13"],
            "ref_rewards": [1.0, 2.0],
            "reward_extra_info": [
                {"reward/helpfulness_score": 1.5},
                {"reward/helpfulness_score": 2.5},
            ],
        },
        {
            "prompt_id": "p1",
            K.DATASET_INDEX: 11,
            K.REF_VERSION: "ref_v1",
            "ref_completions": ["21 22", "23"],
            "ref_rewards": [3.0, 4.0],
            "reward_extra_info": [
                {"reward/helpfulness_score": 3.5},
                {"reward/helpfulness_score": 4.5},
            ],
        },
    ]


def test_ref_rollout_output_to_store_rows_requires_complete_ref_indices():
    request_batch = ref_reward_rollout_input_from_prompt_records(
        prompt_records=[_prompt("p0")],
        ref_version="ref_v1",
        num_ref_completions=2,
        config={"data_source": "activeultrafeedback"},
        dataset_indices=[0],
    )
    full_rollout_output = _rollout_output_without_metadata()
    rollout_output = DataProto.from_dict(
        tensors={
            key: value[:1]
            for key, value in full_rollout_output.batch.items()
        },
        non_tensors={},
        meta_info={},
    )
    attach_ref_rollout_metadata(
        rollout_output=rollout_output,
        metadata={
            key: values[:1]
            for key, values in extract_ref_rollout_metadata(request_batch).items()
        },
    )

    with pytest.raises(ValueError, match="expected"):
        ref_rollout_output_to_store_rows(
            rollout_output=rollout_output,
            tokenizer=FakeTokenizer(),
            ref_version="ref_v1",
            num_ref_completions=2,
        )
