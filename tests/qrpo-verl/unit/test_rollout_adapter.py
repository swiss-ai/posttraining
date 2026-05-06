import numpy as np
import pytest
import torch
from verl.protocol import DataProto

from batch import keys as K
from batch.candidate_plan import build_candidate_plan, OnlineRolloutRequest
from batch.source_schedule import FixedCountsSourceScheduler
from data.offline_selector import MinMaxRewardSelector
from data.schemas import PromptRecord
from verl_adapters.rollout import (
    attach_online_rollout_train_metadata,
    extract_online_rollout_train_metadata,
    online_rollout_requests_to_dataproto,
)


def trajectory(text: str):
    return ({"role": "assistant", "content": text},)


def prompt(prompt_id: str) -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        prompt_messages=(
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": f"Prompt {prompt_id}"},
        ),
        ref_rewards=(0.1, 0.2, 0.3),
        offline_trajectories=(trajectory("bad"), trajectory("good")),
        offline_rewards=(-1.0, 1.0),
    )


def make_online_requests():
    prompts = [prompt("p0"), prompt("p1")]
    source_counts = FixedCountsSourceScheduler(n_online=2, n_offline=0).plan(prompts)

    plan = build_candidate_plan(
        prompts=prompts,
        source_counts=source_counts,
        offline_selector=MinMaxRewardSelector(),
        actor_version=10,
        ref_version=3,
    )

    return plan.online_requests


def test_online_rollout_requests_to_dataproto_builds_agent_loop_input() -> None:
    requests = make_online_requests()

    data = online_rollout_requests_to_dataproto(
        requests=requests,
        config={
            "agent_name": "tool_agent",
            "data_source": "qrpo_debug",
        },
    )

    assert data.batch is None
    assert len(data) == 4

    assert K.RAW_PROMPT in data.non_tensor_batch
    assert K.AGENT_NAME in data.non_tensor_batch
    assert K.DATA_SOURCE in data.non_tensor_batch
    assert K.PROMPT_ID in data.non_tensor_batch
    assert K.TRAJECTORY_ID in data.non_tensor_batch
    assert K.SOURCE in data.non_tensor_batch
    assert K.REF_REWARDS in data.non_tensor_batch

    assert data.non_tensor_batch[K.AGENT_NAME].tolist() == ["tool_agent"] * 4
    assert data.non_tensor_batch[K.DATA_SOURCE].tolist() == ["qrpo_debug"] * 4
    assert data.non_tensor_batch[K.PROMPT_ID].tolist() == ["p0", "p0", "p1", "p1"]
    assert data.non_tensor_batch[K.SOURCE].tolist() == ["online"] * 4

    assert data.non_tensor_batch[K.TRAJECTORY_ID].tolist() == [
        "p0::online::actor_10::0",
        "p0::online::actor_10::1",
        "p1::online::actor_10::0",
        "p1::online::actor_10::1",
    ]

    raw_prompt = data.non_tensor_batch[K.RAW_PROMPT][0]
    assert isinstance(raw_prompt, tuple)
    assert raw_prompt[0]["role"] == "system"
    assert raw_prompt[1]["content"] == "Prompt p0"

    assert data.non_tensor_batch[K.RAW_PROMPT].shape == (4,)
    assert data.non_tensor_batch[K.REF_REWARDS].shape == (4,)
    assert data.non_tensor_batch[K.REF_REWARDS][0] == (0.1, 0.2, 0.3)

    assert data.meta_info["qrpo_batch_format"] == "online_rollout_requests"
    assert "source" not in data.meta_info
    assert data.non_tensor_batch[K.SOURCE].tolist() == [K.SOURCE_ONLINE] * len(data)
    assert data.meta_info["validate"] is False


def test_online_rollout_requests_to_dataproto_builds_reward_loop_metadata() -> None:
    requests = make_online_requests()

    data = online_rollout_requests_to_dataproto(
        requests=requests,
        config={
            "agent_name": "single_turn_agent",
            "data_source": "activeultrafeedback",
        },
    )

    assert K.REWARD_MODEL in data.non_tensor_batch
    assert K.EXTRA_INFO in data.non_tensor_batch
    assert K.AGENT_NAME in data.non_tensor_batch
    assert data.non_tensor_batch[K.AGENT_NAME].tolist() == ["single_turn_agent"] * 4

    reward_model = data.non_tensor_batch[K.REWARD_MODEL][0]
    assert "ground_truth" in reward_model
    assert reward_model["ground_truth"]["prompt_id"] == "p0"
    assert reward_model["ground_truth"]["trajectory_id"] == "p0::online::actor_10::0"
    assert reward_model["ground_truth"]["prompt"] == requests[0].prompt_messages

    extra_info = data.non_tensor_batch[K.EXTRA_INFO][0]
    assert extra_info["prompt_id"] == "p0"
    assert extra_info["trajectory_id"] == "p0::online::actor_10::0"
    assert extra_info["online_index"] == 0
    assert extra_info["prompt"] == requests[0].prompt_messages
    assert extra_info["ref_rewards"] == requests[0].ref_rewards


def test_online_rollout_requests_to_dataproto_merges_meta_info() -> None:
    requests = make_online_requests()

    data = online_rollout_requests_to_dataproto(
        requests=requests,
        config={
            "agent_name": "tool_agent",
            "data_source": "activeultrafeedback",
            "validate": True,
        },
        meta_info={
            "global_steps": 5,
        },
    )

    assert data.meta_info["validate"] is True
    assert data.meta_info["global_steps"] == 5
    assert "source" not in data.meta_info


def test_online_rollout_requests_to_dataproto_rejects_empty_requests() -> None:
    with pytest.raises(ValueError, match="empty"):
        online_rollout_requests_to_dataproto(
            requests=[],
            config={"agent_name": "tool_agent"},
        )


def test_online_rollout_requests_to_dataproto_allows_missing_agent_name() -> None:
    data = online_rollout_requests_to_dataproto(
        requests=make_online_requests(),
        config={
            "data_source": "activeultrafeedback",
        },
    )

    assert K.AGENT_NAME not in data.non_tensor_batch
    assert data.non_tensor_batch[K.DATA_SOURCE].tolist() == [
        "activeultrafeedback"
    ] * len(data)


def _online_request(
    *,
    prompt_id: str = "prompt-0",
    online_index: int = 0,
) -> OnlineRolloutRequest:
    return OnlineRolloutRequest(
        prompt_index=0,
        prompt_id=prompt_id,
        online_index=online_index,
        prompt_messages=(
            {"role": "user", "content": "What is 2 + 2?"},
        ),
        ref_rewards=(1.0, 2.0, 3.0),
        trajectory_id=f"{prompt_id}::online::{online_index}",
        tools=None,
    )


def test_online_rollout_requests_include_reward_loop_fields_without_agent_name():
    request = _online_request()

    data = online_rollout_requests_to_dataproto(
        requests=[request],
        config={
            "data_source": "activeultrafeedback",
            "agent_name": None,
            "validate": False,
        },
        meta_info={"global_steps": 7},
    )

    assert data.batch is None

    assert K.RAW_PROMPT in data.non_tensor_batch
    assert K.PROMPT_ID in data.non_tensor_batch
    assert K.TRAJECTORY_ID in data.non_tensor_batch
    assert K.SOURCE in data.non_tensor_batch
    assert K.DATA_SOURCE in data.non_tensor_batch
    assert K.REWARD_MODEL in data.non_tensor_batch
    assert K.EXTRA_INFO in data.non_tensor_batch
    assert K.REF_REWARDS in data.non_tensor_batch

    # agent_name is optional: if absent/null, VERL uses
    # actor_rollout_ref.rollout.agent.default_agent_loop.
    assert K.AGENT_NAME not in data.non_tensor_batch

    assert data.non_tensor_batch[K.PROMPT_ID].tolist() == ["prompt-0"]
    assert data.non_tensor_batch[K.TRAJECTORY_ID].tolist() == [
        "prompt-0::online::0"
    ]
    assert data.non_tensor_batch[K.SOURCE].tolist() == [K.SOURCE_ONLINE]
    assert data.non_tensor_batch[K.DATA_SOURCE].tolist() == ["activeultrafeedback"]

    reward_model = data.non_tensor_batch[K.REWARD_MODEL][0]
    assert reward_model["ground_truth"]["prompt_id"] == "prompt-0"
    assert reward_model["ground_truth"]["trajectory_id"] == "prompt-0::online::0"
    assert reward_model["ground_truth"]["prompt"] == request.prompt_messages

    extra_info = data.non_tensor_batch[K.EXTRA_INFO][0]
    assert extra_info["prompt_id"] == "prompt-0"
    assert extra_info["trajectory_id"] == "prompt-0::online::0"
    assert extra_info["online_index"] == 0
    assert extra_info["prompt"] == request.prompt_messages
    assert extra_info["ref_rewards"] == request.ref_rewards

    assert data.non_tensor_batch[K.REF_REWARDS].shape == (1,)
    assert data.non_tensor_batch[K.REF_REWARDS][0] == request.ref_rewards

    assert data.meta_info["qrpo_batch_format"] == "online_rollout_requests"
    assert data.meta_info["validate"] is False
    assert data.meta_info["global_steps"] == 7

    # Per-sample source belongs in non_tensor_batch, not meta_info.
    assert "source" not in data.meta_info


def test_online_rollout_requests_include_agent_name_when_configured():
    data = online_rollout_requests_to_dataproto(
        requests=[_online_request()],
        config={
            "data_source": "activeultrafeedback",
            "agent_name": "single_turn_agent",
        },
    )

    assert K.AGENT_NAME in data.non_tensor_batch
    assert data.non_tensor_batch[K.AGENT_NAME].tolist() == ["single_turn_agent"]


def test_online_rollout_requests_require_data_source_for_reward_loop():
    with pytest.raises(ValueError, match="data_source"):
        online_rollout_requests_to_dataproto(
            requests=[_online_request()],
            config={
                "agent_name": None,
            },
        )


def test_online_rollout_requests_preserve_multiple_requests_order():
    requests = [
        _online_request(prompt_id="prompt-0", online_index=0),
        _online_request(prompt_id="prompt-0", online_index=1),
        _online_request(prompt_id="prompt-1", online_index=0),
    ]

    data = online_rollout_requests_to_dataproto(
        requests=requests,
        config={"data_source": "activeultrafeedback"},
    )

    assert data.non_tensor_batch[K.PROMPT_ID].tolist() == [
        "prompt-0",
        "prompt-0",
        "prompt-1",
    ]
    assert data.non_tensor_batch[K.TRAJECTORY_ID].tolist() == [
        "prompt-0::online::0",
        "prompt-0::online::1",
        "prompt-1::online::0",
    ]
    assert [
        item["online_index"]
        for item in data.non_tensor_batch[K.EXTRA_INFO].tolist()
    ] == [0, 1, 0]


def test_extract_online_rollout_train_metadata_keeps_only_qrpo_training_fields():
    data = online_rollout_requests_to_dataproto(
        requests=make_online_requests(),
        config={"data_source": "activeultrafeedback"},
    )

    metadata = extract_online_rollout_train_metadata(data)

    assert set(metadata) == {
        K.PROMPT_ID,
        K.TRAJECTORY_ID,
        K.REF_REWARDS,
    }
    assert metadata[K.PROMPT_ID].tolist() == ["p0", "p0", "p1", "p1"]


def test_attach_online_rollout_train_metadata_restores_missing_fields():
    metadata_source = online_rollout_requests_to_dataproto(
        requests=make_online_requests(),
        config={"data_source": "activeultrafeedback"},
    )
    metadata = extract_online_rollout_train_metadata(metadata_source)

    prompts = torch.tensor([[101, 102], [201, 202], [301, 302], [401, 402]])
    responses = torch.tensor([[11, 12], [21, 22], [31, 32], [41, 42]])
    response_mask = torch.ones_like(responses, dtype=torch.bool)
    input_ids = torch.cat([prompts, responses], dim=-1)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).repeat(input_ids.shape[0], 1)
    rm_scores = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [0.0, 4.0],
        ],
        dtype=torch.float32,
    )

    rollout_output = DataProto.from_dict(
        tensors={
            K.PROMPTS: prompts,
            K.RESPONSES: responses,
            K.RESPONSE_MASK: response_mask,
            K.INPUT_IDS: input_ids,
            K.ATTENTION_MASK: attention_mask,
            K.POSITION_IDS: position_ids,
            "rm_scores": rm_scores,
        },
        non_tensors={},
        meta_info={},
    )

    attach_online_rollout_train_metadata(
        rollout_output=rollout_output,
        metadata=metadata,
    )

    assert rollout_output.non_tensor_batch[K.PROMPT_ID].tolist() == [
        "p0",
        "p0",
        "p1",
        "p1",
    ]
    assert rollout_output.non_tensor_batch[K.TRAJECTORY_ID].tolist() == [
        "p0::online::actor_10::0",
        "p0::online::actor_10::1",
        "p1::online::actor_10::0",
        "p1::online::actor_10::1",
    ]
    assert rollout_output.non_tensor_batch[K.REF_REWARDS][0] == (0.1, 0.2, 0.3)


def test_attach_online_rollout_train_metadata_rejects_length_mismatch():
    rollout_output = DataProto.from_dict(
        tensors={
            K.PROMPTS: torch.tensor([[101], [201]]),
            K.RESPONSES: torch.tensor([[11], [21]]),
            K.RESPONSE_MASK: torch.ones(2, 1, dtype=torch.bool),
            K.INPUT_IDS: torch.tensor([[101, 11], [201, 21]]),
            K.ATTENTION_MASK: torch.ones(2, 2, dtype=torch.long),
            K.POSITION_IDS: torch.tensor([[0, 1], [0, 1]]),
            "rm_scores": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        },
        non_tensors={},
        meta_info={},
    )

    metadata = {
        K.PROMPT_ID: np.asarray(["p0"], dtype=object),
        K.TRAJECTORY_ID: np.asarray(["t0"], dtype=object),
        K.REF_REWARDS: np.asarray([(0.1, 0.2)], dtype=object),
    }

    with pytest.raises(ValueError, match="expected 2"):
        attach_online_rollout_train_metadata(
            rollout_output=rollout_output,
            metadata=metadata,
        )
