import numpy as np
import pytest

from batch import keys as K
from batch.candidate_plan import build_candidate_plan
from batch.source_schedule import FixedCountsSourceScheduler
from data.offline_selector import MinMaxRewardSelector
from data.schemas import PromptRecord
from verl_adapters.rollout import online_rollout_requests_to_dataproto


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
    assert data.meta_info["source"] == "online"
    assert data.meta_info["validate"] is False


def test_online_rollout_requests_to_dataproto_supports_reward_model_metadata() -> None:
    requests = make_online_requests()

    reward_model = {
        "style": "rule",
        "ground_truth": "42",
    }

    data = online_rollout_requests_to_dataproto(
        requests=requests,
        config={
            "agent_name": "single_turn_agent",
            "reward_model": reward_model,
        },
    )

    assert K.REWARD_MODEL in data.non_tensor_batch
    assert data.non_tensor_batch[K.REWARD_MODEL][0] == reward_model
    assert data.non_tensor_batch[K.AGENT_NAME].tolist() == ["single_turn_agent"] * 4


def test_online_rollout_requests_to_dataproto_merges_meta_info() -> None:
    requests = make_online_requests()

    data = online_rollout_requests_to_dataproto(
        requests=requests,
        config={
            "agent_name": "tool_agent",
            "validate": True,
        },
        meta_info={
            "global_steps": 5,
        },
    )

    assert data.meta_info["validate"] is True
    assert data.meta_info["global_steps"] == 5


def test_online_rollout_requests_to_dataproto_rejects_empty_requests() -> None:
    with pytest.raises(ValueError, match="empty"):
        online_rollout_requests_to_dataproto(
            requests=[],
            config={"agent_name": "tool_agent"},
        )


def test_online_rollout_requests_to_dataproto_requires_agent_name() -> None:
    with pytest.raises(ValueError, match="agent_name"):
        online_rollout_requests_to_dataproto(
            requests=make_online_requests(),
            config={},
        )
