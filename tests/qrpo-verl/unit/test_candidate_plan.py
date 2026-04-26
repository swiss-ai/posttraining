import pytest

from batch.candidate_plan import build_candidate_plan
from batch.source_schedule import FixedCountsSourceScheduler
from data.offline_selector import MinMaxRewardSelector
from data.schemas import PromptRecord


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
        offline_trajectories=(trajectory("bad"), trajectory("medium"), trajectory("good")),
        offline_rewards=(-1.0, 0.0, 2.0),
    )


def test_build_candidate_plan_creates_online_requests_and_offline_candidates() -> None:
    prompts = [prompt("p0"), prompt("p1")]

    source_counts = FixedCountsSourceScheduler(n_online=2, n_offline=2).plan(prompts)

    plan = build_candidate_plan(
        prompts=prompts,
        source_counts=source_counts,
        offline_selector=MinMaxRewardSelector(),
        actor_version=10,
        ref_version=3,
    )

    assert plan.rollout_prompt_indices == [0, 0, 1, 1]

    assert len(plan.online_requests) == 4
    assert [r.prompt_id for r in plan.online_requests] == ["p0", "p0", "p1", "p1"]
    assert [r.online_index for r in plan.online_requests] == [0, 1, 0, 1]
    assert plan.online_requests[0].trajectory_id == "p0::online::actor_10::0"

    assert len(plan.offline_candidates) == 4
    assert [c.prompt_id for c in plan.offline_candidates] == ["p0", "p0", "p1", "p1"]
    assert [c.reward for c in plan.offline_candidates[:2]] == [2.0, -1.0]
    assert [c.trajectory_messages for c in plan.offline_candidates[:2]] == [
        trajectory("good"),
        trajectory("bad"),
    ]
    assert [c.metadata["selection_reason"] for c in plan.offline_candidates[:2]] == [
        "max_reward",
        "min_reward",
    ]

    assert plan.offline_candidates[0].trajectory_id == "p0::offline::2"
    assert plan.offline_candidates[1].trajectory_id == "p0::offline::0"


def test_build_candidate_plan_allows_online_only() -> None:
    prompts = [prompt("p0")]
    source_counts = FixedCountsSourceScheduler(n_online=3, n_offline=0).plan(prompts)

    plan = build_candidate_plan(
        prompts=prompts,
        source_counts=source_counts,
        offline_selector=MinMaxRewardSelector(),
        actor_version=5,
        ref_version=1,
    )

    assert len(plan.online_requests) == 3
    assert len(plan.offline_candidates) == 0
    assert plan.rollout_prompt_indices == [0, 0, 0]


def test_build_candidate_plan_allows_offline_only() -> None:
    prompts = [prompt("p0")]
    source_counts = FixedCountsSourceScheduler(n_online=0, n_offline=2).plan(prompts)

    plan = build_candidate_plan(
        prompts=prompts,
        source_counts=source_counts,
        offline_selector=MinMaxRewardSelector(),
        actor_version=5,
        ref_version=1,
    )

    assert len(plan.online_requests) == 0
    assert len(plan.offline_candidates) == 2
    assert plan.rollout_prompt_indices == []


def test_build_candidate_plan_rejects_mismatched_source_counts() -> None:
    prompts = [prompt("p0")]
    source_counts = FixedCountsSourceScheduler(n_online=1, n_offline=1).plan(
        [prompt("different")]
    )

    with pytest.raises(ValueError, match="does not match"):
        build_candidate_plan(
            prompts=prompts,
            source_counts=source_counts,
            offline_selector=MinMaxRewardSelector(),
            actor_version=1,
            ref_version=1,
        )
