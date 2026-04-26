import pytest

from batch.source_schedule import (
    FixedCountsSourceScheduler,
    build_rollout_prompt_indices,
)
from data.schemas import PromptRecord


def trajectory(text: str):
    return ({"role": "assistant", "content": text},)


def prompt(prompt_id: str) -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        prompt_messages=({"role": "user", "content": f"Prompt {prompt_id}"},),
        ref_rewards=(0.1, 0.2, 0.3),
        offline_trajectories=(trajectory("bad"), trajectory("good")),
        offline_rewards=(-1.0, 1.0),
    )


def test_fixed_counts_scheduler_assigns_same_counts_to_each_prompt() -> None:
    prompts = [prompt("p0"), prompt("p1")]

    scheduler = FixedCountsSourceScheduler(n_online=3, n_offline=2)
    counts = scheduler.plan(prompts)

    assert len(counts) == 2
    assert counts[0].prompt_index == 0
    assert counts[0].prompt_id == "p0"
    assert counts[0].n_online == 3
    assert counts[0].n_offline == 2
    assert counts[1].prompt_index == 1
    assert counts[1].prompt_id == "p1"


def test_build_rollout_prompt_indices_expands_prompts_for_rollout() -> None:
    prompts = [prompt("p0"), prompt("p1"), prompt("p2")]

    scheduler = FixedCountsSourceScheduler(n_online=2, n_offline=1)
    counts = scheduler.plan(prompts)

    assert build_rollout_prompt_indices(counts) == [0, 0, 1, 1, 2, 2]


def test_build_rollout_prompt_indices_allows_zero_online() -> None:
    prompts = [prompt("p0"), prompt("p1")]

    scheduler = FixedCountsSourceScheduler(n_online=0, n_offline=2)
    counts = scheduler.plan(prompts)

    assert build_rollout_prompt_indices(counts) == []


def test_fixed_counts_scheduler_from_config() -> None:
    scheduler = FixedCountsSourceScheduler.from_config({"n_online": 1, "n_offline": 2})

    assert scheduler.n_online == 1
    assert scheduler.n_offline == 2


def test_fixed_counts_scheduler_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="n_online must be non-negative"):
        FixedCountsSourceScheduler(n_online=-1, n_offline=1)

    with pytest.raises(ValueError, match="n_offline must be non-negative"):
        FixedCountsSourceScheduler(n_online=1, n_offline=-1)


def test_fixed_counts_scheduler_requires_at_least_one_completion() -> None:
    with pytest.raises(ValueError, match="At least one"):
        FixedCountsSourceScheduler(n_online=0, n_offline=0)
