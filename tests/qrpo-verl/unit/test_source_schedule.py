import pytest

from batch.source_schedule import (
    FixedCountsSourceScheduler,
    SingleCompletionMixtureSourceScheduler,
    build_rollout_prompt_indices,
    build_source_scheduler,
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


def test_single_completion_mixture_emits_exactly_one_source_per_prompt() -> None:
    prompts = [prompt("p0"), prompt("p1"), prompt("p2")]

    scheduler = SingleCompletionMixtureSourceScheduler(
        offline_probability=0.5,
        seed=123,
    )
    counts = scheduler.plan(prompts, global_step=7)

    assert [count.prompt_index for count in counts] == [0, 1, 2]
    assert [count.prompt_id for count in counts] == ["p0", "p1", "p2"]
    assert all(count.n_online + count.n_offline == 1 for count in counts)
    assert all(count.n_online in {0, 1} for count in counts)
    assert all(count.n_offline in {0, 1} for count in counts)


def test_single_completion_mixture_extreme_probabilities() -> None:
    prompts = [prompt("p0"), prompt("p1")]

    all_online = SingleCompletionMixtureSourceScheduler(
        offline_probability=0.0,
        seed=1,
    )
    assert [(count.n_online, count.n_offline) for count in all_online.plan(prompts)] == [
        (1, 0),
        (1, 0),
    ]
    assert all_online.can_emit_online
    assert not all_online.can_emit_offline

    all_offline = SingleCompletionMixtureSourceScheduler(
        offline_probability=1.0,
        seed=1,
    )
    assert [(count.n_online, count.n_offline) for count in all_offline.plan(prompts)] == [
        (0, 1),
        (0, 1),
    ]
    assert not all_offline.can_emit_online
    assert all_offline.can_emit_offline


def test_single_completion_mixture_is_deterministic_for_resume() -> None:
    prompts = [prompt(f"p{i}") for i in range(8)]
    scheduler = SingleCompletionMixtureSourceScheduler(
        offline_probability=0.5,
        seed=42,
    )

    first = scheduler.plan(prompts, global_step=11)
    second = scheduler.plan(prompts, global_step=11)
    different_step = scheduler.plan(prompts, global_step=12)

    assert first == second
    assert first != different_step


def test_single_completion_mixture_adjusts_online_count_divisibility() -> None:
    prompts = [prompt(f"p{i}") for i in range(17)]
    scheduler = SingleCompletionMixtureSourceScheduler(
        offline_probability=0.5,
        seed=42,
    )

    counts = scheduler.plan(
        prompts,
        global_step=11,
        online_count_divisible_by=4,
    )

    assert sum(count.n_online for count in counts) % 4 == 0
    assert all(count.n_online + count.n_offline == 1 for count in counts)


def test_single_completion_mixture_preserves_expected_probability_with_divisibility() -> None:
    prompts = [prompt(f"p{i}") for i in range(128)]
    scheduler = SingleCompletionMixtureSourceScheduler(
        offline_probability=0.1,
        seed=42,
    )

    offline_counts = [
        sum(
            count.n_offline
            for count in scheduler.plan(
                prompts,
                global_step=step,
                online_count_divisible_by=32,
            )
        )
        for step in range(1, 101)
    ]

    assert set(offline_counts) <= {0, 32}
    assert sum(offline_counts) / len(offline_counts) == pytest.approx(
        0.1 * len(prompts),
        abs=4.0,
    )


def test_single_completion_mixture_rejects_invalid_online_divisor() -> None:
    scheduler = SingleCompletionMixtureSourceScheduler(
        offline_probability=0.5,
        seed=42,
    )

    with pytest.raises(ValueError, match="online_count_divisible_by"):
        scheduler.plan([prompt("p0")], online_count_divisible_by=0)


def test_single_completion_mixture_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="offline_probability must be in"):
        SingleCompletionMixtureSourceScheduler(offline_probability=-0.1)

    with pytest.raises(ValueError, match="offline_probability must be in"):
        SingleCompletionMixtureSourceScheduler(offline_probability=1.1)


def test_build_source_scheduler_builds_supported_schedulers() -> None:
    fixed = build_source_scheduler(
        {
            "name": "fixed_counts",
            "n_online": 1,
            "n_offline": 0,
        }
    )
    assert isinstance(fixed, FixedCountsSourceScheduler)

    mixture = build_source_scheduler(
        {
            "name": "single_completion_mixture",
            "offline_probability": 0.25,
            "seed": 9,
        }
    )
    assert isinstance(mixture, SingleCompletionMixtureSourceScheduler)


def test_build_source_scheduler_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown source schedule"):
        build_source_scheduler({"name": "unknown"})
