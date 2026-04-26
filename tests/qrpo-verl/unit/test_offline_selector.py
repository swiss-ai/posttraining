import pytest

from data.offline_selector import MinMaxRewardSelector
from data.schemas import PromptRecord


def trajectory(text: str):
    return ({"role": "assistant", "content": text},)


def prompt(
    prompt_id: str = "p0",
    *,
    offline_trajectories=None,
    offline_rewards=None,
) -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        prompt_messages=({"role": "user", "content": f"Prompt {prompt_id}"},),
        ref_rewards=(0.1, 0.2, 0.3),
        offline_trajectories=offline_trajectories
        if offline_trajectories is not None
        else (trajectory("bad"), trajectory("medium"), trajectory("good")),
        offline_rewards=offline_rewards
        if offline_rewards is not None
        else (-1.0, 0.0, 2.0),
    )


def test_minmax_selector_selects_highest_then_lowest_reward() -> None:
    selected = MinMaxRewardSelector().select(prompt("p0"), n_offline=2)

    assert [s.selection_reason for s in selected] == ["max_reward", "min_reward"]
    assert [s.offline_index for s in selected] == [2, 0]


def test_minmax_selector_returns_only_max_when_n_offline_is_one() -> None:
    selected = MinMaxRewardSelector().select(prompt("p0"), n_offline=1)

    assert len(selected) == 1
    assert selected[0].selection_reason == "max_reward"
    assert selected[0].offline_index == 2


def test_minmax_selector_returns_empty_when_n_offline_is_zero() -> None:
    record = prompt("p0", offline_trajectories=(), offline_rewards=())

    assert MinMaxRewardSelector().select(record, n_offline=0) == []


def test_minmax_selector_returns_single_selection_for_single_trajectory_by_default() -> None:
    record = prompt(
        "p0",
        offline_trajectories=(trajectory("only"),),
        offline_rewards=(1.0,),
    )

    selected = MinMaxRewardSelector().select(record)

    assert len(selected) == 1
    assert selected[0].selection_reason == "max_reward"
    assert selected[0].offline_index == 0


def test_minmax_selector_can_duplicate_single_trajectory_when_configured() -> None:
    record = prompt(
        "p0",
        offline_trajectories=(trajectory("only"),),
        offline_rewards=(1.0,),
    )

    selected = MinMaxRewardSelector({"deduplicate_if_same_index": False}).select(record)

    assert len(selected) == 2
    assert [s.selection_reason for s in selected] == ["max_reward", "min_reward"]
    assert [s.offline_index for s in selected] == [0, 0]


def test_minmax_selector_errors_when_requesting_more_than_available() -> None:
    record = prompt(
        "p0",
        offline_trajectories=(trajectory("only"),),
        offline_rewards=(1.0,),
    )

    with pytest.raises(ValueError, match="at most 1 selections"):
        MinMaxRewardSelector().select(record, n_offline=2)


def test_minmax_selector_errors_on_empty_offline_trajectories_when_nonzero_requested() -> None:
    record = prompt("p0", offline_trajectories=(), offline_rewards=())

    with pytest.raises(ValueError, match="no offline trajectories"):
        MinMaxRewardSelector().select(record, n_offline=1)


def test_minmax_selector_errors_on_negative_n_offline() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MinMaxRewardSelector().select(prompt("p0"), n_offline=-1)
