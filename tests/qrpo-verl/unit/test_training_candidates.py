from __future__ import annotations

import pytest

from batch.source_schedule import SourceCounts
from batch.training_candidates import build_training_candidates
from data.offline_selector import MinMaxRewardSelector
from data.schemas import OfflineSelection, PromptRecord


def make_prompt_record(prompt_id: str) -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        prompt_messages=(
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": f"Prompt {prompt_id}"},
        ),
        ref_rewards=(0.1, 0.2, 0.3),
        offline_trajectories=(
            ({"role": "assistant", "content": "low"},),
            ({"role": "assistant", "content": "mid"},),
            ({"role": "assistant", "content": "high"},),
        ),
        offline_rewards=(0.0, 0.5, 1.0),
        tools=None,
    )


def make_counts(
    *,
    prompt_index: int,
    prompt_id: str,
    n_online: int,
    n_offline: int,
) -> SourceCounts:
    return SourceCounts(
        prompt_index=prompt_index,
        prompt_id=prompt_id,
        n_online=n_online,
        n_offline=n_offline,
    )


def test_build_training_candidates_from_source_counts_with_selector_object() -> None:
    prompts = [
        make_prompt_record("p0"),
        make_prompt_record("p1"),
    ]
    source_counts = [
        make_counts(prompt_index=0, prompt_id="p0", n_online=1, n_offline=2),
        make_counts(prompt_index=1, prompt_id="p1", n_online=1, n_offline=1),
    ]

    selector = MinMaxRewardSelector()

    offline_candidates, online_requests = build_training_candidates(
        prompt_records=prompts,
        source_counts=source_counts,
        offline_selector=selector,
        actor_version=10,
    )

    assert [candidate.trajectory_id for candidate in offline_candidates] == [
        "p0::offline::slot_0::idx_2::max_reward",
        "p0::offline::slot_1::idx_0::min_reward",
        "p1::offline::slot_0::idx_2::max_reward",
    ]

    assert [candidate.reward for candidate in offline_candidates] == [
        1.0,
        0.0,
        1.0,
    ]

    assert [candidate.trajectory_messages for candidate in offline_candidates] == [
        ({"role": "assistant", "content": "high"},),
        ({"role": "assistant", "content": "low"},),
        ({"role": "assistant", "content": "high"},),
    ]

    assert [request.trajectory_id for request in online_requests] == [
        "p0::online::actor_10::0",
        "p1::online::actor_10::0",
    ]

    assert [request.prompt_index for request in online_requests] == [0, 1]
    assert [request.prompt_id for request in online_requests] == ["p0", "p1"]
    assert [request.online_index for request in online_requests] == [0, 0]

    assert online_requests[0].prompt_messages == prompts[0].prompt_messages
    assert online_requests[0].ref_rewards == prompts[0].ref_rewards
    assert online_requests[0].tools is None


def test_build_training_candidates_supports_callable_offline_selector() -> None:
    prompt = make_prompt_record("p0")
    source_counts = [
        make_counts(prompt_index=0, prompt_id="p0", n_online=0, n_offline=2),
    ]

    calls = []

    def offline_selector(prompt_record: PromptRecord, n_offline: int):
        calls.append((prompt_record.prompt_id, n_offline))
        return [
            OfflineSelection(offline_index=1, selection_reason="custom_mid"),
            OfflineSelection(offline_index=0, selection_reason="custom_low"),
        ]

    offline_candidates, online_requests = build_training_candidates(
        prompt_records=[prompt],
        source_counts=source_counts,
        offline_selector=offline_selector,
        actor_version="abc",
    )

    assert calls == [("p0", 2)]

    assert [candidate.trajectory_id for candidate in offline_candidates] == [
        "p0::offline::slot_0::idx_1::custom_mid",
        "p0::offline::slot_1::idx_0::custom_low",
    ]

    assert [candidate.reward for candidate in offline_candidates] == [0.5, 0.0]
    assert online_requests == []


def test_build_training_candidates_skips_offline_selector_when_no_offline() -> None:
    prompt = make_prompt_record("p0")
    source_counts = [
        make_counts(prompt_index=0, prompt_id="p0", n_online=2, n_offline=0),
    ]

    def offline_selector(prompt: PromptRecord, n_offline: int):
        raise AssertionError("offline_selector should not be called")

    offline_candidates, online_requests = build_training_candidates(
        prompt_records=[prompt],
        source_counts=source_counts,
        offline_selector=offline_selector,
        actor_version=3,
    )

    assert offline_candidates == []
    assert [request.trajectory_id for request in online_requests] == [
        "p0::online::actor_3::0",
        "p0::online::actor_3::1",
    ]
    assert [request.prompt_index for request in online_requests] == [0, 0]
    assert [request.online_index for request in online_requests] == [0, 1]


def test_build_training_candidates_skips_online_when_no_online() -> None:
    prompt = make_prompt_record("p0")
    source_counts = [
        make_counts(prompt_index=0, prompt_id="p0", n_online=0, n_offline=2),
    ]

    selector = MinMaxRewardSelector()

    offline_candidates, online_requests = build_training_candidates(
        prompt_records=[prompt],
        source_counts=source_counts,
        offline_selector=selector,
        actor_version=3,
    )

    assert [candidate.trajectory_id for candidate in offline_candidates] == [
        "p0::offline::slot_0::idx_2::max_reward",
        "p0::offline::slot_1::idx_0::min_reward",
    ]
    assert online_requests == []


def test_build_training_candidates_propagates_tools_to_online_requests() -> None:
    tools = (
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "parameters": {},
            },
        },
    )
    prompt = PromptRecord(
        prompt_id="p0",
        prompt_messages=(
            {"role": "user", "content": "Use a tool."},
        ),
        ref_rewards=(0.1, 0.2),
        offline_trajectories=(),
        offline_rewards=(),
        tools=tools,
    )

    offline_candidates, online_requests = build_training_candidates(
        prompt_records=[prompt],
        source_counts=[
            make_counts(prompt_index=0, prompt_id="p0", n_online=1, n_offline=0),
        ],
        offline_selector=lambda prompt, n_offline: [],
        actor_version=1,
    )

    assert offline_candidates == []
    assert len(online_requests) == 1
    assert online_requests[0].tools == tools


def test_build_training_candidates_rejects_length_mismatch() -> None:
    prompts = [
        make_prompt_record("p0"),
        make_prompt_record("p1"),
    ]

    with pytest.raises(ValueError, match="same length"):
        build_training_candidates(
            prompt_records=prompts,
            source_counts=[
                make_counts(prompt_index=0, prompt_id="p0", n_online=1, n_offline=0),
            ],
            offline_selector=lambda prompt, n_offline: [],
            actor_version=1,
        )


def test_build_training_candidates_rejects_prompt_id_mismatch() -> None:
    prompt = make_prompt_record("p0")

    with pytest.raises(ValueError, match="prompt_id"):
        build_training_candidates(
            prompt_records=[prompt],
            source_counts=[
                make_counts(prompt_index=0, prompt_id="wrong", n_online=1, n_offline=0),
            ],
            offline_selector=lambda prompt, n_offline: [],
            actor_version=1,
        )


def test_build_training_candidates_rejects_negative_online_count() -> None:
    prompt = make_prompt_record("p0")

    with pytest.raises(ValueError, match="negative n_online"):
        build_training_candidates(
            prompt_records=[prompt],
            source_counts=[
                make_counts(prompt_index=0, prompt_id="p0", n_online=-1, n_offline=1),
            ],
            offline_selector=lambda prompt, n_offline: [],
            actor_version=1,
        )


def test_build_training_candidates_rejects_negative_offline_count() -> None:
    prompt = make_prompt_record("p0")

    with pytest.raises(ValueError, match="negative n_offline"):
        build_training_candidates(
            prompt_records=[prompt],
            source_counts=[
                make_counts(prompt_index=0, prompt_id="p0", n_online=1, n_offline=-1),
            ],
            offline_selector=lambda prompt, n_offline: [],
            actor_version=1,
        )


def test_build_training_candidates_rejects_empty_counts() -> None:
    prompt = make_prompt_record("p0")

    with pytest.raises(ValueError, match="requests no samples"):
        build_training_candidates(
            prompt_records=[prompt],
            source_counts=[
                make_counts(prompt_index=0, prompt_id="p0", n_online=0, n_offline=0),
            ],
            offline_selector=lambda prompt, n_offline: [],
            actor_version=1,
        )


def test_build_training_candidates_rejects_wrong_number_of_offline_selections() -> None:
    prompt = make_prompt_record("p0")

    with pytest.raises(ValueError, match="expected 2"):
        build_training_candidates(
            prompt_records=[prompt],
            source_counts=[
                make_counts(prompt_index=0, prompt_id="p0", n_online=0, n_offline=2),
            ],
            offline_selector=lambda prompt, n_offline: [
                OfflineSelection(offline_index=0, selection_reason="only_one"),
            ],
            actor_version=1,
        )


def test_build_training_candidates_rejects_offline_selection_out_of_range() -> None:
    prompt = make_prompt_record("p0")

    with pytest.raises(IndexError, match="out of range"):
        build_training_candidates(
            prompt_records=[prompt],
            source_counts=[
                make_counts(prompt_index=0, prompt_id="p0", n_online=0, n_offline=1),
            ],
            offline_selector=lambda prompt, n_offline: [
                OfflineSelection(offline_index=99, selection_reason="bad"),
            ],
            actor_version=1,
        )


def test_build_training_candidates_rejects_missing_offline_reward() -> None:
    prompt = PromptRecord(
        prompt_id="p0",
        prompt_messages=(
            {"role": "user", "content": "Prompt"},
        ),
        ref_rewards=(0.1, 0.2),
        offline_trajectories=(
            ({"role": "assistant", "content": "A"},),
        ),
        offline_rewards=(),
        tools=None,
    )

    with pytest.raises(IndexError, match="no reward"):
        build_training_candidates(
            prompt_records=[prompt],
            source_counts=[
                make_counts(prompt_index=0, prompt_id="p0", n_online=0, n_offline=1),
            ],
            offline_selector=lambda prompt, n_offline: [
                OfflineSelection(offline_index=0, selection_reason="bad"),
            ],
            actor_version=1,
        )
