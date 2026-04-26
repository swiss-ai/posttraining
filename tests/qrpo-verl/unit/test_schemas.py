import pytest

from data.schemas import OfflineTrajectoryCandidate, PromptRecord, normalize_messages


def prompt_messages():
    return (
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2 + 2?"},
    )


def assistant_trajectory(text: str):
    return (
        {"role": "assistant", "content": text},
    )


def tool_trajectory():
    return (
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": {"expr": "2+2"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "calculator",
            "content": "4",
        },
        {
            "role": "assistant",
            "content": "The answer is 4.",
        },
    )


def test_normalize_messages_accepts_arbitrary_message_fields() -> None:
    messages = [
        {"role": "system", "content": "System message."},
        {"role": "developer", "content": "Developer instruction."},
        {"role": "user", "content": "Question."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "calculator"}],
        },
    ]

    normalized = normalize_messages(messages, field_name="messages")

    assert isinstance(normalized, tuple)
    assert normalized[1]["role"] == "developer"
    assert normalized[3]["tool_calls"] == [{"name": "calculator"}]


def test_normalize_messages_rejects_missing_role() -> None:
    with pytest.raises(ValueError, match="role"):
        normalize_messages(
            [{"content": "missing role"}],
            field_name="messages",
        )


def test_prompt_record_accepts_tool_trajectory() -> None:
    record = PromptRecord(
        prompt_id="p0",
        prompt_messages=prompt_messages(),
        ref_rewards=(0.1, 0.2, 0.3),
        offline_trajectories=(tool_trajectory(),),
        offline_rewards=(1.0,),
    )

    record.validate(num_ref_rewards=3)


def test_prompt_record_rejects_empty_prompt_id() -> None:
    record = PromptRecord(
        prompt_id="",
        prompt_messages=prompt_messages(),
        ref_rewards=(0.1, 0.2),
        offline_trajectories=(assistant_trajectory("answer"),),
        offline_rewards=(1.0,),
    )

    with pytest.raises(ValueError, match="prompt_id"):
        record.validate(num_ref_rewards=2)


def test_prompt_record_rejects_wrong_num_ref_rewards() -> None:
    record = PromptRecord(
        prompt_id="p0",
        prompt_messages=prompt_messages(),
        ref_rewards=(0.1, 0.2),
        offline_trajectories=(assistant_trajectory("answer"),),
        offline_rewards=(1.0,),
    )

    with pytest.raises(ValueError, match="expected 3"):
        record.validate(num_ref_rewards=3)


def test_prompt_record_rejects_nonfinite_ref_reward() -> None:
    record = PromptRecord(
        prompt_id="p0",
        prompt_messages=prompt_messages(),
        ref_rewards=(0.1, float("nan")),
        offline_trajectories=(assistant_trajectory("answer"),),
        offline_rewards=(1.0,),
    )

    with pytest.raises(ValueError, match="non-finite ref reward"):
        record.validate(num_ref_rewards=2)


def test_prompt_record_rejects_mismatched_offline_lengths() -> None:
    record = PromptRecord(
        prompt_id="p0",
        prompt_messages=prompt_messages(),
        ref_rewards=(0.1, 0.2, 0.3),
        offline_trajectories=(
            assistant_trajectory("answer 0"),
            assistant_trajectory("answer 1"),
        ),
        offline_rewards=(1.0,),
    )

    with pytest.raises(ValueError, match="offline trajectories but 1 offline rewards"):
        record.validate(num_ref_rewards=3)


def test_prompt_record_rejects_missing_message_role() -> None:
    record = PromptRecord(
        prompt_id="p0",
        prompt_messages=(
            {"content": "missing role"},
        ),
        ref_rewards=(0.1, 0.2),
        offline_trajectories=(assistant_trajectory("answer"),),
        offline_rewards=(1.0,),
    )

    with pytest.raises(ValueError, match="role"):
        record.validate(num_ref_rewards=2)


def test_prompt_record_rejects_empty_offline_trajectory() -> None:
    record = PromptRecord(
        prompt_id="p0",
        prompt_messages=prompt_messages(),
        ref_rewards=(0.1, 0.2),
        offline_trajectories=((),),
        offline_rewards=(1.0,),
    )

    with pytest.raises(ValueError, match="empty"):
        record.validate(num_ref_rewards=2)


def test_prompt_record_rejects_nonfinite_offline_reward() -> None:
    record = PromptRecord(
        prompt_id="p0",
        prompt_messages=prompt_messages(),
        ref_rewards=(0.1, 0.2),
        offline_trajectories=(assistant_trajectory("answer"),),
        offline_rewards=(float("inf"),),
    )

    with pytest.raises(ValueError, match="offline reward 0 is non-finite"):
        record.validate(num_ref_rewards=2)


def test_prompt_record_can_skip_deep_message_validation() -> None:
    record = PromptRecord(
        prompt_id="p0",
        prompt_messages=(
            {"content": "missing role"},
        ),
        ref_rewards=(0.1, 0.2),
        offline_trajectories=(
            ({"content": "also missing role"},),
        ),
        offline_rewards=(1.0,),
    )

    record.validate(num_ref_rewards=2, deep=False)


def test_offline_trajectory_candidate_accepts_trajectory_messages() -> None:
    candidate = OfflineTrajectoryCandidate(
        prompt_id="p0",
        trajectory_id="p0::offline::0",
        prompt_messages=prompt_messages(),
        trajectory_messages=assistant_trajectory("answer"),
        reward=1.0,
        ref_rewards=(0.1, 0.2),
    )

    candidate.validate()


def test_offline_trajectory_candidate_rejects_empty_prompt_id() -> None:
    candidate = OfflineTrajectoryCandidate(
        prompt_id="",
        trajectory_id="p0::offline::0",
        prompt_messages=prompt_messages(),
        trajectory_messages=assistant_trajectory("answer"),
        reward=1.0,
        ref_rewards=(0.1, 0.2),
    )

    with pytest.raises(ValueError, match="prompt_id"):
        candidate.validate()


def test_offline_trajectory_candidate_rejects_empty_trajectory_id() -> None:
    candidate = OfflineTrajectoryCandidate(
        prompt_id="p0",
        trajectory_id="",
        prompt_messages=prompt_messages(),
        trajectory_messages=assistant_trajectory("answer"),
        reward=1.0,
        ref_rewards=(0.1, 0.2),
    )

    with pytest.raises(ValueError, match="trajectory_id"):
        candidate.validate()


def test_offline_trajectory_candidate_rejects_empty_trajectory_messages() -> None:
    candidate = OfflineTrajectoryCandidate(
        prompt_id="p0",
        trajectory_id="p0::offline::0",
        prompt_messages=prompt_messages(),
        trajectory_messages=(),
        reward=1.0,
        ref_rewards=(0.1, 0.2),
    )

    with pytest.raises(ValueError, match="empty trajectory_messages"):
        candidate.validate()


def test_offline_trajectory_candidate_rejects_nonfinite_reward() -> None:
    candidate = OfflineTrajectoryCandidate(
        prompt_id="p0",
        trajectory_id="p0::offline::0",
        prompt_messages=prompt_messages(),
        trajectory_messages=assistant_trajectory("answer"),
        reward=float("nan"),
        ref_rewards=(0.1, 0.2),
    )

    with pytest.raises(ValueError, match="non-finite reward"):
        candidate.validate()


def test_offline_trajectory_candidate_rejects_empty_ref_rewards() -> None:
    candidate = OfflineTrajectoryCandidate(
        prompt_id="p0",
        trajectory_id="p0::offline::0",
        prompt_messages=prompt_messages(),
        trajectory_messages=assistant_trajectory("answer"),
        reward=1.0,
        ref_rewards=(),
    )

    with pytest.raises(ValueError, match="no ref_rewards"):
        candidate.validate()


def test_offline_trajectory_candidate_can_skip_deep_message_validation() -> None:
    candidate = OfflineTrajectoryCandidate(
        prompt_id="p0",
        trajectory_id="p0::offline::0",
        prompt_messages=(
            {"content": "missing role"},
        ),
        trajectory_messages=(
            {"content": "missing role"},
        ),
        reward=1.0,
        ref_rewards=(0.1, 0.2),
    )

    candidate.validate(deep=False)
