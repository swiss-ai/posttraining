import pytest
from datasets import Dataset

from data.dataset_adapter import (
    dataset_batch_to_prompt_records,
    row_to_prompt_record,
    rows_to_prompt_records,
)


def prompt_messages(question: str = "What is 2 + 2?"):
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": question},
    ]


def assistant_trajectory(text: str):
    return [
        {"role": "assistant", "content": text},
    ]


def tool_trajectory():
    return [
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
    ]


def test_row_to_prompt_record() -> None:
    row = {
        "prompt_id": "p0",
        "prompt_messages": prompt_messages(),
        "ref_rewards": [0.1, 0.2, 0.3],
        "offline_trajectories": [
            assistant_trajectory("3"),
            assistant_trajectory("4"),
            tool_trajectory(),
        ],
        "offline_rewards": [-1.0, 1.0, 1.0],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate arithmetic expressions.",
                    "parameters": {},
                },
            }
        ],
        "difficulty": "easy",
    }

    cfg = {
        "num_ref_rewards": 3,
        "metadata_keys": ["difficulty"],
        "validate_records": True,
    }

    record = row_to_prompt_record(row, cfg)

    assert record.prompt_id == "p0"
    assert record.prompt_messages[0]["role"] == "system"
    assert record.prompt_messages[1]["content"] == "What is 2 + 2?"

    assert record.ref_rewards == (0.1, 0.2, 0.3)

    assert len(record.offline_trajectories) == 3
    assert record.offline_trajectories[0] == (
        {"role": "assistant", "content": "3"},
    )
    assert record.offline_trajectories[1] == (
        {"role": "assistant", "content": "4"},
    )
    assert record.offline_trajectories[2][1]["role"] == "tool"

    assert record.offline_rewards == (-1.0, 1.0, 1.0)

    assert record.tools is not None
    assert record.tools[0]["function"]["name"] == "calculator"

    assert record.metadata == {"difficulty": "easy"}


def test_row_to_prompt_record_with_custom_column_names() -> None:
    row = {
        "id": "p0",
        "messages": prompt_messages(),
        "reference_rewards": [0.1, 0.2],
        "trajectories": [
            assistant_trajectory("bad"),
            assistant_trajectory("good"),
        ],
        "trajectory_rewards": [-1.0, 1.0],
    }

    cfg = {
        "prompt_id_key": "id",
        "prompt_messages_key": "messages",
        "ref_rewards_key": "reference_rewards",
        "offline_trajectories_key": "trajectories",
        "offline_rewards_key": "trajectory_rewards",
        "num_ref_rewards": 2,
        "validate_records": True,
    }

    record = row_to_prompt_record(row, cfg)

    assert record.prompt_id == "p0"
    assert record.ref_rewards == (0.1, 0.2)
    assert record.offline_rewards == (-1.0, 1.0)


def test_rows_to_prompt_records() -> None:
    rows = [
        {
            "prompt_id": "p0",
            "prompt_messages": prompt_messages("prompt 0"),
            "ref_rewards": [0.1, 0.2],
            "offline_trajectories": [
                assistant_trajectory("bad"),
                assistant_trajectory("good"),
            ],
            "offline_rewards": [-1.0, 1.0],
        },
        {
            "prompt_id": "p1",
            "prompt_messages": prompt_messages("prompt 1"),
            "ref_rewards": [0.3, 0.4],
            "offline_trajectories": [
                assistant_trajectory("bad"),
                assistant_trajectory("good"),
            ],
            "offline_rewards": [-0.5, 2.0],
        },
    ]

    records = rows_to_prompt_records(
        rows,
        {
            "num_ref_rewards": 2,
            "validate_records": True,
        },
    )

    assert [r.prompt_id for r in records] == ["p0", "p1"]
    assert records[1].prompt_messages[1]["content"] == "prompt 1"


def test_dataset_batch_to_prompt_records() -> None:
    dataset = Dataset.from_dict(
        {
            "prompt_id": ["p0", "p1", "p2"],
            "prompt_messages": [
                prompt_messages("prompt 0"),
                prompt_messages("prompt 1"),
                prompt_messages("prompt 2"),
            ],
            "ref_rewards": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            "offline_trajectories": [
                [assistant_trajectory("a"), assistant_trajectory("b")],
                [assistant_trajectory("c"), assistant_trajectory("d")],
                [assistant_trajectory("e"), assistant_trajectory("f")],
            ],
            "offline_rewards": [[0.0, 1.0], [-1.0, 2.0], [0.5, 0.7]],
        }
    )

    records = dataset_batch_to_prompt_records(
        dataset,
        indices=[2, 0],
        config={
            "num_ref_rewards": 2,
            "validate_records": True,
        },
    )

    assert [r.prompt_id for r in records] == ["p2", "p0"]
    assert records[0].offline_trajectories[0] == (
        {"role": "assistant", "content": "e"},
    )


def test_row_to_prompt_record_rejects_wrong_num_ref_rewards() -> None:
    row = {
        "prompt_id": "p0",
        "prompt_messages": prompt_messages(),
        "ref_rewards": [0.1],
        "offline_trajectories": [
            assistant_trajectory("a"),
            assistant_trajectory("b"),
        ],
        "offline_rewards": [0.0, 1.0],
    }

    with pytest.raises(ValueError, match="expected 2"):
        row_to_prompt_record(
            row,
            {
                "num_ref_rewards": 2,
                "validate_records": True,
            },
        )


def test_row_to_prompt_record_rejects_missing_required_key() -> None:
    row = {
        "prompt_id": "p0",
        "prompt_messages": prompt_messages(),
        "ref_rewards": [0.1, 0.2],
        "offline_trajectories": [
            assistant_trajectory("a"),
            assistant_trajectory("b"),
        ],
        # offline_rewards missing
    }

    with pytest.raises(KeyError, match="offline_rewards"):
        row_to_prompt_record(
            row,
            {
                "num_ref_rewards": 2,
                "validate_records": True,
            },
        )


def test_row_to_prompt_record_rejects_missing_message_role_when_validation_enabled() -> None:
    row = {
        "prompt_id": "p0",
        "prompt_messages": [
            {"content": "missing role"},
        ],
        "ref_rewards": [0.1, 0.2],
        "offline_trajectories": [
            assistant_trajectory("a"),
        ],
        "offline_rewards": [0.0],
    }

    with pytest.raises(ValueError, match="role"):
        row_to_prompt_record(
            row,
            {
                "num_ref_rewards": 2,
                "validate_records": True,
            },
        )


def test_row_to_prompt_record_skips_deep_message_validation_when_disabled() -> None:
    row = {
        "prompt_id": "p0",
        "prompt_messages": [
            {"content": "missing role"},
        ],
        "ref_rewards": [0.1, 0.2],
        "offline_trajectories": [
            assistant_trajectory("a"),
        ],
        "offline_rewards": [0.0],
    }

    record = row_to_prompt_record(
        row,
        {
            "num_ref_rewards": 2,
            "validate_records": False,
        },
    )

    assert record.prompt_id == "p0"


def test_row_to_prompt_record_rejects_mismatched_offline_lengths() -> None:
    row = {
        "prompt_id": "p0",
        "prompt_messages": prompt_messages(),
        "ref_rewards": [0.1, 0.2],
        "offline_trajectories": [
            assistant_trajectory("a"),
            assistant_trajectory("b"),
        ],
        "offline_rewards": [0.0],
    }

    with pytest.raises(ValueError, match="offline trajectories but 1 offline rewards"):
        row_to_prompt_record(
            row,
            {
                "num_ref_rewards": 2,
                "validate_records": True,
            },
        )


def test_row_to_prompt_record_checks_num_ref_rewards_even_without_full_validation() -> None:
    row = {
        "prompt_id": "p0",
        "prompt_messages": [
            {"content": "missing role"},
        ],
        "ref_rewards": [0.1],
        "offline_trajectories": [
            assistant_trajectory("a"),
        ],
        "offline_rewards": [0.0],
    }

    with pytest.raises(ValueError, match="expected 2"):
        row_to_prompt_record(
            row,
            {
                "num_ref_rewards": 2,
                "validate_records": False,
            },
        )
