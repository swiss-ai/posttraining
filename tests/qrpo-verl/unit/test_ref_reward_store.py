from __future__ import annotations

import pytest
from datasets import Dataset

from data.schemas import PromptRecord
from ref_rewards import RefRewardStore


def _prompt(prompt_id: str, ref_rewards=(0.0,)) -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        prompt_messages=({"role": "user", "content": prompt_id},),
        ref_rewards=tuple(ref_rewards),
        offline_trajectories=(),
        offline_rewards=(),
    )


def test_ref_reward_store_saves_loads_and_attaches_refs(tmp_path):
    store = RefRewardStore(tmp_path / "refs")
    store.save_version_rows(
        ref_version="ref_v1",
        rows=[
            {
                "prompt_id": "p0",
                "dataset_index": 0,
                "ref_version": "ref_v1",
                "ref_completions": ["a", "b"],
                "ref_rewards": [1.0, 2.0],
                "reward_extra_info": [None, None],
            },
            {
                "prompt_id": "p1",
                "dataset_index": 1,
                "ref_version": "ref_v1",
                "ref_completions": ["c", "d"],
                "ref_rewards": [3.0, 4.0],
                "reward_extra_info": [None, None],
            },
        ],
        manifest={"source": "unit"},
    )

    assert store.has_complete_version("ref_v1")
    assert store.load_ref_rewards("ref_v1") == {
        "p0": (1.0, 2.0),
        "p1": (3.0, 4.0),
    }

    resolved = store.attach_ref_rewards(
        [_prompt("p0", ref_rewards=(-1.0,)), _prompt("p1", ref_rewards=(-2.0,))],
        ref_version="ref_v1",
    )

    assert resolved[0].ref_rewards == (1.0, 2.0)
    assert resolved[1].ref_rewards == (3.0, 4.0)


def test_ref_reward_store_imports_dataset_ref_rewards(tmp_path):
    dataset = Dataset.from_list(
        [
            {"prompt_id": "p0", "ref_rewards": [1.0, 2.0]},
            {"prompt_id": "p1", "ref_rewards": [3.0, 4.0]},
        ]
    )

    store = RefRewardStore(tmp_path / "refs")
    store.import_dataset_ref_rewards(
        dataset=dataset,
        dataset_config={
            "prompt_id_key": "prompt_id",
            "ref_rewards_key": "ref_rewards",
        },
        ref_version="dataset_v0",
    )

    assert store.load_ref_rewards("dataset_v0") == {
        "p0": (1.0, 2.0),
        "p1": (3.0, 4.0),
    }


def test_ref_reward_store_rejects_missing_prompt(tmp_path):
    store = RefRewardStore(tmp_path / "refs")
    store.save_version_rows(
        ref_version="ref_v1",
        rows=[
            {
                "prompt_id": "p0",
                "dataset_index": 0,
                "ref_version": "ref_v1",
                "ref_completions": ["a"],
                "ref_rewards": [1.0],
                "reward_extra_info": [None],
            }
        ],
        manifest={"source": "unit"},
    )

    with pytest.raises(KeyError, match="p1"):
        store.attach_ref_rewards([_prompt("p1")], ref_version="ref_v1")


def test_ref_reward_store_rejects_duplicate_dataset_prompt_ids(tmp_path):
    dataset = Dataset.from_list(
        [
            {"prompt_id": "p0", "ref_rewards": [1.0]},
            {"prompt_id": "p0", "ref_rewards": [2.0]},
        ]
    )
    store = RefRewardStore(tmp_path / "refs")

    with pytest.raises(ValueError, match="duplicate prompt_id"):
        store.import_dataset_ref_rewards(
            dataset=dataset,
            dataset_config={
                "prompt_id_key": "prompt_id",
                "ref_rewards_key": "ref_rewards",
            },
            ref_version="dataset_v0",
        )


def test_ref_reward_store_rejects_reusing_dataset_version_for_different_dataset(tmp_path):
    store = RefRewardStore(tmp_path / "refs")
    store.import_dataset_ref_rewards(
        dataset=Dataset.from_list(
            [{"prompt_id": "p0", "ref_rewards": [1.0]}]
        ),
        dataset_config={
            "prompt_id_key": "prompt_id",
            "ref_rewards_key": "ref_rewards",
        },
        ref_version="dataset_v0",
        metadata={"data_path": "old"},
    )

    with pytest.raises(ValueError, match="data_path"):
        store.import_dataset_ref_rewards(
            dataset=Dataset.from_list(
                [{"prompt_id": "p0", "ref_rewards": [1.0]}]
            ),
            dataset_config={
                "prompt_id_key": "prompt_id",
                "ref_rewards_key": "ref_rewards",
            },
            ref_version="dataset_v0",
            metadata={"data_path": "new"},
        )


def test_ref_reward_store_checks_complete_version_metadata(tmp_path):
    store = RefRewardStore(tmp_path / "refs")
    store.save_version_rows(
        ref_version="ref_v1",
        rows=[
            {
                "prompt_id": "p0",
                "dataset_index": 0,
                "ref_version": "ref_v1",
                "ref_completions": ["a"],
                "ref_rewards": [1.0],
                "reward_extra_info": [None],
            }
        ],
        manifest={"source": "generate", "prompt_count": 1},
    )

    store.check_complete_version_metadata(
        ref_version="ref_v1",
        metadata={"source": "generate", "prompt_count": 1},
    )

    with pytest.raises(ValueError, match="prompt_count"):
        store.check_complete_version_metadata(
            ref_version="ref_v1",
            metadata={"prompt_count": 2},
        )


def test_ref_reward_store_saves_and_loads_generation_chunk(tmp_path):
    store = RefRewardStore(tmp_path / "refs")
    rows = [
        {
            "prompt_id": "p0",
            "dataset_index": 0,
            "ref_version": "ref_v1",
            "ref_completions": ["a", "b"],
            "ref_rewards": [1.0, 2.0],
            "reward_extra_info": [None, {"judge": "unit"}],
        },
        {
            "prompt_id": "p1",
            "dataset_index": 1,
            "ref_version": "ref_v1",
            "ref_completions": ["c", "d"],
            "ref_rewards": [3.0, 4.0],
            "reward_extra_info": [None, None],
        },
    ]
    manifest = {
        "source": "generate",
        "actor_step": 0,
        "prompt_count": 2,
        "num_ref_completions": 2,
    }

    store.save_generation_chunk(
        ref_version="ref_v1",
        chunk_index=0,
        dataset_indices=[0, 1],
        rows=rows,
        manifest=manifest,
    )

    loaded = store.load_generation_chunk(
        ref_version="ref_v1",
        chunk_index=0,
        dataset_indices=[0, 1],
        manifest=manifest,
    )

    assert loaded == rows


def test_ref_reward_store_rejects_stale_generation_chunk(tmp_path):
    store = RefRewardStore(tmp_path / "refs")
    rows = [
        {
            "prompt_id": "p0",
            "dataset_index": 0,
            "ref_version": "ref_v1",
            "ref_completions": ["a"],
            "ref_rewards": [1.0],
            "reward_extra_info": [None],
        }
    ]

    store.save_generation_chunk(
        ref_version="ref_v1",
        chunk_index=0,
        dataset_indices=[0],
        rows=rows,
        manifest={"source": "generate", "actor_step": 0},
    )

    with pytest.raises(ValueError, match="different generation metadata"):
        store.load_generation_chunk(
            ref_version="ref_v1",
            chunk_index=0,
            dataset_indices=[0],
            manifest={"source": "generate", "actor_step": 1},
        )
