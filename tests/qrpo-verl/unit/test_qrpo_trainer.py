from __future__ import annotations

import hashlib
import json
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from batch import keys as K
from batch.source_schedule import SourceCounts
from data.schemas import PromptRecord
from trainers.qrpo_trainer import (
    GLOBAL_POOL_ID,
    QRPOTrainer,
    build_qrpo_resource_pool_mapping,
    build_qrpo_role_worker_mapping,
    validate_qrpo_trainer_config,
)
from verl_adapters.engine_worker import QRPOEngineActorRolloutRefWorker
from verl_adapters.online_completion_logging import build_online_completion_log_rows


class FakeTokenizer:
    pad_token_id = 7

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in token_ids if token_id != 0)


class NoPadTokenizer:
    pad_token_id = None


class FakeLogger:
    def __init__(self) -> None:
        self.calls = []

    def log(self, *, data, step):
        self.calls.append((data, step))


class FakeCheckpointManager:
    def __init__(self) -> None:
        self.updated_steps = []
        self.sleep_calls = 0

    def update_weights(self, step):
        self.updated_steps.append(step)

    def sleep_replicas(self):
        self.sleep_calls += 1


class FakeSourceScheduler:
    def __init__(self, source_counts):
        self.source_counts = source_counts
        self.calls = []

    def plan(self, prompts):
        self.calls.append(prompts)
        return self.source_counts


class FakeActorRolloutWG:
    def __init__(self) -> None:
        self.to_calls = []
        self.sync_ref_calls = []
        self.load_ref_calls = []

    def to(self, device, *, model=True, optimizer=True, grad=True):
        self.to_calls.append(
            {
                "device": device,
                "model": model,
                "optimizer": optimizer,
                "grad": grad,
            }
        )

    def sync_ref_model_from_actor(self, *, local_path, global_step):
        self.sync_ref_calls.append(
            {
                "local_path": local_path,
                "global_step": global_step,
            }
        )

    def load_ref_model_checkpoint(self, *, local_path, del_local_after_load=False):
        self.load_ref_calls.append(
            {
                "local_path": local_path,
                "del_local_after_load": del_local_after_load,
            }
        )


class FakeRefRewardStore:
    def __init__(self, rewards_by_prompt_id=None) -> None:
        self.rewards_by_prompt_id = rewards_by_prompt_id or {}
        self.calls = []
        self.complete_versions = set()
        self.saved_versions = []
        self.saved_chunks = []
        self.chunks = {}
        self.manifests = {}

    def has_complete_version(self, ref_version):
        return ref_version in self.complete_versions

    def check_complete_version_metadata(self, *, ref_version, metadata):
        self.manifests.setdefault(ref_version, dict(metadata))
        self.calls.append(
            {
                "ref_version": ref_version,
                "metadata": dict(metadata),
            }
        )

    def save_version_rows(self, *, ref_version, rows, manifest, overwrite=False):
        self.saved_versions.append(
            {
                "ref_version": ref_version,
                "rows": list(rows),
                "manifest": dict(manifest),
                "overwrite": overwrite,
            }
        )
        self.complete_versions.add(ref_version)
        self.manifests[ref_version] = dict(manifest)

    def save_generation_chunk(
        self,
        *,
        ref_version,
        chunk_index,
        dataset_indices,
        rows,
        manifest,
        overwrite=False,
    ):
        payload = {
            "ref_version": ref_version,
            "chunk_index": chunk_index,
            "dataset_indices": list(dataset_indices),
            "rows": [dict(row) for row in rows],
            "manifest": dict(manifest),
            "overwrite": overwrite,
        }
        self.saved_chunks.append(payload)
        self.chunks[(ref_version, chunk_index)] = payload

    def load_generation_chunk(
        self,
        *,
        ref_version,
        chunk_index,
        dataset_indices,
        manifest,
    ):
        payload = self.chunks.get((ref_version, chunk_index))
        if payload is None:
            return None
        assert payload["dataset_indices"] == list(dataset_indices)
        assert payload["manifest"] == dict(manifest)
        return [dict(row) for row in payload["rows"]]

    def load_manifest(self, ref_version):
        return self.manifests.get(ref_version, {"source": "generate"})

    def model_checkpoint_path(self, ref_version):
        return f"/tmp/ref_reward_store/model_checkpoints/{ref_version}"

    def attach_ref_rewards(self, prompt_records, *, ref_version):
        self.calls.append(
            {
                "prompt_ids": [prompt.prompt_id for prompt in prompt_records],
                "ref_version": ref_version,
            }
        )
        resolved = []
        for prompt in prompt_records:
            rewards = self.rewards_by_prompt_id.get(prompt.prompt_id, prompt.ref_rewards)
            resolved.append(
                PromptRecord(
                    prompt_id=prompt.prompt_id,
                    prompt_messages=prompt.prompt_messages,
                    ref_rewards=tuple(rewards),
                    offline_trajectories=prompt.offline_trajectories,
                    offline_rewards=prompt.offline_rewards,
                    tools=prompt.tools,
                    metadata={
                        **prompt.metadata,
                        K.REF_VERSION: ref_version,
                    },
                )
            )
        return resolved


def make_config(**overrides):
    cfg = {
        "trainer": {
            "use_legacy_worker_impl": "disable",
            "total_epochs": 1,
            "val_before_train": False,
            "val_only": False,
            "save_freq": 0,
            "test_freq": 0,
            "project_name": "unit",
            "experiment_name": "qrpo",
            "logger": [],
        },
        "actor_rollout_ref": {
            "hybrid_engine": True,
            "actor": {
                "strategy": "fsdp",
                "use_kl_loss": False,
            },
            "rollout": {
                "temperature": 0.7,
                "multi_turn": {
                    "enable": False,
                },
            },
        },
        "algorithm": {
            "use_kl_in_reward": False,
        },
        "qrpo": {
            "beta": 6.0,
            "transform": "identity",
            "length_normalization": True,
        },
    }

    for path, value in overrides.items():
        cur = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    return OmegaConf.create(cfg)


def make_trainer_for_unit_test(config=None) -> QRPOTrainer:
    trainer = QRPOTrainer.__new__(QRPOTrainer)
    trainer.config = config if config is not None else make_config()
    trainer.tokenizer = FakeTokenizer()

    trainer.source_scheduler = None
    trainer.offline_selector = None
    trainer.online_reward_fn = None
    trainer.ref_reward_store = FakeRefRewardStore()
    trainer.current_ref_version = "unit_ref"

    return trainer


def make_qrpo_batch(
    *,
    prefix: int = 0,
    prompt_len: int = 2,
    response_len: int = 3,
    source: str = "offline",
    temperature: float = 0.7,
) -> DataProto:
    prompts = (
        torch.arange(2 * prompt_len, dtype=torch.long).reshape(2, prompt_len)
        + prefix
    )
    responses = (
        torch.arange(2 * response_len, dtype=torch.long).reshape(2, response_len)
        + prefix
        + 100
    )

    prompt_attention = torch.ones(2, prompt_len, dtype=torch.long)
    response_attention = torch.ones(2, response_len, dtype=torch.long)

    if prompt_len > 1:
        prompts[0, 0] = 0
        prompt_attention[0, 0] = 0

    if response_len > 1:
        responses[0, -1] = 0
        response_attention[0, -1] = 0

    response_mask = response_attention.bool().clone()
    if response_len > 2:
        response_mask[1, 1] = False

    input_ids = torch.cat([prompts, responses], dim=-1)
    attention_mask = torch.cat([prompt_attention, response_attention], dim=-1)

    return DataProto.from_dict(
        tensors={
            K.PROMPTS: prompts,
            K.RESPONSES: responses,
            K.RESPONSE_MASK: response_mask,
            K.INPUT_IDS: input_ids,
            K.ATTENTION_MASK: attention_mask,
            K.POSITION_IDS: torch.zeros_like(input_ids),
            K.TRAJECTORY_REWARD: torch.tensor([0.0, 4.0], dtype=torch.float32),
            K.REF_REWARDS: torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [1.0, 2.0, 3.0, 4.0],
                ],
                dtype=torch.float32,
            ),
        },
        non_tensors={
            K.PROMPT_ID: np.asarray(
                [f"p{prefix}_0", f"p{prefix}_1"],
                dtype=object,
            ),
            K.TRAJECTORY_ID: np.asarray(
                [f"t{prefix}_0", f"t{prefix}_1"],
                dtype=object,
            ),
            K.SOURCE: np.asarray([source] * 2, dtype=object),
        },
        meta_info={
            K.TEMPERATURE: temperature,
        },
    )


def make_fit_prompt(prompt_id: str) -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        prompt_messages=(
            {"role": "user", "content": prompt_id},
        ),
        ref_rewards=(0.1, 0.2),
        offline_trajectories=(),
        offline_rewards=(),
        tools=None,
    )


def test_validate_qrpo_trainer_config_accepts_minimal_valid_config() -> None:
    validate_qrpo_trainer_config(make_config())


def test_validate_qrpo_trainer_config_requires_new_stack() -> None:
    with pytest.raises(ValueError, match="use_legacy_worker_impl=disable"):
        validate_qrpo_trainer_config(
            make_config(**{"trainer.use_legacy_worker_impl": "auto"})
        )


def test_validate_qrpo_trainer_config_requires_supported_strategy() -> None:
    with pytest.raises(ValueError, match="strategy"):
        validate_qrpo_trainer_config(
            make_config(**{"actor_rollout_ref.actor.strategy": "weird"})
        )


def test_validate_qrpo_trainer_config_requires_hybrid_engine() -> None:
    with pytest.raises(ValueError, match="hybrid_engine"):
        validate_qrpo_trainer_config(
            make_config(**{"actor_rollout_ref.hybrid_engine": False})
        )


def test_validate_qrpo_trainer_config_rejects_actor_kl_loss() -> None:
    with pytest.raises(ValueError, match="use_kl_loss"):
        validate_qrpo_trainer_config(
            make_config(**{"actor_rollout_ref.actor.use_kl_loss": True})
        )


def test_validate_qrpo_trainer_config_rejects_reward_kl() -> None:
    with pytest.raises(ValueError, match="use_kl_in_reward"):
        validate_qrpo_trainer_config(
            make_config(**{"algorithm.use_kl_in_reward": True})
        )


def test_validate_qrpo_trainer_config_requires_beta() -> None:
    cfg = make_config()
    del cfg.qrpo.beta

    with pytest.raises(ValueError, match="qrpo.beta"):
        validate_qrpo_trainer_config(cfg)


def test_validate_qrpo_trainer_config_requires_boolean_length_normalization() -> None:
    with pytest.raises(ValueError, match="length_normalization"):
        validate_qrpo_trainer_config(
            make_config(**{"qrpo.length_normalization": "trainable_tokens"})
        )


def test_validate_qrpo_trainer_config_rejects_unknown_transform() -> None:
    with pytest.raises(ValueError, match="identity"):
        validate_qrpo_trainer_config(make_config(**{"qrpo.transform": "log"}))


def test_validate_qrpo_trainer_config_requires_rollout_n_one() -> None:
    with pytest.raises(ValueError, match="rollout.n=1"):
        validate_qrpo_trainer_config(
            make_config(**{"actor_rollout_ref.rollout.n": 2})
        )


def test_build_qrpo_role_worker_mapping_without_ray_remote() -> None:
    from verl.trainer.ppo.utils import Role

    mapping = build_qrpo_role_worker_mapping(use_ray_remote=False)

    assert mapping == {
        Role.ActorRolloutRef: QRPOEngineActorRolloutRefWorker,
    }


def test_build_qrpo_resource_pool_mapping() -> None:
    from verl.trainer.ppo.utils import Role

    mapping = build_qrpo_resource_pool_mapping()

    assert mapping == {
        Role.ActorRolloutRef: GLOBAL_POOL_ID,
    }


def test_qrpo_update_step_concatenates_adds_fields_ref_logprob_and_updates() -> None:
    trainer = make_trainer_for_unit_test()
    trainer.actor_rollout_wg = FakeActorRolloutWG()

    offline = make_qrpo_batch(
        prefix=0,
        prompt_len=2,
        response_len=3,
        source="offline",
    )
    online = make_qrpo_batch(
        prefix=1000,
        prompt_len=4,
        response_len=5,
        source="online",
    )

    calls = {
        "compute_ref_log_prob": 0,
        "update_actor": 0,
    }

    def fake_compute_ref_log_prob(self, train_batch):
        calls["compute_ref_log_prob"] += 1

        assert len(train_batch) == 4
        assert train_batch.batch[K.PROMPTS].shape == (4, 4)
        assert train_batch.batch[K.RESPONSES].shape == (4, 5)
        assert train_batch.batch[K.RESPONSE_MASK].shape == (4, 5)

        assert K.REF_QUANTILE in train_batch.batch
        assert K.TRANSFORMED_REWARD in train_batch.batch
        assert K.EFFECTIVE_BETA in train_batch.batch
        assert K.BETA_LOG_PARTITION in train_batch.batch

        return DataProto.from_dict(
            tensors={
                K.REF_LOG_PROBS: torch.zeros_like(
                    train_batch.batch[K.RESPONSES],
                    dtype=torch.float32,
                )
            }
        )

    def fake_update_actor(self, train_batch):
        calls["update_actor"] += 1

        assert K.REF_LOG_PROBS in train_batch.batch
        assert train_batch.batch[K.REF_LOG_PROBS].shape == train_batch.batch[K.RESPONSES].shape
        assert train_batch.meta_info["global_steps"] == 7

        return DataProto.from_single_dict(
            data={},
            meta_info={"metrics": {"actor/qrpo_loss": 0.123}},
        )

    trainer._compute_ref_log_prob = MethodType(fake_compute_ref_log_prob, trainer)
    trainer._update_actor = MethodType(fake_update_actor, trainer)

    out = trainer.qrpo_update_step(
        [offline, online],
        pad_token_id=9,
        meta_info={"global_steps": 7},
    )

    assert calls == {
        "compute_ref_log_prob": 1,
        "update_actor": 1,
    }
    assert trainer.actor_rollout_wg.to_calls == [
        {
            "device": "device",
            "model": True,
            "optimizer": False,
            "grad": False,
        }
    ]
    assert out.meta_info["metrics"]["actor/qrpo_loss"] == 0.123
    assert out.meta_info["metrics"]["qrpo/all/trajectory_reward_mean"] == pytest.approx(2.0)
    assert out.meta_info["metrics"]["qrpo/all/ref_quantile_mean"] == pytest.approx(0.5)
    assert out.meta_info["metrics"]["qrpo/all/ref_quantile_at_zero_frac"] == pytest.approx(0.5)
    assert out.meta_info["metrics"]["qrpo/all/ref_quantile_at_one_frac"] == pytest.approx(0.5)
    assert out.meta_info["metrics"]["qrpo/offline/ref_quantile_mean"] == pytest.approx(0.5)
    assert out.meta_info["metrics"]["qrpo/online/ref_quantile_mean"] == pytest.approx(0.5)


def test_qrpo_update_step_uses_tokenizer_pad_token_id_for_single_batch() -> None:
    trainer = make_trainer_for_unit_test()
    batch = make_qrpo_batch()

    trainer._compute_ref_log_prob = MethodType(
        lambda self, train_batch: DataProto.from_dict(
            tensors={
                K.REF_LOG_PROBS: torch.zeros_like(
                    train_batch.batch[K.RESPONSES],
                    dtype=torch.float32,
                )
            }
        ),
        trainer,
    )
    trainer._update_actor = MethodType(
        lambda self, train_batch: DataProto.from_single_dict(
            data={},
            meta_info={"metrics": {"actor/qrpo_loss": 0.1}},
        ),
        trainer,
    )

    out = trainer.qrpo_update_step([batch])

    assert out.meta_info["metrics"]["actor/qrpo_loss"] == 0.1


def test_qrpo_update_step_requires_non_empty_batches() -> None:
    trainer = make_trainer_for_unit_test()

    with pytest.raises(ValueError, match="At least one"):
        trainer.qrpo_update_step([])


def test_qrpo_update_step_requires_pad_token_id_when_no_tokenizer() -> None:
    trainer = make_trainer_for_unit_test()
    trainer.tokenizer = None

    with pytest.raises(ValueError, match="pad_token_id"):
        trainer.qrpo_update_step([make_qrpo_batch()])


def test_qrpo_update_step_requires_pad_token_id_when_tokenizer_has_none() -> None:
    trainer = make_trainer_for_unit_test()
    trainer.tokenizer = NoPadTokenizer()

    with pytest.raises(ValueError, match="pad_token_id"):
        trainer.qrpo_update_step([make_qrpo_batch()])


def test_qrpo_update_step_rejects_bad_ref_log_prob_shape() -> None:
    trainer = make_trainer_for_unit_test()
    batch = make_qrpo_batch()

    def fake_compute_ref_log_prob(self, train_batch):
        return DataProto.from_dict(
            tensors={
                K.REF_LOG_PROBS: torch.zeros(2, 2, dtype=torch.float32),
            }
        )

    trainer._compute_ref_log_prob = MethodType(fake_compute_ref_log_prob, trainer)

    with pytest.raises(ValueError, match=K.REF_LOG_PROBS):
        trainer.qrpo_update_step([batch])


def test_qrpo_update_step_requires_training_schema() -> None:
    trainer = make_trainer_for_unit_test()
    batch = make_qrpo_batch()
    del batch.batch[K.RESPONSE_MASK]

    with pytest.raises(KeyError, match=K.RESPONSE_MASK):
        trainer.qrpo_update_step([batch])


def test_qrpo_update_step_rejects_ref_version_mismatch() -> None:
    trainer = make_trainer_for_unit_test()
    batch = make_qrpo_batch()
    batch.meta_info[K.REF_VERSION] = "old_ref"

    with pytest.raises(ValueError, match="ref versions"):
        trainer.qrpo_update_step(
            [batch],
            meta_info={K.REF_VERSION: "new_ref"},
        )


def test_qrpo_update_step_rejects_missing_expected_ref_version() -> None:
    trainer = make_trainer_for_unit_test()
    batch = make_qrpo_batch()

    with pytest.raises(ValueError, match="missing"):
        trainer.qrpo_update_step(
            [batch],
            meta_info={K.REF_VERSION: "new_ref"},
        )


def test_qrpo_update_step_rejects_partially_versioned_batches() -> None:
    trainer = make_trainer_for_unit_test()
    unversioned = make_qrpo_batch(prefix=0)
    versioned = make_qrpo_batch(prefix=1000)
    versioned.meta_info[K.REF_VERSION] = "ref_v1"

    with pytest.raises(ValueError, match="all define"):
        trainer.qrpo_update_step([unversioned, versioned])


def test_run_qrpo_iteration_builds_mixed_batch_and_updates(monkeypatch) -> None:
    from trainers import qrpo_trainer as qrpo_trainer_module

    trainer = make_trainer_for_unit_test()
    trainer.global_steps = 12

    offline_candidate = object()
    online_request = object()

    offline_train_batch = make_qrpo_batch(prefix=0, prompt_len=2, response_len=3)
    online_rollout_input = _rollout_input_for_training_metadata()
    rollout_output = _agent_loop_rollout_output_without_train_metadata()
    rollout_output.batch["rm_scores"] = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    rollout_output.meta_info["kind"] = "rollout_output"
    online_train_batch = make_qrpo_batch(prefix=1000, prompt_len=4, response_len=5)

    calls = {
        "build_candidates": 0,
        "offline_to_dataproto": 0,
        "online_requests_to_dataproto": 0,
        "generate_sequences": 0,
        "online_rewards": 0,
        "online_output_to_train": 0,
        "qrpo_update": 0,
    }

    def fake_build_training_candidates(
        *,
        prompt_records,
        source_counts,
        offline_selector,
        actor_version,
    ):
        calls["build_candidates"] += 1
        assert actor_version == 12
        return [offline_candidate], [online_request]

    def fake_offline_candidates_to_dataproto(
        *,
        candidates,
        tokenizer,
        config,
        meta_info,
    ):
        calls["offline_to_dataproto"] += 1
        assert candidates == [offline_candidate]
        assert tokenizer is trainer.tokenizer
        assert config == {
            "require_assistant_mask": False,
            K.TEMPERATURE: 0.7,
        }
        assert meta_info == {"global_steps": 12}
        return offline_train_batch

    def fake_online_rollout_requests_to_dataproto(
        *,
        requests,
        config,
        meta_info,
    ):
        calls["online_requests_to_dataproto"] += 1
        assert requests == [online_request]
        assert config == {
            "data_source": "activeultrafeedback",
            "agent_name": "tool_agent",
        }
        assert meta_info == {"global_steps": 12}
        return online_rollout_input

    class FakeAsyncRolloutManager:
        def generate_sequences(self, batch):
            calls["generate_sequences"] += 1
            assert batch is online_rollout_input
            return rollout_output

    def fake_compute_online_rewards_from_rollout_output(self, batch):
        calls["online_rewards"] += 1
        assert batch is rollout_output
        return (
            torch.tensor([1.0, 2.0], dtype=torch.float32),
            {
                "reward/online_mean": 1.5,
                "reward/online_min": 1.0,
                "reward/online_max": 2.0,
            },
        )

    def fake_online_rollout_output_to_train_dataproto(
        *,
        rollout_output,
        rewards,
        temperature,
        meta_info,
    ):
        calls["online_output_to_train"] += 1
        assert rollout_output.meta_info["kind"] == "rollout_output"
        assert torch.equal(rewards, torch.tensor([1.0, 2.0]))
        assert temperature == pytest.approx(0.7)
        assert meta_info == {"global_steps": 12}
        return online_train_batch

    def fake_qrpo_update_step(self, batches, *, pad_token_id=None, meta_info=None):
        calls["qrpo_update"] += 1
        assert batches == [offline_train_batch, online_train_batch]
        assert pad_token_id is None
        assert meta_info == {"global_steps": 12}

        return DataProto.from_single_dict(
            data={},
            meta_info={"metrics": {"actor/qrpo_loss": 0.25}},
        )

    monkeypatch.setattr(
        qrpo_trainer_module,
        "build_training_candidates",
        fake_build_training_candidates,
    )
    monkeypatch.setattr(
        qrpo_trainer_module,
        "offline_candidates_to_dataproto",
        fake_offline_candidates_to_dataproto,
    )
    monkeypatch.setattr(
        qrpo_trainer_module,
        "online_rollout_requests_to_dataproto",
        fake_online_rollout_requests_to_dataproto,
    )
    monkeypatch.setattr(
        qrpo_trainer_module,
        "online_rollout_output_to_train_dataproto",
        fake_online_rollout_output_to_train_dataproto,
    )

    trainer.async_rollout_manager = FakeAsyncRolloutManager()
    trainer._compute_online_rewards_from_rollout_output = MethodType(
        fake_compute_online_rewards_from_rollout_output,
        trainer,
    )
    trainer.qrpo_update_step = MethodType(fake_qrpo_update_step, trainer)

    out = trainer.run_qrpo_iteration(
        prompt_records=[object()],
        source_counts=[object()],
        offline_selector=lambda prompt, n: [],
        offline_tokenization_config={"require_assistant_mask": False},
        online_rollout_config={
            "data_source": "activeultrafeedback",
            "agent_name": "tool_agent",
        },
        meta_info={"global_steps": 12},
    )

    assert calls == {
        "build_candidates": 1,
        "offline_to_dataproto": 1,
        "online_requests_to_dataproto": 1,
        "generate_sequences": 1,
        "online_rewards": 1,
        "online_output_to_train": 1,
        "qrpo_update": 1,
    }

    assert out.meta_info["metrics"]["actor/qrpo_loss"] == 0.25
    assert out.meta_info["metrics"]["reward/online_mean"] == pytest.approx(1.5)


def test_run_qrpo_iteration_supports_offline_only(monkeypatch) -> None:
    from trainers import qrpo_trainer as qrpo_trainer_module

    trainer = make_trainer_for_unit_test()

    offline_train_batch = make_qrpo_batch(prefix=0, prompt_len=2, response_len=3)

    calls = {
        "offline_to_dataproto": 0,
        "qrpo_update": 0,
    }

    monkeypatch.setattr(
        qrpo_trainer_module,
        "build_training_candidates",
        lambda **kwargs: ([object()], []),
    )

    def fake_offline_candidates_to_dataproto(
        *,
        candidates,
        tokenizer,
        config,
        meta_info,
    ):
        calls["offline_to_dataproto"] += 1
        return offline_train_batch

    def fake_qrpo_update_step(self, batches, *, pad_token_id=None, meta_info=None):
        calls["qrpo_update"] += 1
        assert batches == [offline_train_batch]
        return DataProto.from_single_dict(
            data={},
            meta_info={"metrics": {"actor/qrpo_loss": 0.1}},
        )

    monkeypatch.setattr(
        qrpo_trainer_module,
        "offline_candidates_to_dataproto",
        fake_offline_candidates_to_dataproto,
    )

    trainer.qrpo_update_step = MethodType(fake_qrpo_update_step, trainer)

    out = trainer.run_qrpo_iteration(
        prompt_records=[object()],
        source_counts=[object()],
        offline_selector=lambda prompt, n: [],
    )

    assert calls == {
        "offline_to_dataproto": 1,
        "qrpo_update": 1,
    }
    assert out.meta_info["metrics"]["actor/qrpo_loss"] == 0.1


def test_run_qrpo_iteration_supports_online_only(monkeypatch) -> None:
    from trainers import qrpo_trainer as qrpo_trainer_module

    trainer = make_trainer_for_unit_test()

    online_rollout_input = _rollout_input_for_training_metadata()
    rollout_output = _agent_loop_rollout_output_without_train_metadata()
    rollout_output.batch["rm_scores"] = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    rollout_output.meta_info["kind"] = "rollout_output"
    online_train_batch = make_qrpo_batch(prefix=1000, prompt_len=4, response_len=5)

    calls = {
        "online_requests_to_dataproto": 0,
        "generate_sequences": 0,
        "online_rewards": 0,
        "online_output_to_train": 0,
        "qrpo_update": 0,
    }

    monkeypatch.setattr(
        qrpo_trainer_module,
        "build_training_candidates",
        lambda **kwargs: ([], [object()]),
    )

    def fake_online_rollout_requests_to_dataproto(
        *,
        requests,
        config,
        meta_info,
    ):
        calls["online_requests_to_dataproto"] += 1
        assert config == {
            "data_source": "activeultrafeedback",
            "agent_name": "tool_agent",
        }
        return online_rollout_input

    class FakeAsyncRolloutManager:
        def generate_sequences(self, batch):
            calls["generate_sequences"] += 1
            assert batch is online_rollout_input
            return rollout_output

    def fake_compute_online_rewards_from_rollout_output(self, batch):
        calls["online_rewards"] += 1
        assert batch is rollout_output
        return (
            torch.tensor([1.0, 2.0], dtype=torch.float32),
            {
                "reward/online_mean": 1.5,
                "reward/online_min": 1.0,
                "reward/online_max": 2.0,
            },
        )

    def fake_online_rollout_output_to_train_dataproto(
        *,
        rollout_output,
        rewards,
        temperature,
        meta_info,
    ):
        calls["online_output_to_train"] += 1
        assert rollout_output.meta_info["kind"] == "rollout_output"
        assert torch.equal(rewards, torch.tensor([1.0, 2.0]))
        assert temperature == pytest.approx(0.7)
        return online_train_batch

    def fake_qrpo_update_step(self, batches, *, pad_token_id=None, meta_info=None):
        calls["qrpo_update"] += 1
        assert batches == [online_train_batch]
        return DataProto.from_single_dict(
            data={},
            meta_info={"metrics": {"actor/qrpo_loss": 0.2}},
        )

    monkeypatch.setattr(
        qrpo_trainer_module,
        "online_rollout_requests_to_dataproto",
        fake_online_rollout_requests_to_dataproto,
    )
    monkeypatch.setattr(
        qrpo_trainer_module,
        "online_rollout_output_to_train_dataproto",
        fake_online_rollout_output_to_train_dataproto,
    )

    trainer.async_rollout_manager = FakeAsyncRolloutManager()
    trainer._compute_online_rewards_from_rollout_output = MethodType(
        fake_compute_online_rewards_from_rollout_output,
        trainer,
    )
    trainer.qrpo_update_step = MethodType(fake_qrpo_update_step, trainer)

    out = trainer.run_qrpo_iteration(
        prompt_records=[object()],
        source_counts=[object()],
        offline_selector=lambda prompt, n: [],
        online_rollout_config={
            "data_source": "activeultrafeedback",
            "agent_name": "tool_agent",
        },
    )

    assert calls == {
        "online_requests_to_dataproto": 1,
        "generate_sequences": 1,
        "online_rewards": 1,
        "online_output_to_train": 1,
        "qrpo_update": 1,
    }
    assert out.meta_info["metrics"]["actor/qrpo_loss"] == 0.2
    assert out.meta_info["metrics"]["reward/online_mean"] == pytest.approx(1.5)

def test_run_qrpo_iteration_requires_online_rollout_config_when_online_exists(monkeypatch) -> None:
    from trainers import qrpo_trainer as qrpo_trainer_module

    trainer = make_trainer_for_unit_test()

    monkeypatch.setattr(
        qrpo_trainer_module,
        "build_training_candidates",
        lambda **kwargs: ([], [object()]),
    )

    with pytest.raises(ValueError, match="online_rollout_config"):
        trainer.run_qrpo_iteration(
            prompt_records=[object()],
            source_counts=[object()],
            offline_selector=lambda prompt, n: [],
        )


def test_run_qrpo_iteration_requires_agent_loop_rm_scores_when_online_exists(monkeypatch) -> None:
    from trainers import qrpo_trainer as qrpo_trainer_module

    trainer = make_trainer_for_unit_test()

    online_rollout_input = _rollout_input_for_training_metadata()
    rollout_output = _agent_loop_rollout_output_without_train_metadata()

    monkeypatch.setattr(
        qrpo_trainer_module,
        "build_training_candidates",
        lambda **kwargs: ([], [object()]),
    )

    monkeypatch.setattr(
        qrpo_trainer_module,
        "online_rollout_requests_to_dataproto",
        lambda **kwargs: online_rollout_input,
    )

    class FakeAsyncRolloutManager:
        def generate_sequences(self, batch):
            return rollout_output

    trainer.async_rollout_manager = FakeAsyncRolloutManager()

    with pytest.raises(KeyError, match="rm_scores"):
        trainer.run_qrpo_iteration(
            prompt_records=[object()],
            source_counts=[object()],
            offline_selector=lambda prompt, n: [],
            online_rollout_config={
                "data_source": "activeultrafeedback",
                "agent_name": "tool_agent",
            },
        )


def test_run_qrpo_iteration_requires_tokenizer_when_offline_exists(monkeypatch) -> None:
    from trainers import qrpo_trainer as qrpo_trainer_module

    trainer = make_trainer_for_unit_test()
    trainer.tokenizer = None

    monkeypatch.setattr(
        qrpo_trainer_module,
        "build_training_candidates",
        lambda **kwargs: ([object()], []),
    )

    with pytest.raises(ValueError, match="tokenizer"):
        trainer.run_qrpo_iteration(
            prompt_records=[object()],
            source_counts=[object()],
            offline_selector=lambda prompt, n: [],
        )


def test_fit_runs_qrpo_iterations_and_logs() -> None:
    cfg = make_config(
        **{
            "trainer.total_epochs": 1,
            "trainer.val_before_train": False,
            "trainer.val_only": False,
            "trainer.save_freq": 0,
            "trainer.test_freq": 0,
        }
    )

    trainer = make_trainer_for_unit_test(config=cfg)
    trainer.total_training_steps = 2
    trainer.train_dataloader = [
        [make_fit_prompt("p0")],
        [make_fit_prompt("p1")],
    ]
    trainer.checkpoint_manager = FakeCheckpointManager()
    trainer.actor_rollout_wg = FakeActorRolloutWG()

    source_counts = [
        SourceCounts(prompt_index=0, prompt_id="p0", n_online=0, n_offline=1),
    ]
    trainer.source_scheduler = FakeSourceScheduler(source_counts)
    trainer.offline_selector = lambda prompt, n: []
    trainer.online_reward_fn = None

    fake_logger = FakeLogger()
    trainer._make_tracking_logger = lambda: fake_logger
    trainer._load_checkpoint = lambda: None

    calls = []

    def fake_run_qrpo_iteration(
        *,
        prompt_records,
        source_counts,
        actor_version,
        meta_info,
        **kwargs,
    ):
        calls.append(
            {
                "prompt_id": prompt_records[0].prompt_id,
                "source_counts": source_counts,
                "actor_version": actor_version,
                "meta_info": meta_info,
            }
        )
        return DataProto.from_single_dict(
            data={},
            meta_info={
                "metrics": {
                    "actor/qrpo_loss": 0.1 * actor_version,
                }
            },
        )

    trainer.run_qrpo_iteration = fake_run_qrpo_iteration

    trainer.fit()

    assert trainer.global_steps == 2
    assert trainer.checkpoint_manager.updated_steps == []

    assert [call["prompt_id"] for call in calls] == ["p0", "p1"]
    assert [call["actor_version"] for call in calls] == [1, 2]
    assert [call["meta_info"]["global_steps"] for call in calls] == [1, 2]

    assert len(fake_logger.calls) == 2
    assert fake_logger.calls[0][0]["training/global_step"] == 1
    assert fake_logger.calls[1][0]["training/global_step"] == 2


def test_fit_resolves_ref_rewards_from_store_before_source_planning() -> None:
    cfg = make_config(
        **{
            "trainer.total_epochs": 1,
            "trainer.val_before_train": False,
            "trainer.save_freq": 0,
            "trainer.test_freq": 0,
        }
    )

    trainer = make_trainer_for_unit_test(config=cfg)
    trainer.total_training_steps = 1
    trainer.train_dataloader = [[make_fit_prompt("p0")]]
    trainer.checkpoint_manager = FakeCheckpointManager()
    trainer.actor_rollout_wg = FakeActorRolloutWG()
    trainer.offline_selector = lambda prompt, n: []
    trainer.ref_reward_store = FakeRefRewardStore({"p0": (9.0, 10.0)})
    trainer.current_ref_version = "ref_v1"

    class AssertingSourceScheduler:
        def plan(self, prompts):
            assert prompts[0].ref_rewards == (9.0, 10.0)
            return [
                SourceCounts(
                    prompt_index=0,
                    prompt_id="p0",
                    n_online=0,
                    n_offline=1,
                )
            ]

    trainer.source_scheduler = AssertingSourceScheduler()
    trainer._make_tracking_logger = lambda: FakeLogger()
    trainer._load_checkpoint = lambda: None
    trainer.run_qrpo_iteration = lambda **kwargs: DataProto.from_single_dict(
        data={},
        meta_info={"metrics": {}},
    )

    trainer.fit()

    assert trainer.ref_reward_store.calls == [
        {
            "prompt_ids": ["p0"],
            "ref_version": "ref_v1",
        }
    ]


def test_prepare_initial_ref_rewards_generates_and_saves_full_dataset() -> None:
    cfg = make_config(
        **{
            "data.ref_rewards_key": None,
            "data.num_ref_rewards": 2,
            "ref_rewards.initial_source": "generate",
            "ref_rewards.num_ref_completions": 2,
            "ref_rewards.generation_prompt_batch_size": 1,
            "ref_rewards.reuse_existing": True,
            "online_rollout.data_source": "activeultrafeedback",
            "online_rollout.agent_name": None,
            "online_rollout.validate": False,
        }
    )
    trainer = make_trainer_for_unit_test(config=cfg)
    trainer.current_ref_version = "ref_step_000000"
    trainer.ref_reward_store = FakeRefRewardStore()
    trainer.checkpoint_manager = FakeCheckpointManager()
    trainer.global_steps = 0
    trainer._qrpo_train_dataset = [
        {
            "prompt_id": "p0",
            "prompt_messages": [{"role": "user", "content": "prompt 0"}],
            "offline_trajectories": [],
            "offline_rewards": [],
        },
        {
            "prompt_id": "p1",
            "prompt_messages": [{"role": "user", "content": "prompt 1"}],
            "offline_trajectories": [],
            "offline_rewards": [],
        },
    ]

    class FakeRolloutManager:
        def __init__(self):
            self.inputs = []

        def generate_sequences(self, rollout_input):
            self.inputs.append(rollout_input)
            size = len(rollout_input)
            prompts = torch.ones(size, 2, dtype=torch.long)
            responses = torch.arange(10, 10 + size * 2, dtype=torch.long).reshape(
                size,
                2,
            )
            input_ids = torch.cat([prompts, responses], dim=-1)
            attention_mask = torch.ones_like(input_ids)
            return DataProto.from_dict(
                tensors={
                    K.PROMPTS: prompts,
                    K.RESPONSES: responses,
                    K.INPUT_IDS: input_ids,
                    K.ATTENTION_MASK: attention_mask,
                    "rm_scores": torch.ones(size, 2, dtype=torch.float32),
                },
                non_tensors={},
                meta_info={},
            )

    trainer.async_rollout_manager = FakeRolloutManager()

    trainer.prepare_initial_ref_rewards()

    assert trainer.checkpoint_manager.updated_steps == [0]
    assert trainer.checkpoint_manager.sleep_calls == 1
    assert len(trainer.async_rollout_manager.inputs) == 2

    saved = trainer.ref_reward_store.saved_versions[0]
    assert saved["ref_version"] == "ref_step_000000"
    assert saved["manifest"]["source"] == "generate"
    assert saved["manifest"]["prompt_count"] == 2
    assert saved["manifest"]["num_ref_completions"] == 2
    assert [row["prompt_id"] for row in saved["rows"]] == ["p0", "p1"]
    assert [row[K.DATASET_INDEX] for row in saved["rows"]] == [0, 1]
    assert saved["rows"][0]["ref_rewards"] == [2.0, 2.0]
    assert [
        chunk["dataset_indices"]
        for chunk in trainer.ref_reward_store.saved_chunks
    ] == [[0], [1]]


def test_prepare_initial_ref_rewards_reuses_saved_generation_chunk() -> None:
    cfg = make_config(
        **{
            "data.ref_rewards_key": None,
            "data.num_ref_rewards": 2,
            "ref_rewards.initial_source": "generate",
            "ref_rewards.num_ref_completions": 2,
            "ref_rewards.generation_prompt_batch_size": 1,
            "ref_rewards.reuse_existing": True,
            "online_rollout.data_source": "activeultrafeedback",
            "online_rollout.agent_name": None,
            "online_rollout.validate": False,
        }
    )
    trainer = make_trainer_for_unit_test(config=cfg)
    trainer.current_ref_version = "ref_step_000000"
    trainer.ref_reward_store = FakeRefRewardStore()
    trainer.checkpoint_manager = FakeCheckpointManager()
    trainer.global_steps = 0
    trainer._qrpo_train_dataset = [
        {
            "prompt_id": "p0",
            "prompt_messages": [{"role": "user", "content": "prompt 0"}],
            "offline_trajectories": [],
            "offline_rewards": [],
        },
        {
            "prompt_id": "p1",
            "prompt_messages": [{"role": "user", "content": "prompt 1"}],
            "offline_trajectories": [],
            "offline_rewards": [],
        },
    ]

    manifest = trainer._initial_ref_generation_manifest(
        train_dataset=trainer._qrpo_train_dataset,
        online_rollout_config={
            "data_source": "activeultrafeedback",
            "agent_name": None,
            "validate": False,
        },
        ref_config={
            "initial_source": "generate",
            "num_ref_completions": 2,
            "generation_prompt_batch_size": 1,
            "reuse_existing": True,
        },
        num_ref_completions=2,
        actor_step=0,
    )
    trainer.ref_reward_store.save_generation_chunk(
        ref_version="ref_step_000000",
        chunk_index=0,
        dataset_indices=[0],
        rows=[
            {
                "prompt_id": "p0",
                "dataset_index": 0,
                "ref_version": "ref_step_000000",
                "ref_completions": ["cached 0", "cached 1"],
                "ref_rewards": [1.0, 2.0],
                "reward_extra_info": [None, None],
            }
        ],
        manifest=manifest,
    )
    trainer.ref_reward_store.saved_chunks.clear()

    class FakeRolloutManager:
        def __init__(self):
            self.inputs = []

        def generate_sequences(self, rollout_input):
            self.inputs.append(rollout_input)
            size = len(rollout_input)
            prompts = torch.ones(size, 2, dtype=torch.long)
            responses = torch.arange(10, 10 + size * 2, dtype=torch.long).reshape(
                size,
                2,
            )
            input_ids = torch.cat([prompts, responses], dim=-1)
            attention_mask = torch.ones_like(input_ids)
            return DataProto.from_dict(
                tensors={
                    K.PROMPTS: prompts,
                    K.RESPONSES: responses,
                    K.INPUT_IDS: input_ids,
                    K.ATTENTION_MASK: attention_mask,
                    "rm_scores": torch.ones(size, 2, dtype=torch.float32),
                },
                non_tensors={},
                meta_info={},
            )

    trainer.async_rollout_manager = FakeRolloutManager()

    trainer.prepare_initial_ref_rewards()

    assert len(trainer.async_rollout_manager.inputs) == 1
    assert trainer.ref_reward_store.saved_chunks[0]["dataset_indices"] == [1]
    saved = trainer.ref_reward_store.saved_versions[0]
    assert [row["prompt_id"] for row in saved["rows"]] == ["p0", "p1"]
    assert saved["rows"][0]["ref_rewards"] == [1.0, 2.0]


def test_ref_generation_num_chunks_overrides_prompt_batch_size() -> None:
    batch_size = QRPOTrainer._resolve_ref_generation_prompt_batch_size(
        ref_config={
            "generation_num_chunks": 3,
            "generation_prompt_batch_size": 100,
        },
        dataset_size=10,
    )

    assert batch_size == 4


def test_prepare_initial_ref_rewards_reuses_complete_version(
    monkeypatch,
    tmp_path,
) -> None:
    reward_path = tmp_path / "reward.py"
    reward_path.write_text("async def compute_score(): return 1.0\n")
    prompts_path = tmp_path / "prompts.py"
    prompts_path.write_text("HELPFULNESS_ANNOTATION_PROMPT = 'prompt'\n")

    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("ACTIVE_UF_ASPECTS", "helpfulness")
    monkeypatch.setenv("ACTIVE_UF_PROMPTS_PATH", str(prompts_path))

    cfg = make_config(
        **{
            "actor_rollout_ref.model.path": "/tmp/actor",
            "data.max_prompt_length": 128,
            "data.max_response_length": 64,
            "ref_rewards.initial_source": "generate",
            "ref_rewards.num_ref_completions": 2,
            "ref_rewards.reuse_existing": True,
            "ref_rewards.manifest_env_keys": [
                "JUDGE_MODEL",
                "ACTIVE_UF_ASPECTS",
                "ACTIVE_UF_PROMPTS_PATH",
            ],
            "online_rollout.data_source": "activeultrafeedback",
            "reward.custom_reward_function.path": str(reward_path),
            "reward.custom_reward_function.name": "compute_score",
        }
    )
    trainer = make_trainer_for_unit_test(config=cfg)
    trainer.current_ref_version = "ref_step_000000"
    trainer.ref_reward_store.complete_versions.add("ref_step_000000")
    trainer.checkpoint_manager = FakeCheckpointManager()
    trainer._qrpo_train_dataset = [
        {
            "prompt_id": "p0",
            "prompt_messages": [{"role": "user", "content": "prompt 0"}],
            "offline_trajectories": [],
            "offline_rewards": [],
        },
    ]

    trainer.prepare_initial_ref_rewards()

    assert trainer.checkpoint_manager.updated_steps == []
    assert trainer.ref_reward_store.saved_versions == []
    assert trainer.ref_reward_store.calls == [
        {
            "ref_version": "ref_step_000000",
            "metadata": {
                "source": "generate",
                "actor_step": 0,
                "prompt_count": 1,
                "num_ref_completions": 2,
                "data_path": None,
                "dataset_fingerprint": None,
                "data_source": "activeultrafeedback",
                "actor_model_path": "/tmp/actor",
                "max_prompt_length": 128,
                "max_response_length": 64,
                "rollout_temperature": 0.7,
                "rollout_top_k": None,
                "rollout_top_p": None,
                "rollout_do_sample": None,
                "reward_function_path": str(reward_path),
                "reward_function_sha256": hashlib.sha256(
                    reward_path.read_bytes()
                ).hexdigest(),
                "reward_function_name": "compute_score",
                "env/JUDGE_MODEL": "judge-model",
                "env/JUDGE_MODEL_sha256": None,
                "env/ACTIVE_UF_ASPECTS": "helpfulness",
                "env/ACTIVE_UF_ASPECTS_sha256": None,
                "env/ACTIVE_UF_PROMPTS_PATH": str(prompts_path),
                "env/ACTIVE_UF_PROMPTS_PATH_sha256": hashlib.sha256(
                    prompts_path.read_bytes()
                ).hexdigest(),
            },
        }
    ]


def test_prepare_initial_ref_rewards_rejects_missing_store_after_step_zero() -> None:
    cfg = make_config(
        **{
            "ref_rewards.initial_source": "generate",
            "ref_rewards.num_ref_completions": 2,
            "online_rollout.data_source": "activeultrafeedback",
        }
    )
    trainer = make_trainer_for_unit_test(config=cfg)
    trainer.current_ref_version = "ref_step_000000"
    trainer.global_steps = 3
    trainer._qrpo_train_dataset = [
        {
            "prompt_id": "p0",
            "prompt_messages": [{"role": "user", "content": "prompt 0"}],
            "offline_trajectories": [],
            "offline_rewards": [],
        },
    ]

    with pytest.raises(ValueError, match="after training has advanced"):
        trainer.prepare_initial_ref_rewards()


def test_prepare_initial_ref_rewards_checks_complete_generated_version_after_step_zero() -> None:
    cfg = make_config(
        **{
            "ref_rewards.initial_source": "generate",
            "ref_rewards.num_ref_completions": 2,
            "online_rollout.data_source": "activeultrafeedback",
        }
    )
    trainer = make_trainer_for_unit_test(config=cfg)
    trainer.current_ref_version = "ref_step_000000"
    trainer.ref_reward_store.complete_versions.add("ref_step_000000")
    trainer.global_steps = 3
    trainer._qrpo_train_dataset = [
        {
            "prompt_id": "p0",
            "prompt_messages": [{"role": "user", "content": "prompt 0"}],
            "offline_trajectories": [],
            "offline_rewards": [],
        },
    ]

    trainer.prepare_initial_ref_rewards()

    assert trainer.ref_reward_store.calls == [
        {
            "ref_version": "ref_step_000000",
            "metadata": {
                "source": "generate",
                "actor_step": 0,
                "prompt_count": 1,
                "num_ref_completions": 2,
                "data_path": None,
                "dataset_fingerprint": None,
                "data_source": "activeultrafeedback",
                "actor_model_path": None,
                "max_prompt_length": None,
                "max_response_length": None,
                "rollout_temperature": 0.7,
                "rollout_top_k": None,
                "rollout_top_p": None,
                "rollout_do_sample": None,
                "reward_function_path": None,
                "reward_function_sha256": None,
                "reward_function_name": None,
            },
        }
    ]


def test_prepare_initial_ref_rewards_rejects_refreshed_store_without_model_checkpoint() -> None:
    cfg = make_config(
        **{
            "ref_rewards.initial_source": "store",
        }
    )
    trainer = make_trainer_for_unit_test(config=cfg)
    trainer.current_ref_version = "ref_step_000017"
    trainer.ref_reward_store.manifests["ref_step_000017"] = {"source": "refresh"}

    with pytest.raises(FileNotFoundError, match="ref model checkpoint is missing"):
        trainer.prepare_initial_ref_rewards()


def test_refresh_ref_rewards_generates_rows_and_syncs_ref_model() -> None:
    cfg = make_config(
        **{
            "ref_rewards.num_ref_completions": 2,
            "ref_rewards.generation_prompt_batch_size": 1,
            "ref_rewards.reuse_existing": True,
            "online_rollout.data_source": "activeultrafeedback",
            "online_rollout.agent_name": None,
            "online_rollout.validate": False,
        }
    )
    trainer = make_trainer_for_unit_test(config=cfg)
    trainer.current_ref_version = "old_ref"
    trainer.ref_reward_store = FakeRefRewardStore()
    trainer.checkpoint_manager = FakeCheckpointManager()
    trainer.actor_rollout_wg = FakeActorRolloutWG()
    trainer._qrpo_train_dataset = [
        {
            "prompt_id": "p0",
            "prompt_messages": [{"role": "user", "content": "prompt 0"}],
            "offline_trajectories": [],
            "offline_rewards": [],
        },
    ]

    class FakeRolloutManager:
        def generate_sequences(self, rollout_input):
            size = len(rollout_input)
            prompts = torch.ones(size, 2, dtype=torch.long)
            responses = torch.arange(10, 10 + size * 2, dtype=torch.long).reshape(
                size,
                2,
            )
            input_ids = torch.cat([prompts, responses], dim=-1)
            attention_mask = torch.ones_like(input_ids)
            return DataProto.from_dict(
                tensors={
                    K.PROMPTS: prompts,
                    K.RESPONSES: responses,
                    K.INPUT_IDS: input_ids,
                    K.ATTENTION_MASK: attention_mask,
                    "rm_scores": torch.ones(size, 2, dtype=torch.float32),
                },
                non_tensors={},
                meta_info={},
            )

    trainer.async_rollout_manager = FakeRolloutManager()

    ref_version = trainer.refresh_ref_rewards(actor_step=17)

    assert ref_version == "ref_step_000017"
    assert trainer.current_ref_version == "ref_step_000017"
    assert trainer.checkpoint_manager.updated_steps == [17]
    assert trainer.checkpoint_manager.sleep_calls == 1
    assert trainer.actor_rollout_wg.sync_ref_calls == [
        {
            "local_path": "/tmp/ref_reward_store/model_checkpoints/ref_step_000017",
            "global_step": 17,
        }
    ]

    saved = trainer.ref_reward_store.saved_versions[0]
    assert saved["ref_version"] == "ref_step_000017"
    assert saved["manifest"]["source"] == "refresh"
    assert saved["manifest"]["actor_step"] == 17
    assert saved["manifest"]["ref_model_checkpoint_path"] == (
        "/tmp/ref_reward_store/model_checkpoints/ref_step_000017"
    )
    assert saved["rows"][0]["prompt_id"] == "p0"
    assert saved["rows"][0]["ref_rewards"] == [2.0, 2.0]


def test_generated_ref_row_coverage_rejects_duplicate_dataset_index() -> None:
    with pytest.raises(ValueError, match="duplicate dataset indices"):
        QRPOTrainer._check_generated_ref_rows_cover_dataset(
            rows=[
                {K.DATASET_INDEX: 0},
                {K.DATASET_INDEX: 0},
            ],
            dataset_size=2,
        )


def test_sync_ref_model_from_actor_uses_store_checkpoint_path() -> None:
    trainer = make_trainer_for_unit_test()
    trainer.actor_rollout_wg = FakeActorRolloutWG()

    path = trainer._sync_ref_model_from_actor(
        ref_version="ref_step_000017",
        global_step=17,
    )

    assert path == "/tmp/ref_reward_store/model_checkpoints/ref_step_000017"
    assert trainer.actor_rollout_wg.sync_ref_calls == [
        {
            "local_path": path,
            "global_step": 17,
        }
    ]


def test_load_ref_model_checkpoint_uses_store_checkpoint_path() -> None:
    trainer = make_trainer_for_unit_test()
    trainer.actor_rollout_wg = FakeActorRolloutWG()

    path = trainer._load_ref_model_checkpoint(ref_version="ref_step_000017")

    assert path == "/tmp/ref_reward_store/model_checkpoints/ref_step_000017"
    assert trainer.actor_rollout_wg.load_ref_calls == [
        {
            "local_path": path,
            "del_local_after_load": False,
        }
    ]


def test_qrpo_checkpoint_state_roundtrip(tmp_path) -> None:
    cfg = make_config(**{"trainer.default_local_dir": str(tmp_path)})
    trainer = make_trainer_for_unit_test(config=cfg)
    trainer.global_steps = 17
    trainer.current_ref_version = "ref_step_000017"

    trainer._save_qrpo_checkpoint_state()

    state_path = tmp_path / "global_step_17" / "qrpo_state.json"
    assert json.loads(state_path.read_text()) == {
        K.REF_VERSION: "ref_step_000017",
    }

    resumed = make_trainer_for_unit_test(config=cfg)
    resumed.global_steps = 17
    resumed.current_ref_version = "old_ref"
    resumed._load_qrpo_checkpoint_state()

    assert resumed.current_ref_version == "ref_step_000017"


def test_save_checkpoint_writes_qrpo_state_before_verl_tracker(monkeypatch, tmp_path) -> None:
    from trainers import qrpo_trainer as qrpo_trainer_module

    cfg = make_config(**{"trainer.default_local_dir": str(tmp_path)})
    trainer = make_trainer_for_unit_test(config=cfg)
    trainer.global_steps = 17
    trainer.current_ref_version = "ref_step_000017"

    observed = {}

    def fake_super_save(self):
        state_path = tmp_path / "global_step_17" / "qrpo_state.json"
        observed["state_exists_during_super_save"] = state_path.exists()

    monkeypatch.setattr(
        qrpo_trainer_module.RayPPOTrainer,
        "_save_checkpoint",
        fake_super_save,
    )

    trainer._save_checkpoint()

    assert observed["state_exists_during_super_save"] is True


def test_fit_runs_validation_and_checkpoint() -> None:
    cfg = make_config(
        **{
            "trainer.total_epochs": 1,
            "trainer.val_before_train": True,
            "trainer.val_only": False,
            "trainer.save_freq": 1,
            "trainer.test_freq": 1,
        }
    )

    trainer = make_trainer_for_unit_test(config=cfg)
    trainer.total_training_steps = 1
    trainer.train_dataloader = [[make_fit_prompt("p0")]]
    trainer.checkpoint_manager = FakeCheckpointManager()
    trainer.actor_rollout_wg = FakeActorRolloutWG()

    trainer.source_scheduler = FakeSourceScheduler(
        [
            SourceCounts(prompt_index=0, prompt_id="p0", n_online=0, n_offline=1),
        ]
    )
    trainer.offline_selector = lambda prompt, n: []
    trainer.online_reward_fn = None

    fake_logger = FakeLogger()
    trainer._make_tracking_logger = lambda: fake_logger
    trainer._load_checkpoint = lambda: None

    calls = {
        "validate": 0,
        "save": 0,
    }

    def fake_validate():
        calls["validate"] += 1
        return {f"val/score_{calls['validate']}": float(calls["validate"])}

    def fake_save_checkpoint():
        calls["save"] += 1

    trainer._validate = fake_validate
    trainer._save_checkpoint = fake_save_checkpoint

    trainer.run_qrpo_iteration = lambda **kwargs: DataProto.from_single_dict(
        data={},
        meta_info={"metrics": {"actor/qrpo_loss": 0.3}},
    )

    trainer.fit()

    assert calls["validate"] == 2
    assert calls["save"] == 1
    assert trainer.checkpoint_manager.updated_steps == [0, 1]


def test_fit_builds_fixed_counts_source_scheduler_from_config() -> None:
    cfg = make_config(
        **{
            "source_schedule.n_online": 0,
            "source_schedule.n_offline": 1,
        }
    )
    trainer = make_trainer_for_unit_test(config=cfg)

    scheduler = trainer._resolve_source_scheduler()

    assert scheduler.n_online == 0
    assert scheduler.n_offline == 1


def test_batch_to_prompt_records_accepts_sequence_and_mapping() -> None:
    p0 = make_fit_prompt("p0")
    p1 = make_fit_prompt("p1")

    assert QRPOTrainer._batch_to_prompt_records([p0, p1]) == [p0, p1]
    assert QRPOTrainer._batch_to_prompt_records({"prompt_records": [p0]}) == [p0]
    assert QRPOTrainer._batch_to_prompt_records({"records": [p1]}) == [p1]
    assert QRPOTrainer._batch_to_prompt_records(p0) == [p0]


def test_batch_to_prompt_records_rejects_unknown_payload() -> None:
    with pytest.raises(TypeError, match="PromptRecord"):
        QRPOTrainer._batch_to_prompt_records({"x": 1})


def _minimal_qrpo_config(
    *,
    n_online: int = 0,
    reward_num_workers: int = 0,
    reward_path=None,
    reward_name="compute_score",
    train_batch_size: int | None = None,
):
    cfg = {
        "trainer": {
            "use_legacy_worker_impl": "disable",
        },
        "actor_rollout_ref": {
            "hybrid_engine": True,
            "actor": {
                "strategy": "fsdp",
                "use_kl_loss": False,
            },
        },
        "algorithm": {
            "use_kl_in_reward": False,
        },
        "source_schedule": {
            "n_online": n_online,
            "n_offline": 1,
        },
        "reward": {
            "num_workers": reward_num_workers,
            "custom_reward_function": {
                "path": reward_path,
                "name": reward_name,
            },
        },
        "qrpo": {
            "beta": 6.0,
            "transform": "identity",
            "length_normalization": True,
        },
    }

    if train_batch_size is not None:
        cfg["data"] = {"train_batch_size": train_batch_size}

    return OmegaConf.create(cfg)


def _rollout_output_for_reward_loop() -> DataProto:
    prompts = torch.tensor(
        [
            [101, 102],
            [201, 202],
        ],
        dtype=torch.long,
    )
    responses = torch.tensor(
        [
            [11, 12, 0],
            [21, 0, 0],
        ],
        dtype=torch.long,
    )
    response_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
        ],
        dtype=torch.bool,
    )
    input_ids = torch.cat([prompts, responses], dim=-1)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    position_ids = torch.arange(input_ids.shape[1]).repeat(input_ids.shape[0], 1)

    ref_rewards = np.empty(2, dtype=object)
    ref_rewards[:] = [(0.1, 0.2), (0.3, 0.4)]

    return DataProto.from_dict(
        tensors={
            K.PROMPTS: prompts,
            K.RESPONSES: responses,
            K.RESPONSE_MASK: response_mask,
            K.INPUT_IDS: input_ids,
            K.ATTENTION_MASK: attention_mask,
            K.POSITION_IDS: position_ids,
        },
        non_tensors={
            K.PROMPT_ID: np.asarray(["prompt-0", "prompt-1"], dtype=object),
            K.TRAJECTORY_ID: np.asarray(["traj-0", "traj-1"], dtype=object),
            K.REF_REWARDS: ref_rewards,
        },
        meta_info={},
    )


def _agent_loop_rollout_output_without_train_metadata() -> DataProto:
    base = _rollout_output_for_reward_loop()
    return DataProto.from_dict(
        tensors={
            K.PROMPTS: base.batch[K.PROMPTS],
            K.RESPONSES: base.batch[K.RESPONSES],
            K.RESPONSE_MASK: base.batch[K.RESPONSE_MASK],
            K.INPUT_IDS: base.batch[K.INPUT_IDS],
            K.ATTENTION_MASK: base.batch[K.ATTENTION_MASK],
            K.POSITION_IDS: base.batch[K.POSITION_IDS],
        },
        non_tensors={},
        meta_info=dict(base.meta_info),
    )


def _rollout_input_for_training_metadata() -> DataProto:
    ref_rewards = np.empty(2, dtype=object)
    ref_rewards[:] = [(0.1, 0.2), (0.3, 0.4)]

    return DataProto(
        non_tensor_batch={
            K.PROMPT_ID: np.asarray(["prompt-0", "prompt-1"], dtype=object),
            K.TRAJECTORY_ID: np.asarray(["traj-0", "traj-1"], dtype=object),
            K.REF_REWARDS: ref_rewards,
        },
        meta_info={"kind": "online_rollout_input"},
    )


def test_compute_online_rewards_from_rollout_output_sums_rm_scores():
    rollout_output = _agent_loop_rollout_output_without_train_metadata()
    rollout_output.batch["rm_scores"] = torch.tensor(
        [
            [0.0, 0.0, 3.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    rollout_output.non_tensor_batch["reward/helpfulness_score"] = np.asarray(
        [4.0, 2.0],
        dtype=object,
    )
    rollout_output.meta_info["reward_extra_keys"] = ["reward/helpfulness_score"]

    trainer = QRPOTrainer.__new__(QRPOTrainer)

    rewards, metrics = trainer._compute_online_rewards_from_rollout_output(
        rollout_output
    )

    assert torch.allclose(rewards, torch.tensor([3.0, 2.0]))
    assert metrics["reward/online_mean"] == pytest.approx(2.5)
    assert metrics["reward/online_min"] == pytest.approx(2.0)
    assert metrics["reward/online_max"] == pytest.approx(3.0)
    assert metrics["reward/helpfulness_score/mean"] == pytest.approx(3.0)


def test_compute_online_rewards_from_rollout_output_requires_rm_scores():
    rollout_output = _rollout_output_for_reward_loop()
    trainer = QRPOTrainer.__new__(QRPOTrainer)

    with pytest.raises(KeyError, match="rm_scores"):
        trainer._compute_online_rewards_from_rollout_output(rollout_output)


def test_compute_online_rewards_from_rollout_output_rejects_bad_rm_score_shape():
    rollout_output = _rollout_output_for_reward_loop()
    rollout_output.batch["rm_scores"] = torch.zeros(2, 2)
    trainer = QRPOTrainer.__new__(QRPOTrainer)

    with pytest.raises(ValueError, match="rm_scores shape"):
        trainer._compute_online_rewards_from_rollout_output(rollout_output)


def test_build_online_completion_log_rows_selects_top_reward_and_decodes():
    trainer = make_trainer_for_unit_test(
        config=make_config(
            **{
                "online_rollout.completion_logging.enabled": True,
                "online_rollout.completion_logging.selection": "top_reward",
                "online_rollout.completion_logging.n": 1,
                "online_rollout.completion_logging.max_chars": 100,
            }
        )
    )
    rollout_output = _rollout_output_for_reward_loop()
    rollout_output.non_tensor_batch[K.RAW_PROMPT] = np.asarray(
        [
            ({"role": "user", "content": "prompt zero"},),
            ({"role": "user", "content": "prompt one"},),
        ],
        dtype=object,
    )
    rollout_output.non_tensor_batch["reward/helpfulness_score"] = np.asarray(
        [1.0, 4.0],
        dtype=object,
    )
    rollout_output.meta_info["reward_extra_keys"] = ["reward/helpfulness_score"]

    requests = [
        SimpleNamespace(
            prompt_id="prompt-0",
            trajectory_id="traj-0",
            prompt_messages=({"role": "user", "content": "fallback zero"},),
            ref_rewards=(0.1, 0.2),
        ),
        SimpleNamespace(
            prompt_id="prompt-1",
            trajectory_id="traj-1",
            prompt_messages=({"role": "user", "content": "fallback one"},),
            ref_rewards=(0.3, 0.4),
        ),
    ]

    rows = build_online_completion_log_rows(
        config=trainer.config,
        online_requests=requests,
        rollout_output=rollout_output,
        rewards=torch.tensor([1.0, 4.0], dtype=torch.float32),
        online_rollout_config=trainer.config.online_rollout,
        tokenizer=trainer.tokenizer,
        meta_info={"global_steps": 7, "epoch": 1},
    )

    assert len(rows) == 1
    assert rows[0]["sample_index"] == 1
    assert rows[0][K.PROMPT_ID] == "prompt-1"
    assert rows[0][K.TRAJECTORY_ID] == "traj-1"
    assert rows[0][K.TRAJECTORY_REWARD] == pytest.approx(4.0)
    assert rows[0][K.REF_REWARDS] == [0.3, 0.4]
    assert rows[0]["prompt"] == "user: prompt one"
    assert rows[0]["completion"] == "21"
    assert rows[0]["reward/helpfulness_score"] == pytest.approx(4.0)


def test_qrpo_update_step_enriches_online_completion_log_rows() -> None:
    trainer = make_trainer_for_unit_test()
    batch = make_qrpo_batch(source=K.SOURCE_ONLINE)

    def fake_compute_ref_log_prob(self, train_batch):
        return DataProto.from_dict(
            tensors={
                K.REF_LOG_PROBS: torch.zeros_like(
                    train_batch.batch[K.RESPONSES],
                    dtype=torch.float32,
                )
            }
        )

    def fake_update_actor(self, train_batch):
        return DataProto.from_single_dict(
            data={},
            meta_info={"metrics": {}},
        )

    trainer._compute_ref_log_prob = MethodType(fake_compute_ref_log_prob, trainer)
    trainer._update_actor = MethodType(fake_update_actor, trainer)

    output = trainer.qrpo_update_step(
        [batch],
        online_completion_log_rows=[
            {
                K.TRAJECTORY_ID: "t0_1",
                K.PROMPT_ID: "p0_1",
                "prompt": "prompt",
                "completion": "completion",
            }
        ],
    )

    rows = output.meta_info[K.ONLINE_COMPLETION_LOG_ROWS]
    assert len(rows) == 1
    assert rows[0][K.TRAJECTORY_ID] == "t0_1"
    assert rows[0][K.TRAJECTORY_REWARD] == pytest.approx(4.0)
    assert rows[0][K.REF_REWARDS] == [1.0, 2.0, 3.0, 4.0]
    assert rows[0][K.REF_QUANTILE] == pytest.approx(1.0)
    assert K.TRANSFORMED_REWARD in rows[0]
    assert K.TRAJECTORY_LENGTH in rows[0]
    assert K.EFFECTIVE_BETA in rows[0]
    assert K.BETA_LOG_PARTITION in rows[0]


def test_run_qrpo_iteration_uses_agent_loop_rewards_for_online_rewards():
    rollout_output = _agent_loop_rollout_output_without_train_metadata()
    rollout_output.batch["rm_scores"] = torch.tensor(
        [
            [0.0, 0.0, 3.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    rollout_output.meta_info["kind"] = "rollout_output"

    class FakeRolloutManager:
        def __init__(self):
            self.inputs = []

        def generate_sequences(self, rollout_input: DataProto) -> DataProto:
            self.inputs.append(rollout_input)
            return rollout_output

    class DummyOfflineSelector:
        def select(self, *args, **kwargs):
            return []

    trainer = QRPOTrainer.__new__(QRPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "online_rollout": {
                "data_source": "activeultrafeedback",
                "agent_name": None,
                "validate": False,
            }
        }
    )
    trainer.async_rollout_manager = FakeRolloutManager()
    trainer.checkpoint_manager = FakeCheckpointManager()
    trainer.ref_reward_store = FakeRefRewardStore({"prompt-0": (0.1, 0.2)})
    trainer.current_ref_version = "unit_ref"

    captured = {}

    def fake_qrpo_update_step(
        batches,
        *,
        pad_token_id=None,
        meta_info=None,
    ):
        assert len(batches) == 1
        captured["train_batch"] = batches[0]
        captured["pad_token_id"] = pad_token_id
        captured["meta_info"] = meta_info
        return DataProto(meta_info={"metrics": {}})

    trainer.qrpo_update_step = fake_qrpo_update_step

    prompt = PromptRecord(
        prompt_id="prompt-0",
        prompt_messages=(
            {"role": "user", "content": "Give a short answer."},
        ),
        ref_rewards=(0.1, 0.2),
        offline_trajectories=(),
        offline_rewards=(),
    )

    output = trainer.run_qrpo_iteration(
        prompt_records=[prompt],
        source_counts=[
            SourceCounts(
                prompt_index=0,
                prompt_id="prompt-0",
                n_online=2,
                n_offline=0,
            )
        ],
        offline_selector=DummyOfflineSelector(),
        actor_version=123,
        meta_info={"global_steps": 1, "epoch": 0},
    )

    assert trainer.async_rollout_manager.inputs
    assert trainer.checkpoint_manager.sleep_calls == 1
    rollout_input = trainer.async_rollout_manager.inputs[0]
    assert rollout_input.non_tensor_batch[K.DATA_SOURCE].tolist() == [
        "activeultrafeedback",
        "activeultrafeedback",
    ]
    assert K.REWARD_MODEL in rollout_input.non_tensor_batch
    assert K.EXTRA_INFO in rollout_input.non_tensor_batch

    train_batch = captured["train_batch"]
    assert torch.allclose(
        train_batch.batch[K.TRAJECTORY_REWARD],
        torch.tensor([3.0, 2.0]),
    )

    assert output.meta_info["metrics"]["reward/online_mean"] == pytest.approx(2.5)
    assert output.meta_info["metrics"]["reward/online_min"] == pytest.approx(2.0)
    assert output.meta_info["metrics"]["reward/online_max"] == pytest.approx(3.0)


def test_validate_qrpo_config_allows_offline_without_reward_loop():
    config = _minimal_qrpo_config(
        n_online=0,
        reward_num_workers=0,
        reward_path=None,
    )

    validate_qrpo_trainer_config(config)


def test_validate_qrpo_config_requires_reward_workers_for_online(tmp_path):
    reward_path = tmp_path / "reward.py"
    reward_path.write_text("async def compute_score(*args, **kwargs): return 1.0\n")

    config = _minimal_qrpo_config(
        n_online=1,
        reward_num_workers=0,
        reward_path=str(reward_path),
    )

    with pytest.raises(ValueError, match="reward.num_workers"):
        validate_qrpo_trainer_config(config)


def test_validate_qrpo_config_requires_reward_loop_for_generated_refs(tmp_path):
    reward_path = tmp_path / "reward.py"
    reward_path.write_text("async def compute_score(*args, **kwargs): return 1.0\n")

    config = _minimal_qrpo_config(
        n_online=0,
        reward_num_workers=0,
        reward_path=str(reward_path),
    )
    config.ref_rewards = {"initial_source": "generate"}

    with pytest.raises(ValueError, match="reward.num_workers"):
        validate_qrpo_trainer_config(config)


def test_validate_qrpo_config_requires_reward_loop_for_ref_refresh(tmp_path):
    reward_path = tmp_path / "reward.py"
    reward_path.write_text("async def compute_score(*args, **kwargs): return 1.0\n")

    config = _minimal_qrpo_config(
        n_online=0,
        reward_num_workers=0,
        reward_path=str(reward_path),
    )
    config.ref_rewards = {
        "initial_source": "dataset",
        "refresh_interval_epochs": 0.5,
    }

    with pytest.raises(ValueError, match="reward.num_workers"):
        validate_qrpo_trainer_config(config)


def test_validate_qrpo_config_requires_reward_path_for_online():
    config = _minimal_qrpo_config(
        n_online=1,
        reward_num_workers=2,
        reward_path=None,
    )

    with pytest.raises(ValueError, match="custom_reward_function.path"):
        validate_qrpo_trainer_config(config)


def test_validate_qrpo_config_allows_online_with_reward_loop(tmp_path):
    reward_path = tmp_path / "reward.py"
    reward_path.write_text("async def compute_score(*args, **kwargs): return 1.0\n")

    config = _minimal_qrpo_config(
        n_online=1,
        reward_num_workers=2,
        reward_path=str(reward_path),
        reward_name="compute_score",
    )

    validate_qrpo_trainer_config(config)


def test_validate_qrpo_config_rejects_non_positive_effective_beta_max():
    for value in (0.0, -1.0, float("nan")):
        config = _minimal_qrpo_config()
        config.qrpo.effective_beta_max = value

        with pytest.raises(ValueError, match="effective_beta_max"):
            validate_qrpo_trainer_config(config)


def test_validate_qrpo_config_allows_streaming_reward_workers_without_batch_divisibility(
    tmp_path,
):
    reward_path = tmp_path / "reward.py"
    reward_path.write_text("async def compute_score(*args, **kwargs): return 1.0\n")

    config = _minimal_qrpo_config(
        n_online=1,
        reward_num_workers=4,
        reward_path=str(reward_path),
        reward_name="compute_score",
        train_batch_size=2,
    )

    validate_qrpo_trainer_config(config)
