from __future__ import annotations

from types import MethodType

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


class FakeTokenizer:
    pad_token_id = 7


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

    def update_weights(self, step):
        self.updated_steps.append(step)


class FakeSourceScheduler:
    def __init__(self, source_counts):
        self.source_counts = source_counts
        self.calls = []

    def plan(self, prompts):
        self.calls.append(prompts)
        return self.source_counts


class FakeActorRolloutWG:
    pass


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

    return trainer


def make_qrpo_batch(
    *,
    prefix: int = 0,
    prompt_len: int = 2,
    response_len: int = 3,
    source: str = "offline",
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
    assert out.meta_info["metrics"]["actor/qrpo_loss"] == 0.123


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


def test_run_qrpo_iteration_builds_mixed_batch_and_updates(monkeypatch) -> None:
    from trainers import qrpo_trainer as qrpo_trainer_module

    trainer = make_trainer_for_unit_test()
    trainer.global_steps = 12

    offline_candidate = object()
    online_request = object()

    offline_train_batch = make_qrpo_batch(prefix=0, prompt_len=2, response_len=3)
    online_rollout_input = DataProto(meta_info={"kind": "online_rollout_input"})
    rollout_output = DataProto(meta_info={"kind": "rollout_output"})
    online_train_batch = make_qrpo_batch(prefix=1000, prompt_len=4, response_len=5)

    calls = {
        "build_candidates": 0,
        "offline_to_dataproto": 0,
        "online_requests_to_dataproto": 0,
        "generate_sequences": 0,
        "reward": 0,
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
        assert config == {"require_assistant_mask": False}
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
        assert config == {"agent_name": "tool_agent"}
        assert meta_info == {"global_steps": 12}
        return online_rollout_input

    class FakeAsyncRolloutManager:
        def generate_sequences(self, batch):
            calls["generate_sequences"] += 1
            assert batch is online_rollout_input
            return rollout_output

    def fake_reward_fn(batch):
        calls["reward"] += 1
        assert batch is rollout_output
        return torch.tensor([1.0, 2.0], dtype=torch.float32)

    def fake_online_rollout_output_to_train_dataproto(
        *,
        rollout_output,
        rewards,
        meta_info,
    ):
        calls["online_output_to_train"] += 1
        assert rollout_output.meta_info["kind"] == "rollout_output"
        assert torch.equal(rewards, torch.tensor([1.0, 2.0]))
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
    trainer.qrpo_update_step = MethodType(fake_qrpo_update_step, trainer)

    out = trainer.run_qrpo_iteration(
        prompt_records=[object()],
        source_counts=[object()],
        offline_selector=lambda prompt, n: [],
        offline_tokenization_config={"require_assistant_mask": False},
        online_rollout_config={"agent_name": "tool_agent"},
        online_reward_fn=fake_reward_fn,
        meta_info={"global_steps": 12},
    )

    assert calls == {
        "build_candidates": 1,
        "offline_to_dataproto": 1,
        "online_requests_to_dataproto": 1,
        "generate_sequences": 1,
        "reward": 1,
        "online_output_to_train": 1,
        "qrpo_update": 1,
    }

    assert out.meta_info["metrics"]["actor/qrpo_loss"] == 0.25


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

    online_rollout_input = DataProto(meta_info={"kind": "online_rollout_input"})
    rollout_output = DataProto(meta_info={"kind": "rollout_output"})
    online_train_batch = make_qrpo_batch(prefix=1000, prompt_len=4, response_len=5)

    calls = {
        "online_requests_to_dataproto": 0,
        "generate_sequences": 0,
        "reward": 0,
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
        return online_rollout_input

    class FakeAsyncRolloutManager:
        def generate_sequences(self, batch):
            calls["generate_sequences"] += 1
            assert batch is online_rollout_input
            return rollout_output

    def fake_reward_fn(batch):
        calls["reward"] += 1
        assert batch is rollout_output
        return torch.tensor([1.0, 2.0], dtype=torch.float32)

    def fake_online_rollout_output_to_train_dataproto(
        *,
        rollout_output,
        rewards,
        meta_info,
    ):
        calls["online_output_to_train"] += 1
        assert rollout_output.meta_info["kind"] == "rollout_output"
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
    trainer.qrpo_update_step = MethodType(fake_qrpo_update_step, trainer)

    out = trainer.run_qrpo_iteration(
        prompt_records=[object()],
        source_counts=[object()],
        offline_selector=lambda prompt, n: [],
        online_rollout_config={"agent_name": "tool_agent"},
        online_reward_fn=fake_reward_fn,
    )

    assert calls == {
        "online_requests_to_dataproto": 1,
        "generate_sequences": 1,
        "reward": 1,
        "online_output_to_train": 1,
        "qrpo_update": 1,
    }
    assert out.meta_info["metrics"]["actor/qrpo_loss"] == 0.2


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
            online_reward_fn=lambda batch: torch.ones(1),
        )


def test_run_qrpo_iteration_requires_online_reward_fn_when_online_exists(monkeypatch) -> None:
    from trainers import qrpo_trainer as qrpo_trainer_module

    trainer = make_trainer_for_unit_test()

    monkeypatch.setattr(
        qrpo_trainer_module,
        "build_training_candidates",
        lambda **kwargs: ([], [object()]),
    )

    with pytest.raises(ValueError, match="online_reward_fn"):
        trainer.run_qrpo_iteration(
            prompt_records=[object()],
            source_counts=[object()],
            offline_selector=lambda prompt, n: [],
            online_rollout_config={"agent_name": "tool_agent"},
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
    assert trainer.checkpoint_manager.updated_steps == [0, 1, 2]

    assert [call["prompt_id"] for call in calls] == ["p0", "p1"]
    assert [call["actor_version"] for call in calls] == [1, 2]
    assert [call["meta_info"]["global_steps"] for call in calls] == [1, 2]

    assert len(fake_logger.calls) == 2
    assert fake_logger.calls[0][0]["training/global_step"] == 1
    assert fake_logger.calls[1][0]["training/global_step"] == 2


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
