from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pprint import pprint
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role
from verl.utils.metric import reduce_metrics

from algorithm.qrpo_fields import add_qrpo_fields
from batch import keys as K
from batch.offline_tokenization import offline_candidates_to_dataproto
from batch.source_schedule import FixedCountsSourceScheduler, SourceCounts
from batch.training_candidates import OfflineSelector, build_training_candidates
from data.schemas import PromptRecord
from verl_adapters.engine_worker import QRPOEngineActorRolloutRefWorker
from verl_adapters.rollout import (
    online_rollout_output_to_train_dataproto,
    online_rollout_requests_to_dataproto,
)
from verl_adapters.train_batch import concat_qrpo_training_dataprotos


GLOBAL_POOL_ID = "global_pool"
OnlineRewardFn = Callable[[DataProto], torch.Tensor]


class QRPOTrainer(RayPPOTrainer):
    """QRPO trainer on top of VERL 0.7.1's new model-engine RayPPOTrainer.

    Main public methods:

      qrpo_update_step(...)
        One monolithic actor update from already-built online/offline train batches.

      run_qrpo_iteration(...)
        One QRPO iteration from prompt records and source counts.

      fit(...)
        Training loop.

    This trainer reuses VERL's distributed worker orchestration, no-padding
    conversion, reference-logprob computation, checkpointing, validation, and
    actor-update dispatch.
    """

    def __init__(
        self,
        *args,
        source_scheduler: Any | None = None,
        offline_selector: OfflineSelector | None = None,
        online_reward_fn: OnlineRewardFn | None = None,
        **kwargs,
    ):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]

        validate_qrpo_trainer_config(config)

        self.source_scheduler = source_scheduler
        self.offline_selector = offline_selector
        self.online_reward_fn = online_reward_fn

        super().__init__(*args, **kwargs)

        # RayPPOTrainer derives this from PPO KL config. QRPO always needs
        # reference-policy log-probs even when PPO-style KL settings are off.
        self.use_reference_policy = True

    def qrpo_update_step(
        self,
        batches: Sequence[DataProto | None],
        *,
        pad_token_id: int | None = None,
        meta_info: dict[str, Any] | None = None,
    ) -> DataProto:
        """Run one QRPO actor update from one or more train batches.

        Input batches may come from online rollouts, offline trajectories, or
        both. They must already use the shared VERL prompt/response schema:

          prompts
          responses
          response_mask
          input_ids
          attention_mask
          position_ids
          trajectory_reward
          ref_rewards

        After concatenation, QRPO treats the result as one monolithic training
        batch. Source information remains only as metadata.
        """

        non_empty_batches = [batch for batch in batches if batch is not None]
        if not non_empty_batches:
            raise ValueError("At least one QRPO training batch is required.")

        pad_token_id = self._resolve_pad_token_id(pad_token_id)

        train_batch = concat_qrpo_training_dataprotos(
            non_empty_batches,
            pad_token_id=pad_token_id,
            meta_info=meta_info,
        )

        self._check_qrpo_train_batch_has_required_keys(train_batch)

        train_batch = add_qrpo_fields(
            train_batch,
            config=self.config.qrpo,
        )

        ref_log_prob = self._compute_ref_log_prob(train_batch)
        train_batch = train_batch.union(ref_log_prob)

        self._check_ref_log_prob_shape(train_batch)

        return self._update_actor(train_batch)

    def run_qrpo_iteration(
        self,
        *,
        prompt_records: Sequence[PromptRecord],
        source_counts: Sequence[SourceCounts],
        offline_selector: OfflineSelector | None = None,
        tokenizer: Any | None = None,
        actor_version: str | int | None = None,
        offline_tokenization_config: Mapping[str, Any] | None = None,
        online_rollout_config: Mapping[str, Any] | None = None,
        online_reward_fn: OnlineRewardFn | None = None,
        pad_token_id: int | None = None,
        meta_info: dict[str, Any] | None = None,
    ) -> DataProto:
        """Run one QRPO iteration from prompt records and source counts.

        Flow:

          1. source counts + offline selector -> offline candidates / online requests
          2. offline candidates -> offline train batch
          3. online requests -> rollout input
          4. rollout input -> online rollout output
          5. rollout output -> online rewards
          6. rollout output + rewards -> online train batch
          7. online/offline train batches -> one monolithic QRPO update

        This method does not do logging, checkpointing, validation, or ref-model
        updates. Those belong to fit().
        """

        if actor_version is None:
            actor_version = getattr(self, "global_steps", 0)

        if offline_selector is None:
            offline_selector = self._resolve_offline_selector()

        offline_candidates, online_requests = build_training_candidates(
            prompt_records=prompt_records,
            source_counts=source_counts,
            offline_selector=offline_selector,
            actor_version=actor_version,
        )

        train_batches: list[DataProto] = []

        if offline_candidates:
            tokenizer = self._resolve_tokenizer(tokenizer)

            if offline_tokenization_config is None:
                offline_tokenization_config = (
                    _select(self.config, "offline_tokenization", default={}) or {}
                )

            offline_train_batch = offline_candidates_to_dataproto(
                candidates=offline_candidates,
                tokenizer=tokenizer,
                config=offline_tokenization_config,
                meta_info=meta_info,
            )
            train_batches.append(offline_train_batch)

        if online_requests:
            if online_rollout_config is None:
                online_rollout_config = _select(
                    self.config,
                    "online_rollout",
                    default=None,
                )

            if online_rollout_config is None:
                raise ValueError(
                    "online_rollout_config is required when online requests are present."
                )

            if online_reward_fn is None:
                online_reward_fn = self._resolve_online_reward_fn()

            if online_reward_fn is None:
                raise ValueError(
                    "online_reward_fn is required when online requests are present."
                )

            online_rollout_input = online_rollout_requests_to_dataproto(
                requests=online_requests,
                config=online_rollout_config,
                meta_info=meta_info,
            )

            rollout_output = self.async_rollout_manager.generate_sequences(
                online_rollout_input
            )

            online_rewards = online_reward_fn(rollout_output)

            online_train_batch = online_rollout_output_to_train_dataproto(
                rollout_output=rollout_output,
                rewards=online_rewards,
                meta_info=meta_info,
            )
            train_batches.append(online_train_batch)

        if not train_batches:
            raise RuntimeError("No QRPO train batches were produced.")

        return self.qrpo_update_step(
            train_batches,
            pad_token_id=pad_token_id,
            meta_info=meta_info,
        )

    def fit(self):
        """Run QRPO training.

        Reused from RayPPOTrainer:
          - worker initialization/checkpoint state
          - _compute_ref_log_prob(...)
          - _update_actor(...)
          - _save_checkpoint(...)
          - _validate(...)

        QRPO does not use:
          - old_log_probs
          - PPO advantages/returns
          - critic/value updates
          - token-level reward shaping
          - PPO KL-in-reward
        """

        logger = self._make_tracking_logger()

        self.global_steps = getattr(self, "global_steps", 0)

        self._load_checkpoint()
        # self.checkpoint_manager.update_weights(self.global_steps)

        if self.config.trainer.get("val_before_train", True):
            # Validation uses rollout generation, so rollout weights must be synced.
            self.checkpoint_manager.update_weights(self.global_steps)

            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)

            if self.config.trainer.get("val_only", False):
                return

        source_scheduler = self._resolve_source_scheduler()

        # Fail early for offline selection. Do not require online_reward_fn here:
        # offline-only training should work without it.
        self._resolve_offline_selector()

        current_epoch = self.global_steps // len(self.train_dataloader)

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="QRPO Training Progress",
        )

        last_val_metrics = None

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_payload in self.train_dataloader:
                if self.global_steps >= self.total_training_steps:
                    progress_bar.close()
                    return

                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)

                prompt_records = self._batch_to_prompt_records(batch_payload)
                source_counts = source_scheduler.plan(prompt_records)

                self.global_steps += 1
                is_last_step = self.global_steps >= self.total_training_steps

                meta_info = {
                    "global_steps": self.global_steps,
                    "epoch": epoch,
                }

                actor_output = self.run_qrpo_iteration(
                    prompt_records=prompt_records,
                    source_counts=source_counts,
                    actor_version=self.global_steps,
                    meta_info=meta_info,
                )

                metrics = {}

                actor_metrics = actor_output.meta_info.get("metrics", {})
                metrics.update(reduce_metrics(actor_metrics))

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "qrpo/n_prompts": len(prompt_records),
                        "qrpo/n_online": sum(
                            counts.n_online for counts in source_counts
                        ),
                        "qrpo/n_offline": sum(
                            counts.n_offline for counts in source_counts
                        ),
                    }
                )

                if self.config.trainer.save_freq > 0 and (
                    is_last_step
                    or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    self._save_checkpoint()

                will_validate = self.config.trainer.test_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                )

                needs_rollout_sync = (
                        any(counts.n_online > 0 for counts in source_counts)
                        or will_validate
                )

                if needs_rollout_sync:
                    # Needed for online rollout and validation, but skipped for pure offline training.
                    self.checkpoint_manager.update_weights(self.global_steps)

                if will_validate:
                    val_metrics = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def _resolve_tokenizer(self, tokenizer: Any | None):
        if tokenizer is not None:
            return tokenizer

        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "A tokenizer must be provided, or QRPOTrainer.tokenizer must be set."
            )

        return tokenizer

    def _resolve_pad_token_id(self, pad_token_id: int | None) -> int:
        if pad_token_id is not None:
            return int(pad_token_id)

        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "pad_token_id must be provided when QRPOTrainer.tokenizer is not set."
            )

        tokenizer_pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if tokenizer_pad_token_id is None:
            raise ValueError(
                "pad_token_id must be provided because tokenizer.pad_token_id is None."
            )

        return int(tokenizer_pad_token_id)

    @staticmethod
    def _check_qrpo_train_batch_has_required_keys(batch: DataProto) -> None:
        if batch.batch is None:
            raise ValueError("QRPO train batch must contain tensor batch data.")

        required = [
            K.PROMPTS,
            K.RESPONSES,
            K.RESPONSE_MASK,
            K.INPUT_IDS,
            K.ATTENTION_MASK,
            K.POSITION_IDS,
            K.TRAJECTORY_REWARD,
            K.REF_REWARDS,
        ]

        for key in required:
            if key not in batch.batch:
                raise KeyError(f"QRPO train batch is missing required key {key!r}.")

    @staticmethod
    def _check_ref_log_prob_shape(batch: DataProto) -> None:
        if batch.batch is None:
            raise ValueError("QRPO train batch must contain tensor batch data.")

        if K.REF_LOG_PROBS not in batch.batch:
            raise KeyError(
                f"VERL _compute_ref_log_prob(...) did not add {K.REF_LOG_PROBS!r}."
            )

        if batch.batch[K.REF_LOG_PROBS].shape != batch.batch[K.RESPONSES].shape:
            raise ValueError(
                f"{K.REF_LOG_PROBS!r} shape {tuple(batch.batch[K.REF_LOG_PROBS].shape)} "
                f"does not match {K.RESPONSES!r} shape "
                f"{tuple(batch.batch[K.RESPONSES].shape)}."
            )

    def _make_tracking_logger(self):
        from verl.utils.tracking import Tracking

        return Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=_to_plain_container(self.config),
        )

    def _resolve_source_scheduler(self):
        if self.source_scheduler is not None:
            return self.source_scheduler

        scheduler_config = (
            _select(self.config, "source_schedule", default=None)
            or _select(self.config, "qrpo.source_schedule", default=None)
        )
        if scheduler_config is None:
            raise ValueError(
                "QRPOTrainer requires source_scheduler or config.source_schedule."
            )

        scheduler_config = _to_plain_container(scheduler_config)
        self.source_scheduler = FixedCountsSourceScheduler.from_config(scheduler_config)
        return self.source_scheduler

    def _resolve_offline_selector(self) -> OfflineSelector:
        if self.offline_selector is None:
            raise ValueError("QRPOTrainer requires offline_selector.")
        return self.offline_selector

    def _resolve_online_reward_fn(self) -> OnlineRewardFn | None:
        return self.online_reward_fn

    @staticmethod
    def _batch_to_prompt_records(batch_payload: Any) -> list[PromptRecord]:
        if isinstance(batch_payload, PromptRecord):
            return [batch_payload]

        if isinstance(batch_payload, Sequence) and not isinstance(
            batch_payload,
            (str, bytes, bytearray),
        ):
            if all(isinstance(item, PromptRecord) for item in batch_payload):
                return list(batch_payload)

        if isinstance(batch_payload, Mapping):
            for key in ("prompt_records", "records", "prompts"):
                value = batch_payload.get(key)
                if value is not None:
                    if isinstance(value, PromptRecord):
                        return [value]
                    if isinstance(value, Sequence) and all(
                        isinstance(item, PromptRecord) for item in value
                    ):
                        return list(value)

        raise TypeError(
            "QRPOTrainer.fit expects train_dataloader batches to be PromptRecord, "
            "Sequence[PromptRecord], or a mapping containing 'prompt_records'."
        )


def validate_qrpo_trainer_config(config: Any) -> None:
    """Validate minimal assumptions for QRPO on VERL's new model-engine stack."""

    if config is None:
        raise ValueError("QRPOTrainer requires a config.")

    use_legacy_worker_impl = _select(
        config,
        "trainer.use_legacy_worker_impl",
        default=None,
    )
    if use_legacy_worker_impl != "disable":
        raise ValueError(
            "QRPOTrainer currently targets VERL's new model-engine stack. "
            "Set trainer.use_legacy_worker_impl=disable."
        )

    strategy = _select(config, "actor_rollout_ref.actor.strategy", default=None)
    if strategy not in {"fsdp", "fsdp2", "megatron"}:
        raise ValueError(
            "QRPOTrainer expects actor_rollout_ref.actor.strategy to be one of "
            f"'fsdp', 'fsdp2', or 'megatron', got {strategy!r}."
        )

    hybrid_engine = _select(config, "actor_rollout_ref.hybrid_engine", default=True)
    if not hybrid_engine:
        raise ValueError("QRPOTrainer requires actor_rollout_ref.hybrid_engine=true.")

    actor_use_kl_loss = bool(
        _select(config, "actor_rollout_ref.actor.use_kl_loss", default=False)
    )
    if actor_use_kl_loss:
        raise ValueError(
            "Disable actor_rollout_ref.actor.use_kl_loss for QRPO. "
            "QRPO's residual already contains the reference-policy regularization term."
        )

    use_kl_in_reward = bool(
        _select(
            config,
            "algorithm.use_kl_in_reward",
            default=False,
        )
    )
    if use_kl_in_reward:
        raise ValueError(
            "Disable algorithm.use_kl_in_reward for QRPO. "
            "QRPO should use raw sequence rewards plus explicit ref_log_prob."
        )

    if _select(config, "qrpo.beta", default=None) is None:
        raise ValueError("QRPOTrainer requires qrpo.beta.")

    length_normalization = _select(
        config,
        "qrpo.length_normalization",
        default=True,
    )
    if not isinstance(length_normalization, bool):
        raise ValueError("qrpo.length_normalization must be a boolean.")

    transform = _select(config, "qrpo.transform", default="identity")
    if transform != "identity":
        raise ValueError("Only qrpo.transform='identity' is implemented for now.")


def build_qrpo_role_worker_mapping(*, use_ray_remote: bool = True):
    """Build role-worker mapping for QRPO's new-stack fused Actor/Rollout/Ref worker."""

    worker_cls = QRPOEngineActorRolloutRefWorker

    if use_ray_remote:
        import ray

        worker_cls = ray.remote(worker_cls)

    return {
        Role.ActorRolloutRef: worker_cls,
    }


def build_qrpo_resource_pool_mapping():
    """Build resource-pool mapping for QRPO's fused Actor/Rollout/Ref worker."""

    return {
        Role.ActorRolloutRef: GLOBAL_POOL_ID,
    }


def _select(config: Any, path: str, *, default: Any) -> Any:
    if isinstance(config, DictConfig):
        return OmegaConf.select(config, path, default=default)

    cur = config
    for part in path.split("."):
        try:
            cur = cur[part]
        except Exception:
            return default

    return cur


def _to_plain_container(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value
