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
from verl_adapters.online_completion_logging import (
    build_online_completion_log_rows,
    enrich_online_completion_log_rows,
    log_online_completion_rows,
)
from verl_adapters.rollout import (
    attach_online_rollout_train_metadata,
    extract_online_rollout_train_metadata,
    online_rollout_output_to_train_dataproto,
    online_rollout_requests_to_dataproto,
)
from verl_adapters.train_batch import concat_qrpo_training_dataprotos


GLOBAL_POOL_ID = "global_pool"


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
        **kwargs,
    ):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]

        validate_qrpo_trainer_config(config)

        self.source_scheduler = source_scheduler
        self.offline_selector = offline_selector

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
        online_completion_log_rows: Sequence[Mapping[str, Any]] | None = None,
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
        qrpo_batch_metrics = self._compute_qrpo_batch_metrics(train_batch)
        enriched_completion_log_rows = enrich_online_completion_log_rows(
            train_batch=train_batch,
            rows=online_completion_log_rows,
        )

        if getattr(self, "ref_in_actor", False):
            self._ensure_actor_model_on_device()

        ref_log_prob = self._compute_ref_log_prob(train_batch)
        train_batch = train_batch.union(ref_log_prob)

        self._check_ref_log_prob_shape(train_batch)

        self._ensure_actor_model_on_device()
        actor_output = self._update_actor(train_batch)
        if qrpo_batch_metrics:
            actor_output.meta_info.setdefault("metrics", {}).update(qrpo_batch_metrics)
        if enriched_completion_log_rows:
            actor_output.meta_info[K.ONLINE_COMPLETION_LOG_ROWS] = (
                enriched_completion_log_rows
            )
        return actor_output

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
        online_reward_metrics: dict[str, float] = {}
        online_completion_log_rows: list[dict[str, Any]] = []
        rollout_temperature = self._resolve_rollout_temperature()

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
            offline_tokenization_config = dict(offline_tokenization_config)
            offline_tokenization_config.setdefault(K.TEMPERATURE, rollout_temperature)

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

            online_rollout_input = online_rollout_requests_to_dataproto(
                requests=online_requests,
                config=online_rollout_config,
                meta_info=meta_info,
            )
            online_train_metadata = extract_online_rollout_train_metadata(
                online_rollout_input
            )

            rollout_output = self.async_rollout_manager.generate_sequences(
                online_rollout_input
            )
            self._sleep_rollout_replicas_after_generation()
            attach_online_rollout_train_metadata(
                rollout_output=rollout_output,
                metadata=online_train_metadata,
            )

            online_rewards, online_reward_metrics = (
                self._compute_online_rewards_from_rollout_output(rollout_output)
            )
            online_completion_log_rows = build_online_completion_log_rows(
                config=self.config,
                online_requests=online_requests,
                rollout_output=rollout_output,
                rewards=online_rewards,
                online_rollout_config=online_rollout_config,
                tokenizer=(
                    tokenizer
                    if tokenizer is not None
                    else getattr(self, "tokenizer", None)
                ),
                meta_info=meta_info,
            )

            online_train_batch = online_rollout_output_to_train_dataproto(
                rollout_output=rollout_output,
                rewards=online_rewards,
                temperature=rollout_temperature,
                meta_info=meta_info,
            )
            train_batches.append(online_train_batch)

        if not train_batches:
            raise RuntimeError("No QRPO train batches were produced.")

        qrpo_update_kwargs = {
            "pad_token_id": pad_token_id,
            "meta_info": meta_info,
        }
        if online_completion_log_rows:
            qrpo_update_kwargs["online_completion_log_rows"] = online_completion_log_rows

        actor_output = self.qrpo_update_step(
            train_batches,
            **qrpo_update_kwargs,
        )

        if online_reward_metrics:
            actor_output.meta_info.setdefault("metrics", {}).update(online_reward_metrics)

        return actor_output

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

        # True iff rollout/vLLM weights are known to match the current actor.
        # Offline-only training should avoid rollout sync entirely.
        rollout_weights_synced = False

        if self.config.trainer.get("val_before_train", True):
            # Validation uses rollout generation, so sync immediately before validation.
            self.checkpoint_manager.update_weights(self.global_steps)
            rollout_weights_synced = True

            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)

            if self.config.trainer.get("val_only", False):
                return

        source_scheduler = self._resolve_source_scheduler()

        # Fail early for offline selection.
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

                has_online = any(counts.n_online > 0 for counts in source_counts)

                # Online rollout generation uses the rollout/vLLM engine, so sync
                # immediately before generation if the rollout weights are stale.
                if has_online and not rollout_weights_synced:
                    self.checkpoint_manager.update_weights(self.global_steps)
                    rollout_weights_synced = True

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

                # run_qrpo_iteration(...) ends with _update_actor(...), so actor
                # weights changed and rollout/vLLM weights are stale again.
                rollout_weights_synced = False

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
                log_online_completion_rows(
                    config=self.config,
                    rows=actor_output.meta_info.get(K.ONLINE_COMPLETION_LOG_ROWS, []),
                    step=self.global_steps,
                )

                if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    self._save_checkpoint()

                will_validate = self.config.trainer.test_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.test_freq == 0
                )

                if will_validate:
                    # Validation uses rollout generation, so sync immediately before it.
                    self.checkpoint_manager.update_weights(self.global_steps)
                    rollout_weights_synced = True

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

    def _compute_online_rewards_from_rollout_output(
            self,
            rollout_output: DataProto,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute sequence-level online rewards from AgentLoop rollout output."""

        if rollout_output.batch is None:
            raise ValueError("rollout_output.batch is required.")

        if "rm_scores" not in rollout_output.batch:
            raise KeyError(
                "QRPO online training requires rollout_output.batch['rm_scores'] "
                "from AgentLoop reward computation. Make sure reward is computed "
                "during rollout, e.g. by configuring reward.num_workers > 0 and "
                "reward.custom_reward_function."
            )

        rm_scores = rollout_output.batch["rm_scores"].float()

        responses = rollout_output.batch[K.RESPONSES]
        if rm_scores.shape != responses.shape:
            raise ValueError(
                f"AgentLoop rm_scores shape {tuple(rm_scores.shape)} does not match "
                f"responses shape {tuple(responses.shape)}."
            )

        rewards = rm_scores.sum(dim=-1).float()

        if rewards.ndim != 1 or rewards.shape[0] != len(rollout_output):
            raise ValueError(
                f"Online rewards must have shape ({len(rollout_output)},), "
                f"got {tuple(rewards.shape)}."
            )

        metrics: dict[str, float] = {
            "reward/online_mean": float(rewards.mean().item()),
            "reward/online_min": float(rewards.min().item()),
            "reward/online_max": float(rewards.max().item()),
        }

        reward_extra_keys = (rollout_output.meta_info or {}).get("reward_extra_keys", [])
        for key in reward_extra_keys:
            if key not in rollout_output.non_tensor_batch:
                continue

            values = rollout_output.non_tensor_batch[key]
            try:
                numeric_values = torch.tensor(
                    [float(x) for x in values],
                    dtype=torch.float32,
                )
            except Exception:
                continue

            metrics[f"{key}/mean"] = float(numeric_values.mean().item())

        return rewards, metrics

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

    def _resolve_rollout_temperature(self) -> float:
        value = _select(
            self.config,
            "actor_rollout_ref.rollout.temperature",
            default=1.0,
        )
        return float(value)

    def _sleep_rollout_replicas_after_generation(self) -> None:
        checkpoint_manager = getattr(self, "checkpoint_manager", None)
        if checkpoint_manager is None:
            return

        sleep_replicas = getattr(checkpoint_manager, "sleep_replicas", None)
        if sleep_replicas is None:
            return

        sleep_replicas()

    def _ensure_actor_model_on_device(self) -> None:
        actor_rollout_wg = getattr(self, "actor_rollout_wg", None)
        if actor_rollout_wg is None:
            return

        to_device = getattr(actor_rollout_wg, "to", None)
        if to_device is None:
            return

        # VERL 0.7.1's naive rollout weight sync offloads the actor model to
        # CPU even when actor.fsdp_config.param_offload=false. The engine context
        # only auto-loads it back when param offload is enabled, so QRPO must
        # restore the model before entering the actor update.
        to_device("device", model=True, optimizer=False, grad=False)

    @staticmethod
    def _compute_qrpo_batch_metrics(batch: DataProto) -> dict[str, float]:
        if batch.batch is None:
            return {}

        metrics: dict[str, float] = {}
        metric_keys = (
            (K.TRAJECTORY_REWARD, "trajectory_reward"),
            (K.REF_QUANTILE, "ref_quantile"),
            (K.TRANSFORMED_REWARD, "transformed_reward"),
            (K.TRAJECTORY_LENGTH, "trajectory_length"),
            (K.EFFECTIVE_BETA, "effective_beta"),
            (K.BETA_LOG_PARTITION, "beta_log_partition"),
        )

        groups: list[tuple[str, list[int] | None]] = [("all", None)]
        sources = batch.non_tensor_batch.get(K.SOURCE)
        if sources is not None:
            for source in (K.SOURCE_ONLINE, K.SOURCE_OFFLINE):
                indices = [
                    i
                    for i, value in enumerate(sources)
                    if str(value) == source
                ]
                if indices:
                    groups.append((source, indices))

        for group_name, indices in groups:
            for key, metric_name in metric_keys:
                if key not in batch.batch:
                    continue

                values = batch.batch[key].detach().float()
                if values.ndim != 1:
                    continue

                if indices is not None:
                    index = torch.tensor(indices, device=values.device)
                    values = values.index_select(0, index)

                if values.numel() == 0:
                    continue

                prefix = f"qrpo/{group_name}/{metric_name}"
                metrics[f"{prefix}_mean"] = float(values.mean().item())
                metrics[f"{prefix}_min"] = float(values.min().item())
                metrics[f"{prefix}_max"] = float(values.max().item())

                if key == K.REF_QUANTILE:
                    metrics[f"{prefix}_at_zero_frac"] = float(
                        (values <= 0.0).float().mean().item()
                    )
                    metrics[f"{prefix}_at_one_frac"] = float(
                        (values >= 1.0).float().mean().item()
                    )

        return metrics

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

    n_online = int(
        _select(config, "source_schedule.n_online", default=0)
        or _select(config, "qrpo.source_schedule.n_online", default=0)
        or 0
    )

    if n_online > 0:
        reward_num_workers = int(_select(config, "reward.num_workers", default=0))
        if reward_num_workers <= 0:
            raise ValueError(
                "Online QRPO requires reward.num_workers > 0 so AgentLoop "
                "computes reward during rollout."
            )

        reward_path = _select(config, "reward.custom_reward_function.path", default=None)
        if reward_path is None:
            raise ValueError(
                "Online QRPO requires reward.custom_reward_function.path."
            )

        reward_name = _select(config, "reward.custom_reward_function.name", default=None)
        if not reward_name:
            raise ValueError(
                "Online QRPO requires reward.custom_reward_function.name."
            )


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
