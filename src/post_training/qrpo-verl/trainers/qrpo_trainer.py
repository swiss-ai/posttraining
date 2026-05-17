from __future__ import annotations

import hashlib
import json
import math
import os
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
from data.dataset_adapter import dataset_batch_to_prompt_records
from data.schemas import PromptRecord
from ref_rewards import RefRewardStore
from ref_rewards.generation import (
    attach_ref_rollout_metadata,
    extract_ref_rollout_metadata,
    pad_ref_rollout_input_for_agent_workers,
    ref_reward_rollout_input_from_prompt_records,
    ref_rollout_output_to_store_rows,
    truncate_ref_rollout_output,
)
from verl_adapters.engine_worker import QRPOEngineActorRolloutRefWorker
from verl_adapters.online_completion_logging import (
    build_online_completion_log_rows,
    enrich_online_completion_log_rows,
    log_online_completion_rows,
)
from verl_adapters.rollout import (
    attach_online_rollout_train_metadata,
    expand_online_rollout_candidate_requests,
    extract_online_rollout_train_metadata,
    online_rollout_output_to_train_dataproto,
    online_rollout_requests_to_dataproto,
    select_online_rollout_candidates,
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
        ref_reward_store: RefRewardStore | None = None,
        ref_version: str | None = None,
        **kwargs,
    ):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]

        validate_qrpo_trainer_config(config)

        self.source_scheduler = source_scheduler
        self.offline_selector = offline_selector
        self.ref_reward_store = ref_reward_store
        self.current_ref_version = ref_version
        self._qrpo_train_dataset = kwargs.get("train_dataset", None)

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
        self._check_ref_version_consistency(
            non_empty_batches,
            expected_ref_version=(
                None if meta_info is None else meta_info.get(K.REF_VERSION)
            ),
        )

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

        resolved_ref_rewards = all(
            isinstance(prompt, PromptRecord)
            for prompt in prompt_records
        )
        if resolved_ref_rewards:
            prompt_records = self._attach_current_ref_rewards(prompt_records)
            meta_info = dict(meta_info or {})
            meta_info.setdefault(K.REF_VERSION, self.current_ref_version)

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

            candidate_plan = expand_online_rollout_candidate_requests(
                requests=online_requests,
                config=online_rollout_config,
                meta_info=meta_info,
                divisible_by=self._resolve_agent_loop_num_workers(),
            )
            online_rollout_input = online_rollout_requests_to_dataproto(
                requests=list(candidate_plan.candidate_requests),
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

            online_rewards, _ = (
                self._compute_online_rewards_from_rollout_output(rollout_output)
            )
            (
                rollout_output,
                online_rewards,
                candidate_selection_metrics,
            ) = select_online_rollout_candidates(
                rollout_output=rollout_output,
                rewards=online_rewards,
                candidate_plan=candidate_plan,
            )
            online_reward_metrics = self._compute_online_reward_metrics(
                rollout_output=rollout_output,
                rewards=online_rewards,
            )
            online_reward_metrics.update(candidate_selection_metrics)
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
        self.prepare_initial_ref_rewards()

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
        ref_refresh_interval_steps = self._resolve_ref_refresh_interval_steps()

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
                prompt_records = self._attach_current_ref_rewards(prompt_records)
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
                    K.REF_VERSION: self.current_ref_version,
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

                if (
                    ref_refresh_interval_steps is not None
                    and not is_last_step
                    and self.global_steps % ref_refresh_interval_steps == 0
                ):
                    self.refresh_ref_rewards(actor_step=self.global_steps)
                    rollout_weights_synced = False
                    metrics["ref_rewards/refreshed"] = 1.0

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

    def _save_checkpoint(self) -> None:
        self._save_qrpo_checkpoint_state()
        super()._save_checkpoint()

    def _load_checkpoint(self):
        result = super()._load_checkpoint()
        self._load_qrpo_checkpoint_state()
        return result

    def prepare_initial_ref_rewards(self) -> None:
        """Generate and persist the initial ref-reward version when requested."""

        ref_config = _to_plain_container(
            _select(self.config, "ref_rewards", default={}) or {}
        )
        if ref_config.get("initial_source", "dataset") != "generate":
            self._load_current_ref_model_checkpoint_if_required()
            return

        if self.ref_reward_store is None:
            raise ValueError("QRPOTrainer requires ref_reward_store.")
        if not self.current_ref_version:
            raise ValueError("QRPOTrainer requires current_ref_version.")

        reuse_existing = bool(ref_config.get("reuse_existing", True))
        actor_step = 0

        train_dataset = getattr(self, "_qrpo_train_dataset", None)
        if train_dataset is None:
            raise ValueError(
                "ref_rewards.initial_source='generate' requires train_dataset."
            )
        if len(train_dataset) == 0:
            raise ValueError("Cannot generate ref rewards for an empty train_dataset.")

        num_ref_completions = int(ref_config.get("num_ref_completions", 0))
        if num_ref_completions <= 0:
            raise ValueError(
                "ref_rewards.num_ref_completions must be positive when "
                "ref_rewards.initial_source='generate'."
            )

        generation_prompt_batch_size = self._resolve_ref_generation_prompt_batch_size(
            ref_config=ref_config,
            dataset_size=len(train_dataset),
        )

        online_rollout_config = _select(self.config, "online_rollout", default=None)
        if online_rollout_config is None:
            raise ValueError(
                "ref_rewards.initial_source='generate' requires online_rollout config."
            )
        online_rollout_config = _to_plain_container(online_rollout_config)

        manifest = self._initial_ref_generation_manifest(
            train_dataset=train_dataset,
            online_rollout_config=online_rollout_config,
            ref_config=ref_config,
            num_ref_completions=num_ref_completions,
            actor_step=actor_step,
        )
        if int(getattr(self, "global_steps", 0)) != actor_step:
            if not self.ref_reward_store.has_complete_version(self.current_ref_version):
                raise ValueError(
                    "Cannot generate initial ref rewards after training has advanced. "
                    "Use an existing complete ref_rewards.initial_version in the "
                    "RefRewardStore, or restart from global step 0."
                )

            stored_manifest = self.ref_reward_store.load_manifest(
                self.current_ref_version
            )
            if stored_manifest.get("source") == "refresh":
                self._load_current_ref_model_checkpoint_if_required()
                return

            self.ref_reward_store.check_complete_version_metadata(
                ref_version=self.current_ref_version,
                metadata=manifest,
            )
            self._load_current_ref_model_checkpoint_if_required()
            return

        if (
            reuse_existing
            and self.ref_reward_store.has_complete_version(self.current_ref_version)
        ):
            self.ref_reward_store.check_complete_version_metadata(
                ref_version=self.current_ref_version,
                metadata=manifest,
            )
            self._load_current_ref_model_checkpoint_if_required()
            return

        if getattr(self, "async_rollout_manager", None) is None:
            raise ValueError(
                "ref_rewards.initial_source='generate' requires async_rollout_manager."
            )

        tokenizer = self._resolve_tokenizer(None)
        dataset_config = self._ref_generation_dataset_config()

        checkpoint_manager = getattr(self, "checkpoint_manager", None)
        if checkpoint_manager is not None:
            checkpoint_manager.update_weights(actor_step)

        rows = self._generate_ref_reward_rows_for_dataset(
            train_dataset=train_dataset,
            dataset_config=dataset_config,
            tokenizer=tokenizer,
            ref_version=self.current_ref_version,
            actor_step=actor_step,
            num_ref_completions=num_ref_completions,
            generation_prompt_batch_size=generation_prompt_batch_size,
            online_rollout_config=online_rollout_config,
            manifest=manifest,
            reuse_existing=reuse_existing,
            description="Generating Initial Ref Rewards",
        )

        self.ref_reward_store.save_version_rows(
            ref_version=self.current_ref_version,
            rows=rows,
            manifest=manifest,
            overwrite=not reuse_existing,
        )

    def refresh_ref_rewards(self, *, actor_step: int) -> str:
        """Refresh full-dataset ref rewards from the current actor."""

        ref_config = _to_plain_container(
            _select(self.config, "ref_rewards", default={}) or {}
        )
        if ref_config.get("refresh_scope", "full_dataset") != "full_dataset":
            raise NotImplementedError(
                "Only ref_rewards.refresh_scope='full_dataset' is implemented."
            )

        if self.ref_reward_store is None:
            raise ValueError("QRPOTrainer requires ref_reward_store.")
        if getattr(self, "async_rollout_manager", None) is None:
            raise ValueError("Ref reward refresh requires async_rollout_manager.")

        train_dataset = getattr(self, "_qrpo_train_dataset", None)
        if train_dataset is None:
            raise ValueError("Ref reward refresh requires train_dataset.")
        if len(train_dataset) == 0:
            raise ValueError("Cannot refresh ref rewards for an empty train_dataset.")

        num_ref_completions = int(ref_config.get("num_ref_completions", 0))
        if num_ref_completions <= 0:
            raise ValueError("ref_rewards.num_ref_completions must be positive.")

        generation_prompt_batch_size = self._resolve_ref_generation_prompt_batch_size(
            ref_config=ref_config,
            dataset_size=len(train_dataset),
        )

        online_rollout_config = _select(self.config, "online_rollout", default=None)
        if online_rollout_config is None:
            raise ValueError("Ref reward refresh requires online_rollout config.")
        online_rollout_config = _to_plain_container(online_rollout_config)

        ref_version = _ref_version_for_step(actor_step)
        manifest = self._initial_ref_generation_manifest(
            train_dataset=train_dataset,
            online_rollout_config=online_rollout_config,
            ref_config=ref_config,
            num_ref_completions=num_ref_completions,
            actor_step=actor_step,
            source="refresh",
        )
        reuse_existing = bool(ref_config.get("reuse_existing", True))

        if (
            reuse_existing
            and self.ref_reward_store.has_complete_version(ref_version)
        ):
            self.ref_reward_store.check_complete_version_metadata(
                ref_version=ref_version,
                metadata=manifest,
            )
            checkpoint_path = self.ref_reward_store.model_checkpoint_path(ref_version)
            if os.path.exists(checkpoint_path):
                self._load_ref_model_checkpoint(ref_version=ref_version)
            else:
                self._check_ref_model_checkpoint_not_required(
                    ref_version=ref_version,
                    checkpoint_path=checkpoint_path,
                )
            self.current_ref_version = ref_version
            return ref_version

        checkpoint_manager = getattr(self, "checkpoint_manager", None)
        if checkpoint_manager is not None:
            checkpoint_manager.update_weights(actor_step)

        rows = self._generate_ref_reward_rows_for_dataset(
            train_dataset=train_dataset,
            dataset_config=self._ref_generation_dataset_config(),
            tokenizer=self._resolve_tokenizer(None),
            ref_version=ref_version,
            actor_step=actor_step,
            num_ref_completions=num_ref_completions,
            generation_prompt_batch_size=generation_prompt_batch_size,
            online_rollout_config=online_rollout_config,
            manifest=manifest,
            reuse_existing=reuse_existing,
            description="Refreshing Ref Rewards",
        )

        ref_model_checkpoint_path = self._sync_ref_model_from_actor(
            ref_version=ref_version,
            global_step=actor_step,
        )
        manifest["ref_model_checkpoint_path"] = ref_model_checkpoint_path

        self.ref_reward_store.save_version_rows(
            ref_version=ref_version,
            rows=rows,
            manifest=manifest,
            overwrite=not reuse_existing,
        )
        self.current_ref_version = ref_version
        return ref_version

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

        return rewards, self._compute_online_reward_metrics(
            rollout_output=rollout_output,
            rewards=rewards,
        )

    def _compute_online_reward_metrics(
        self,
        *,
        rollout_output: DataProto,
        rewards: torch.Tensor,
    ) -> dict[str, float]:
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

        return metrics

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

    def _resolve_agent_loop_num_workers(self) -> int:
        value = _select(
            self.config,
            "actor_rollout_ref.rollout.agent.num_workers",
            default=1,
        )
        return max(1, int(value or 1))

    def _resolve_ref_refresh_interval_steps(self) -> int | None:
        value = _select(
            self.config,
            "ref_rewards.refresh_interval_epochs",
            default=None,
        )
        if value is None:
            return None

        interval_epochs = float(value)
        if interval_epochs <= 0.0:
            raise ValueError("ref_rewards.refresh_interval_epochs must be positive.")

        steps_per_epoch = len(self.train_dataloader)
        if steps_per_epoch <= 0:
            raise ValueError("Cannot refresh ref rewards with an empty train_dataloader.")

        return max(1, int(math.ceil(interval_epochs * steps_per_epoch)))

    def _ref_generation_dataset_config(self) -> dict[str, Any]:
        config = dict(
            _to_plain_container(_select(self.config, "data", default={}) or {})
        )
        config["ref_rewards_key"] = None
        return config

    def _initial_ref_generation_manifest(
        self,
        *,
        train_dataset: Any,
        online_rollout_config: Mapping[str, Any],
        ref_config: Mapping[str, Any],
        num_ref_completions: int,
        actor_step: int,
        source: str = "generate",
    ) -> dict[str, Any]:
        reward_function_path = _select(
            self.config,
            "reward.custom_reward_function.path",
            default=None,
        )
        manifest_env_keys = tuple(ref_config.get("manifest_env_keys", ()) or ())

        manifest = {
            "source": source,
            "actor_step": int(actor_step),
            "prompt_count": len(train_dataset),
            "num_ref_completions": int(num_ref_completions),
            "data_path": _select(self.config, "data.path", default=None),
            "dataset_fingerprint": getattr(train_dataset, "_fingerprint", None),
            "data_source": online_rollout_config.get("data_source"),
            "actor_model_path": _select(
                self.config,
                "actor_rollout_ref.model.path",
                default=None,
            ),
            "max_prompt_length": _select(
                self.config,
                "data.max_prompt_length",
                default=None,
            ),
            "max_response_length": _select(
                self.config,
                "data.max_response_length",
                default=None,
            ),
            "rollout_temperature": _select(
                self.config,
                "actor_rollout_ref.rollout.temperature",
                default=None,
            ),
            "rollout_top_k": _select(
                self.config,
                "actor_rollout_ref.rollout.top_k",
                default=None,
            ),
            "rollout_top_p": _select(
                self.config,
                "actor_rollout_ref.rollout.top_p",
                default=None,
            ),
            "rollout_do_sample": _select(
                self.config,
                "actor_rollout_ref.rollout.do_sample",
                default=None,
            ),
            "reward_function_path": reward_function_path,
            "reward_function_sha256": _file_sha256(reward_function_path),
            "reward_function_name": _select(
                self.config,
                "reward.custom_reward_function.name",
                default=None,
            ),
        }
        for env_key in manifest_env_keys:
            env_key = str(env_key)
            env_value = os.environ.get(env_key)
            manifest[f"env/{env_key}"] = env_value
            manifest[f"env/{env_key}_sha256"] = _file_sha256(env_value)

        return manifest

    def _generate_ref_reward_rows_for_dataset(
        self,
        *,
        train_dataset: Any,
        dataset_config: Mapping[str, Any],
        tokenizer: Any,
        ref_version: str,
        actor_step: int,
        num_ref_completions: int,
        generation_prompt_batch_size: int,
        online_rollout_config: Mapping[str, Any],
        manifest: Mapping[str, Any],
        reuse_existing: bool,
        description: str,
    ) -> list[dict[str, Any]]:
        if self.ref_reward_store is None:
            raise ValueError("Ref reward generation requires ref_reward_store.")

        rows: list[dict[str, Any]] = []
        starts = range(0, len(train_dataset), generation_prompt_batch_size)
        for chunk_index, start in enumerate(
            tqdm(
                starts,
                desc=description,
            ),
        ):
            indices = list(
                range(
                    start,
                    min(start + generation_prompt_batch_size, len(train_dataset)),
                )
            )

            cached_rows = None
            if reuse_existing:
                cached_rows = self.ref_reward_store.load_generation_chunk(
                    ref_version=ref_version,
                    chunk_index=chunk_index,
                    dataset_indices=indices,
                    manifest=manifest,
                )
            if cached_rows is not None:
                rows.extend(cached_rows)
                continue

            chunk_rows = self._generate_ref_reward_rows_for_indices(
                train_dataset=train_dataset,
                dataset_config=dataset_config,
                tokenizer=tokenizer,
                ref_version=ref_version,
                actor_step=actor_step,
                num_ref_completions=num_ref_completions,
                online_rollout_config=online_rollout_config,
                indices=indices,
            )
            self.ref_reward_store.save_generation_chunk(
                ref_version=ref_version,
                chunk_index=chunk_index,
                dataset_indices=indices,
                rows=chunk_rows,
                manifest=manifest,
                overwrite=not reuse_existing,
            )
            rows.extend(chunk_rows)

        self._sleep_rollout_replicas_after_generation()
        self._check_generated_ref_rows_cover_dataset(
            rows=rows,
            dataset_size=len(train_dataset),
        )
        return rows

    def _generate_ref_reward_rows_for_indices(
        self,
        *,
        train_dataset: Any,
        dataset_config: Mapping[str, Any],
        tokenizer: Any,
        ref_version: str,
        actor_step: int,
        num_ref_completions: int,
        online_rollout_config: Mapping[str, Any],
        indices: Sequence[int],
    ) -> list[dict[str, Any]]:
        prompt_records = dataset_batch_to_prompt_records(
            train_dataset,
            indices=indices,
            config=dataset_config,
        )
        rollout_input = ref_reward_rollout_input_from_prompt_records(
            prompt_records=prompt_records,
            ref_version=ref_version,
            num_ref_completions=num_ref_completions,
            config=online_rollout_config,
            dataset_indices=indices,
            meta_info={
                "global_steps": actor_step,
            },
        )
        rollout_metadata = extract_ref_rollout_metadata(rollout_input)
        rollout_input, original_size = pad_ref_rollout_input_for_agent_workers(
            rollout_input,
            num_workers=self._agent_loop_worker_count(),
        )
        rollout_output = self.async_rollout_manager.generate_sequences(
            rollout_input
        )
        rollout_output = truncate_ref_rollout_output(
            rollout_output,
            size=original_size,
        )
        attach_ref_rollout_metadata(
            rollout_output=rollout_output,
            metadata=rollout_metadata,
        )
        return ref_rollout_output_to_store_rows(
            rollout_output=rollout_output,
            tokenizer=tokenizer,
            ref_version=ref_version,
            num_ref_completions=num_ref_completions,
        )

    def _agent_loop_worker_count(self) -> int:
        workers = getattr(self.async_rollout_manager, "agent_loop_workers", None)
        if workers is not None:
            return len(workers)
        return int(
            _select(
                self.config,
                "actor_rollout_ref.rollout.agent.num_workers",
                default=1,
            )
            or 1
        )

    @staticmethod
    def _resolve_ref_generation_prompt_batch_size(
        *,
        ref_config: Mapping[str, Any],
        dataset_size: int,
    ) -> int:
        generation_num_chunks = ref_config.get("generation_num_chunks", None)
        if generation_num_chunks is not None:
            generation_num_chunks = int(generation_num_chunks)
            if generation_num_chunks <= 0:
                raise ValueError("ref_rewards.generation_num_chunks must be positive.")
            return max(1, math.ceil(dataset_size / generation_num_chunks))

        generation_prompt_batch_size = int(
            ref_config.get("generation_prompt_batch_size")
            or ref_config.get("generation_batch_size")
            or dataset_size
        )
        if generation_prompt_batch_size <= 0:
            raise ValueError(
                "ref_rewards.generation_prompt_batch_size must be positive."
            )
        return generation_prompt_batch_size

    @staticmethod
    def _check_generated_ref_rows_cover_dataset(
        *,
        rows: Sequence[Mapping[str, Any]],
        dataset_size: int,
    ) -> None:
        if len(rows) != dataset_size:
            raise ValueError(
                f"Generated {len(rows)} ref reward rows for dataset_size="
                f"{dataset_size}."
            )

        observed_list = [int(row[K.DATASET_INDEX]) for row in rows]
        if len(set(observed_list)) != len(observed_list):
            raise ValueError(
                "Generated ref rewards contain duplicate dataset indices: "
                f"{observed_list}."
            )

        observed_indices = {int(row[K.DATASET_INDEX]) for row in rows}
        expected_indices = set(range(dataset_size))
        if observed_indices != expected_indices:
            missing = sorted(expected_indices - observed_indices)
            extra = sorted(observed_indices - expected_indices)
            raise ValueError(
                "Generated ref rewards do not cover the training dataset exactly. "
                f"missing dataset indices={missing}, extra dataset indices={extra}."
            )

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

    def _sync_ref_model_from_actor(
        self,
        *,
        ref_version: str,
        global_step: int,
    ) -> str:
        if self.ref_reward_store is None:
            raise ValueError("QRPOTrainer requires ref_reward_store.")

        actor_rollout_wg = getattr(self, "actor_rollout_wg", None)
        if actor_rollout_wg is None:
            raise ValueError("QRPOTrainer requires actor_rollout_wg.")

        sync_ref = getattr(actor_rollout_wg, "sync_ref_model_from_actor", None)
        if sync_ref is None:
            raise RuntimeError(
                "QRPO actor_rollout_wg does not support sync_ref_model_from_actor."
            )

        checkpoint_path = str(self.ref_reward_store.model_checkpoint_path(ref_version))
        sync_ref(
            local_path=checkpoint_path,
            global_step=int(global_step),
        )
        return checkpoint_path

    def _load_ref_model_checkpoint(self, *, ref_version: str) -> str:
        if self.ref_reward_store is None:
            raise ValueError("QRPOTrainer requires ref_reward_store.")

        actor_rollout_wg = getattr(self, "actor_rollout_wg", None)
        if actor_rollout_wg is None:
            raise ValueError("QRPOTrainer requires actor_rollout_wg.")

        load_ref = getattr(actor_rollout_wg, "load_ref_model_checkpoint", None)
        if load_ref is None:
            raise RuntimeError(
                "QRPO actor_rollout_wg does not support load_ref_model_checkpoint."
            )

        checkpoint_path = str(self.ref_reward_store.model_checkpoint_path(ref_version))
        load_ref(local_path=checkpoint_path, del_local_after_load=False)
        return checkpoint_path

    def _save_qrpo_checkpoint_state(self) -> None:
        state_path = self._qrpo_checkpoint_state_path(self.global_steps)
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    K.REF_VERSION: self.current_ref_version,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

    def _load_qrpo_checkpoint_state(self) -> None:
        global_steps = int(getattr(self, "global_steps", 0))
        if global_steps <= 0:
            return

        state_path = self._qrpo_checkpoint_state_path(global_steps)
        if not os.path.exists(state_path):
            return

        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)

        ref_version = state.get(K.REF_VERSION)
        if not ref_version:
            raise ValueError(
                f"QRPO checkpoint state {state_path} is missing "
                f"{K.REF_VERSION!r}."
            )

        self.current_ref_version = str(ref_version)

        if self.ref_reward_store is None:
            return

        checkpoint_path = self.ref_reward_store.model_checkpoint_path(
            self.current_ref_version
        )
        if os.path.exists(checkpoint_path):
            self._load_ref_model_checkpoint(ref_version=self.current_ref_version)
            return

        self._check_ref_model_checkpoint_not_required(
            ref_version=self.current_ref_version,
            checkpoint_path=checkpoint_path,
        )

    def _load_current_ref_model_checkpoint_if_required(self) -> None:
        if self.ref_reward_store is None or not self.current_ref_version:
            return

        checkpoint_path = self.ref_reward_store.model_checkpoint_path(
            self.current_ref_version
        )
        if os.path.exists(checkpoint_path):
            self._load_ref_model_checkpoint(ref_version=self.current_ref_version)
            return

        self._check_ref_model_checkpoint_not_required(
            ref_version=self.current_ref_version,
            checkpoint_path=checkpoint_path,
        )

    def _check_ref_model_checkpoint_not_required(
        self,
        *,
        ref_version: str,
        checkpoint_path: os.PathLike[str] | str,
    ) -> None:
        if self.ref_reward_store is None:
            return

        manifest = self.ref_reward_store.load_manifest(ref_version)
        if manifest.get("source") != "refresh":
            return

        raise FileNotFoundError(
            f"Ref reward version {ref_version!r} was produced by a ref refresh, "
            f"but its ref model checkpoint is missing at {checkpoint_path!s}."
        )

    def _qrpo_checkpoint_state_path(self, global_steps: int) -> str:
        return os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{int(global_steps)}",
            "qrpo_state.json",
        )

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
    def _check_ref_version_consistency(
        batches: Sequence[DataProto],
        *,
        expected_ref_version: Any | None,
    ) -> None:
        observed_versions = [
            batch.meta_info.get(K.REF_VERSION)
            for batch in batches
            if batch.meta_info.get(K.REF_VERSION) is not None
        ]

        if expected_ref_version is not None:
            missing_count = len(batches) - len(observed_versions)
            if missing_count:
                raise ValueError(
                    f"{missing_count} QRPO train batch(es) are missing "
                    f"meta_info[{K.REF_VERSION!r}] while expected ref version is "
                    f"{expected_ref_version!r}."
                )

            mismatched = [
                version
                for version in observed_versions
                if version != expected_ref_version
            ]
            if mismatched:
                raise ValueError(
                    f"QRPO train batch ref versions {mismatched} do not match "
                    f"expected ref version {expected_ref_version!r}."
                )
            return

        if observed_versions and len(observed_versions) != len(batches):
            raise ValueError(
                f"QRPO train batches must either all define {K.REF_VERSION!r} "
                "or all omit it."
            )

        if observed_versions and len(set(observed_versions)) != 1:
            raise ValueError(
                f"QRPO train batches must agree on {K.REF_VERSION!r}, got "
                f"{observed_versions}."
            )

    def _attach_current_ref_rewards(
        self,
        prompt_records: Sequence[PromptRecord],
    ) -> list[PromptRecord]:
        if self.ref_reward_store is None:
            raise ValueError("QRPOTrainer requires ref_reward_store.")
        if not self.current_ref_version:
            raise ValueError("QRPOTrainer requires current_ref_version.")

        prompt_records = list(prompt_records)
        if all(
            isinstance(prompt, PromptRecord)
            and prompt.metadata.get(K.REF_VERSION) == self.current_ref_version
            for prompt in prompt_records
        ):
            return list(prompt_records)

        return self.ref_reward_store.attach_ref_rewards(
            prompt_records,
            ref_version=self.current_ref_version,
        )

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

    effective_beta_max = _select(config, "qrpo.effective_beta_max", default=None)
    if effective_beta_max is not None:
        effective_beta_max = float(effective_beta_max)
        if not math.isfinite(effective_beta_max) or effective_beta_max <= 0.0:
            raise ValueError(
                "qrpo.effective_beta_max must be finite and positive when set."
            )

    transform = _select(config, "qrpo.transform", default="identity")
    if transform != "identity":
        raise ValueError("Only qrpo.transform='identity' is implemented for now.")

    rollout_n = int(_select(config, "actor_rollout_ref.rollout.n", default=1) or 1)
    if rollout_n != 1:
        raise ValueError(
            "QRPO expects actor_rollout_ref.rollout.n=1 because it expands "
            "online and ref-reward requests explicitly."
        )

    ref_initial_source = _select(
        config,
        "ref_rewards.initial_source",
        default="dataset",
    )
    if ref_initial_source not in {"dataset", "store", "generate"}:
        raise ValueError(
            "ref_rewards.initial_source must be one of 'dataset', 'store', or "
            f"'generate', got {ref_initial_source!r}."
        )

    ref_refresh_scope = _select(
        config,
        "ref_rewards.refresh_scope",
        default="full_dataset",
    )
    if ref_refresh_scope != "full_dataset":
        raise NotImplementedError(
            "Only ref_rewards.refresh_scope='full_dataset' is implemented for now."
        )

    ref_refresh_interval_epochs = _select(
        config,
        "ref_rewards.refresh_interval_epochs",
        default=None,
    )
    if (
        ref_refresh_interval_epochs is not None
        and float(ref_refresh_interval_epochs) <= 0.0
    ):
        raise ValueError("ref_rewards.refresh_interval_epochs must be positive.")

    n_online = int(
        _select(config, "source_schedule.n_online", default=0)
        or _select(config, "qrpo.source_schedule.n_online", default=0)
        or 0
    )

    if (
        n_online > 0
        or ref_initial_source == "generate"
        or ref_refresh_interval_epochs is not None
    ):
        reward_num_workers = int(_select(config, "reward.num_workers", default=0))
        if reward_num_workers <= 0:
            raise ValueError(
                "Online QRPO and generated/refreshed ref rewards require "
                "reward.num_workers > 0 so AgentLoop computes reward during rollout."
            )

        reward_path = _select(config, "reward.custom_reward_function.path", default=None)
        if reward_path is None:
            raise ValueError(
                "Online QRPO and generated/refreshed ref rewards require "
                "reward.custom_reward_function.path."
            )

        reward_name = _select(config, "reward.custom_reward_function.name", default=None)
        if not reward_name:
            raise ValueError(
                "Online QRPO and generated/refreshed ref rewards require "
                "reward.custom_reward_function.name."
            )

    candidate_selection = _select(
        config,
        "online_rollout.candidate_selection",
        default={},
    ) or {}
    candidate_enabled = bool(_select(candidate_selection, "enabled", default=False))
    candidate_count = _select(
        candidate_selection,
        "candidates_per_train_sample",
        default=None,
    )
    candidate_num = int(1 if candidate_count is None else candidate_count)
    candidate_probability = float(
        _select(candidate_selection, "probability", default=1.0)
    )
    candidate_selection_name = str(
        _select(candidate_selection, "selection", default="best_reward")
    )
    if candidate_num <= 0:
        raise ValueError(
            "online_rollout.candidate_selection.candidates_per_train_sample "
            "must be positive."
        )
    if not 0.0 <= candidate_probability <= 1.0:
        raise ValueError(
            "online_rollout.candidate_selection.probability must be in [0, 1]."
        )
    if candidate_selection_name != "best_reward":
        raise ValueError(
            "online_rollout.candidate_selection.selection must be 'best_reward'."
        )
    if candidate_enabled and n_online <= 0:
        raise ValueError(
            "online_rollout.candidate_selection requires source_schedule.n_online > 0."
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


def _file_sha256(path: Any | None) -> str | None:
    if not path:
        return None

    path = os.fspath(path)
    if not os.path.isfile(path):
        return None

    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _ref_version_for_step(global_step: int) -> str:
    return f"ref_step_{int(global_step):06d}"
