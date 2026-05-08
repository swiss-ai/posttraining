from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from typing import Iterable

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker

from verl_adapters.engine_loss import qrpo_engine_loss


class QRPOEngineActorRolloutRefWorker(ActorRolloutRefWorker):
    """New-stack VERL Actor/Rollout/Ref worker with QRPO actor loss.

    This worker uses VERL 0.7.1's model-engine stack and keeps the standard:

      - TrainingWorker construction
      - FSDP/FSDP2/Megatron engine selection
      - rollout engine
      - reference model
      - compute_ref_log_prob(...)
      - update_actor(...)
      - checkpointing / weight sync

    It replaces only the actor loss function after the base worker initializes.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()

        if not getattr(self, "_is_actor", False):
            return

        self.loss_fn = partial(qrpo_engine_loss, config=self.config.actor)
        self.actor.set_loss_fn(self.loss_fn)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_actor_model_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
    ):
        """Save only actor model shards.

        QRPO uses this for dynamic ref refresh. The actor training checkpoint
        still keeps optimizer/extra state; ref refresh needs only model weights.
        """

        if not getattr(self, "_is_actor", False) or self.actor is None:
            raise RuntimeError("save_actor_model_checkpoint requires actor role.")

        with _temporary_checkpoint_contents(
            self.actor,
            save_contents=("model",),
        ):
            return self.actor.save_checkpoint(
                local_path=local_path,
                hdfs_path=hdfs_path,
                global_step=global_step,
                max_ckpt_to_keep=None,
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_ref_model_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        del_local_after_load: bool = False,
    ):
        """Load only ref model shards from a QRPO ref-model checkpoint."""

        if not getattr(self, "_is_ref", False) or self.ref is None:
            raise RuntimeError("load_ref_model_checkpoint requires ref role.")

        with _temporary_checkpoint_contents(
            self.ref,
            load_contents=("model",),
        ):
            return self.ref.load_checkpoint(
                local_path=local_path,
                hdfs_path=hdfs_path,
                del_local_after_load=del_local_after_load,
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sync_ref_model_from_actor(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
    ):
        """Persist current actor weights and load them into the ref model."""

        if not local_path:
            raise ValueError("local_path is required for ref-model sync.")

        self.save_actor_model_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
        )
        return self.load_ref_model_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            del_local_after_load=False,
        )


@contextmanager
def _temporary_checkpoint_contents(
    training_worker,
    *,
    save_contents: Iterable[str] | None = None,
    load_contents: Iterable[str] | None = None,
):
    checkpoint_manager = getattr(
        getattr(training_worker, "engine", None),
        "checkpoint_manager",
        None,
    )
    if checkpoint_manager is None:
        raise RuntimeError("Training worker engine has no checkpoint_manager.")

    previous_save_contents = checkpoint_manager.checkpoint_save_contents
    previous_load_contents = checkpoint_manager.checkpoint_load_contents
    previous_global_step = getattr(checkpoint_manager, "previous_global_step", None)
    previous_saved_paths = list(getattr(checkpoint_manager, "previous_saved_paths", []))
    try:
        if save_contents is not None:
            checkpoint_manager.checkpoint_save_contents = list(save_contents)
        if load_contents is not None:
            checkpoint_manager.checkpoint_load_contents = list(load_contents)
        yield
    finally:
        checkpoint_manager.checkpoint_save_contents = previous_save_contents
        checkpoint_manager.checkpoint_load_contents = previous_load_contents
        checkpoint_manager.previous_global_step = previous_global_step
        checkpoint_manager.previous_saved_paths = previous_saved_paths
