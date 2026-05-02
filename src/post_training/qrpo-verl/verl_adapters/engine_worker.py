from __future__ import annotations

from functools import partial

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