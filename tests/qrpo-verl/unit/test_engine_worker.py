from functools import partial

from verl_adapters import engine_worker as engine_worker_module
from verl_adapters.engine_loss import qrpo_engine_loss
from verl_adapters.engine_worker import QRPOEngineActorRolloutRefWorker


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class FakeTrainingWorker:
    def __init__(self) -> None:
        self.loss_fn = None

    def set_loss_fn(self, loss_fn) -> None:
        self.loss_fn = loss_fn


def test_qrpo_engine_worker_replaces_actor_loss_fn(monkeypatch) -> None:
    def fake_base_init_model(self):
        self._is_actor = True
        self.config = AttrDict(
            {
                "actor": AttrDict(
                    {
                        "use_kl_loss": False,
                    }
                )
            }
        )
        self.actor = FakeTrainingWorker()

    monkeypatch.setattr(
        engine_worker_module.ActorRolloutRefWorker,
        "init_model",
        fake_base_init_model,
    )

    worker = QRPOEngineActorRolloutRefWorker.__new__(QRPOEngineActorRolloutRefWorker)

    worker.init_model()

    assert isinstance(worker.actor.loss_fn, partial)
    assert worker.actor.loss_fn.func is qrpo_engine_loss
    assert worker.actor.loss_fn.keywords["config"] is worker.config.actor


def test_qrpo_engine_worker_does_not_set_loss_when_not_actor(monkeypatch) -> None:
    def fake_base_init_model(self):
        self._is_actor = False
        self.config = AttrDict(
            {
                "actor": AttrDict(
                    {
                        "use_kl_loss": False,
                    }
                )
            }
        )
        self.actor = FakeTrainingWorker()

    monkeypatch.setattr(
        engine_worker_module.ActorRolloutRefWorker,
        "init_model",
        fake_base_init_model,
    )

    worker = QRPOEngineActorRolloutRefWorker.__new__(QRPOEngineActorRolloutRefWorker)

    worker.init_model()

    assert worker.actor.loss_fn is None
