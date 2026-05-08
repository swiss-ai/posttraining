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


class FakeCheckpointManager:
    def __init__(self) -> None:
        self.checkpoint_save_contents = ["model", "optimizer", "extra"]
        self.checkpoint_load_contents = ["model", "optimizer", "extra"]
        self.previous_global_step = 5
        self.previous_saved_paths = ["/tmp/actor-ckpt"]


class FakeCheckpointEngine:
    def __init__(self) -> None:
        self.checkpoint_manager = FakeCheckpointManager()


class FakeCheckpointTrainingWorker:
    def __init__(self) -> None:
        self.engine = FakeCheckpointEngine()
        self.save_calls = []
        self.load_calls = []

    def save_checkpoint(
        self,
        local_path,
        hdfs_path=None,
        global_step=0,
        max_ckpt_to_keep=None,
    ):
        manager = self.engine.checkpoint_manager
        manager.previous_global_step = global_step
        manager.previous_saved_paths.append(local_path)
        self.save_calls.append(
            {
                "local_path": local_path,
                "hdfs_path": hdfs_path,
                "global_step": global_step,
                "max_ckpt_to_keep": max_ckpt_to_keep,
                "save_contents": list(
                    self.engine.checkpoint_manager.checkpoint_save_contents
                ),
            }
        )

    def load_checkpoint(
        self,
        local_path,
        hdfs_path=None,
        del_local_after_load=False,
    ):
        self.load_calls.append(
            {
                "local_path": local_path,
                "hdfs_path": hdfs_path,
                "del_local_after_load": del_local_after_load,
                "load_contents": list(
                    self.engine.checkpoint_manager.checkpoint_load_contents
                ),
            }
        )


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


def test_qrpo_engine_worker_syncs_ref_from_actor_with_model_only_checkpoint() -> None:
    worker = QRPOEngineActorRolloutRefWorker.__new__(QRPOEngineActorRolloutRefWorker)
    worker._is_actor = True
    worker._is_ref = True
    worker.actor = FakeCheckpointTrainingWorker()
    worker.ref = FakeCheckpointTrainingWorker()

    worker.sync_ref_model_from_actor(
        local_path="/tmp/ref-model",
        hdfs_path=None,
        global_step=17,
    )

    assert worker.actor.save_calls == [
        {
            "local_path": "/tmp/ref-model",
            "hdfs_path": None,
            "global_step": 17,
            "max_ckpt_to_keep": None,
            "save_contents": ["model"],
        }
    ]
    assert worker.ref.load_calls == [
        {
            "local_path": "/tmp/ref-model",
            "hdfs_path": None,
            "del_local_after_load": False,
            "load_contents": ["model"],
        }
    ]
    assert worker.actor.engine.checkpoint_manager.checkpoint_save_contents == [
        "model",
        "optimizer",
        "extra",
    ]
    assert worker.ref.engine.checkpoint_manager.checkpoint_load_contents == [
        "model",
        "optimizer",
        "extra",
    ]
    assert worker.actor.engine.checkpoint_manager.previous_global_step == 5
    assert worker.actor.engine.checkpoint_manager.previous_saved_paths == [
        "/tmp/actor-ckpt",
    ]
