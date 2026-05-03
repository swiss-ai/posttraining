from __future__ import annotations

import socket
from collections.abc import Mapping
from functools import partial
from pprint import pprint
from typing import Any

import hydra
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import SequentialSampler
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local

from batch.source_schedule import FixedCountsSourceScheduler
from data.dataset_adapter import load_hf_dataset_from_config, rows_to_prompt_records
from data.offline_selector import build_offline_selector
from trainers.qrpo_trainer import (
    GLOBAL_POOL_ID,
    QRPOTrainer,
    build_qrpo_resource_pool_mapping,
    build_qrpo_role_worker_mapping,
)


@hydra.main(
    config_path="../configs",
    config_name="qrpo",
    version_base=None,
)
def main(config: DictConfig) -> None:
    auto_set_device(config)
    run_qrpo(config)


def run_qrpo(config: DictConfig) -> None:
    """Run QRPO training.

    Offline-only is selected by config:

        source_schedule.n_online=0
        source_schedule.n_offline>0

    Online training will use the same entrypoint once online reward integration
    is implemented.
    """

    print(f"QRPO runner hostname: {socket.gethostname()}")
    pprint(OmegaConf.to_container(config, resolve=True))

    OmegaConf.resolve(config)

    _init_ray(config)

    local_model_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )

    trust_remote_code = bool(config.data.get("trust_remote_code", False))

    tokenizer = hf_tokenizer(
        local_model_path,
        trust_remote_code=trust_remote_code,
    )
    processor = hf_processor(
        local_model_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    _ensure_tokenizer_has_pad_token(tokenizer, config)

    train_dataset = load_hf_dataset_from_config(config.data)
    val_dataset = _load_validation_dataset_or_train_subset(
        train_dataset=train_dataset,
        config=config,
    )

    collate_fn = partial(
        _collate_prompt_records,
        dataset_config=OmegaConf.to_container(config.data, resolve=True),
    )

    train_sampler = _build_train_sampler(config, train_dataset)

    role_worker_mapping = build_qrpo_role_worker_mapping(use_ray_remote=True)
    resource_pool_mapping = build_qrpo_resource_pool_mapping()

    resource_pool_manager = _build_resource_pool_manager(
        config=config,
        resource_pool_mapping=resource_pool_mapping,
    )

    source_scheduler = FixedCountsSourceScheduler.from_config(
        OmegaConf.to_container(config.source_schedule, resolve=True)
    )

    offline_selector = build_offline_selector(
        OmegaConf.to_container(config.offline_selector, resolve=True)
    )

    online_reward_fn = _build_online_reward_fn(config)

    trainer = QRPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
        source_scheduler=source_scheduler,
        offline_selector=offline_selector,
        online_reward_fn=online_reward_fn,
    )

    trainer.init_workers()
    trainer.fit()


def _init_ray(config: DictConfig) -> None:
    if ray.is_initialized():
        return

    default_runtime_env = get_ppo_ray_runtime_env()

    ray_init_kwargs = config.get("ray_kwargs", {}).get("ray_init", {})
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)

    ray_init_kwargs = OmegaConf.create(
        {
            **ray_init_kwargs,
            "runtime_env": runtime_env,
        }
    )

    print(f"ray init kwargs: {ray_init_kwargs}")
    ray.init(**OmegaConf.to_container(ray_init_kwargs, resolve=True))


def _ensure_tokenizer_has_pad_token(tokenizer: Any, config: DictConfig) -> None:
    if tokenizer.pad_token_id is not None:
        return

    use_eos_as_pad = bool(config.get("tokenizer", {}).get("use_eos_as_pad", False))
    if not use_eos_as_pad:
        raise ValueError(
            "tokenizer.pad_token_id is None. Set tokenizer.use_eos_as_pad=true "
            "explicitly if you want to use eos_token as pad_token."
        )

    if tokenizer.eos_token is None:
        raise ValueError(
            "Cannot use eos_token as pad_token because tokenizer.eos_token is None."
        )

    tokenizer.pad_token = tokenizer.eos_token


def _load_validation_dataset_or_train_subset(*, train_dataset, config: DictConfig):
    val_config = config.get("val_data", None)
    if val_config is not None and val_config.get("path", None) is not None:
        return load_hf_dataset_from_config(val_config)

    val_size = int(config.data.get("val_subset_size", config.data.get("val_batch_size", 1)))
    val_size = max(1, min(val_size, len(train_dataset)))

    return train_dataset.select(range(val_size))


def _collate_prompt_records(
    rows: list[Mapping[str, Any]],
    *,
    dataset_config: Mapping[str, Any],
):
    return rows_to_prompt_records(rows, dataset_config)


def _build_train_sampler(config: DictConfig, train_dataset):
    if bool(config.data.get("shuffle", False)):
        from torchdata.stateful_dataloader.sampler import RandomSampler

        generator = torch.Generator()
        seed = config.data.get("seed", None)
        if seed is not None:
            generator.manual_seed(int(seed))

        return RandomSampler(
            data_source=train_dataset,
            generator=generator,
        )

    return SequentialSampler(train_dataset)


def _build_resource_pool_manager(
    *,
    config: DictConfig,
    resource_pool_mapping: dict,
) -> ResourcePoolManager:
    resource_pool_spec = {
        GLOBAL_POOL_ID: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }

    return ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=resource_pool_mapping,
    )


def _build_online_reward_fn(config: DictConfig):
    n_online = int(config.source_schedule.get("n_online", 0))

    if n_online == 0:
        return None

    raise NotImplementedError(
        "Online QRPO reward integration is not implemented yet. "
        "For offline-only training, set source_schedule.n_online=0."
    )


if __name__ == "__main__":
    main()