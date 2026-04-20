from collections import defaultdict
from datetime import datetime
from pathlib import Path

"""Nomenclature:

dataset = f"{dataset}"
model = f"{sft_model}"
model_sftid = f"{model}-(sftid)"
reward_model = f"{reward_model}"

dataset_for_model = f"{dataset}-{model}-(sft_id)-maxlen{max_seq_len}"
# dataset_for_model depends on the SFTid through the tokenizer and chat template to filter by max_seq_len

dataset_with_ref_completions = f"{dataset}-{model}-(sftid)-maxlen{max_seq_len}-Nref{NRefDataset}"

dataset_with_ref_logprobs = f"{dataset}-{model}-(sftid)-maxlen{max_seq_len}-Nref{NRefDataset}-logprobs"

dataset_with_ref_rewards = f"{dataset}-{model}-(sftid)-maxlen{max_seq_len}-Nref{NRefDataset}-logprobs-{reward_model}"

train_dataset = f"{dataset}-{model}-(sftid)-maxlen{max_seq_len}-Nref{NRefDataset}-logprobs-{reward_model}-(train_id)"
"""

stdout_prefix = "init"
stdout_root = (
    Path(__file__).parent.resolve()
    / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

job_name = "apertus-first-sweep"

datasets = ["swissai-olmo2-32b-preference"]
train_dataset_paths = [
    # "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/processing_for_alignment/datasets/ahey/MaxMin_3600-Filtered",
    "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/processing_for_alignment/datasets/ahey/MaxMin_4096-Filtered",
    "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/processing_for_alignment/datasets/ahey/MaxMin_Tr_3600-Filtered",
    # "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/processing_for_alignment/datasets/ahey/MaxMin_Tr_4096-Filtered",
    # "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/processing_for_alignment/datasets/ahey/MaxMin_TrPh_3600-Filtered",
    # "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/processing_for_alignment/datasets/ahey/MaxMin_TrPh_4096-Filtered",
]

models = ["apertus-1.5-3.0b-sft"]
model_hps = {
    # "apertus-70b-sft": {
    #     "ids_paths": [
    #         (
    #             "mixture-7-d0012600a8854237",
    #             "\${artifacts_dir}/shared/outputs/train_sft/final-run/Apertus70B-tokens15T-longcontext64k-apertus-sft-mixture-7-ln-v2-bs1024-lr2e-06-maxgnorm1-epochs1-ademamix/checkpoints/d0012600a8854237/checkpoint-4462",
    #         ),
    #     ],
    #     "batch_size": 512,
    #     "num_nodes_per_job": 64,
    #     "per_device_train_batch_size": 2,
    #     "accelerate_config": "src/post_training/configs/accelerate/ds-zero3.yaml",
    # },
    "apertus-8b-sft": {
        "ids_paths": [ # Important
            (
                "Apertus-8B-Instruct-2509-SFT",
                "/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--swiss-ai--Apertus-8B-Instruct-2509-SFT/snapshots/d57e4f1a3baa6315c60707346b5498b48b40a364",
            ),
        ],
        "batch_size": 128,
        "num_nodes_per_job": 16,
        "per_device_train_batch_size": 2,
        "accelerate_config": "src/post_training/configs/accelerate/ds-zero2.yaml",
    },
    "apertus-8b-sft-68": {
        "ids_paths": [ # Important
            (
                "Apertus-8B-Instruct-2509-SFT-68",
                "/iopsstor/scratch/cscs/dmelikidze/sft-models/hanna-68",
            ),
        ],
        "batch_size": 128,
        "num_nodes_per_job": 16,
        "per_device_train_batch_size": 2,
        "accelerate_config": "src/post_training/configs/accelerate/ds-zero2.yaml",
    },
    "apertus-8b-sft-64": {
        "ids_paths": [ # Important
            (
                "Apertus-8B-Instruct-2509-SFT-64",
                "/iopsstor/scratch/cscs/dmelikidze/sft-models/hanna-64",
            ),
        ],
        "batch_size": 128,
        "num_nodes_per_job": 16,
        "per_device_train_batch_size": 2,
        "accelerate_config": "src/post_training/configs/accelerate/ds-zero2.yaml",
    },
    "apertus-8b-sft-new": {
        "ids_paths": [ # Important
            (
                "Apertus-8B-Instruct-2509-SFT-New",
                "/iopsstor/scratch/cscs/dmelikidze/sft-models/sft-ifeval-new",
            ),
        ],
        "batch_size": 128,
        "num_nodes_per_job": 16,
        "per_device_train_batch_size": 2,
        "accelerate_config": "src/post_training/configs/accelerate/ds-zero2.yaml",
    },
    "apertus-1.5-0.6b-sft": {
        "ids_paths": [ # Important
            (
                "Apertus-8B-Instruct-2509-SFT-New",
                "/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--daslab-testing--Apertus-0.6B-it800000-SFT/snapshots/b88d2d7167404b9be4ee088118d8e22dc433e6ca",
            ),
        ],
        "batch_size": 128,
        "num_nodes_per_job": 8,
        "per_device_train_batch_size": 2,
        "accelerate_config": "src/post_training/configs/accelerate/ds-zero2.yaml",
    },
    "apertus-1.5-1.7b-sft": {
        "ids_paths": [ # Important
            (
                "Apertus-8B-Instruct-2509-SFT-New",
                "/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--daslab-testing--Apertus-1.7B-it800000-SFT/snapshots/bf90093c4220c1ac525609b9ac51ba2c44bb48b5",
            ),
        ],
        "batch_size": 128,
        "num_nodes_per_job": 16,
        "per_device_train_batch_size": 2,
        "accelerate_config": "src/post_training/configs/accelerate/ds-zero2.yaml",
    },
    "apertus-1.5-3.0b-sft": {
        "ids_paths": [ # Important
            (
                "Apertus-8B-Instruct-2509-SFT-New",
                "/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--daslab-testing--Apertus-3B-it800000-SFT/snapshots/3ba80c67d186140198d894c2aa5e965521a31c4b",
            ),
        ],
        "batch_size": 128,
        "num_nodes_per_job": 16,
        "per_device_train_batch_size": 2,
        "accelerate_config": "src/post_training/configs/accelerate/ds-zero2.yaml",
    }
}
reward_models = ["skywork-llama3-8b"]

ref_logprobs_from_dataset = False
train_num_ref_rewards = -1  # Directly use the quantile rewards from the dataset.

losses = ["dpo"]
normalize_beta_by_length = True # Important
betas = {
    "qrpo": [2.0],
    "dpo": [25.0],
}
learning_rates = [1e-6] # [5e-7] for QRPO
optimizers = ["adamw_torch"]
max_grad_norm = 20  # Disable but still log.
num_epochs = [1]

num_devices_per_node = 4
seed = 5315

commands = []
total_nodes_needed = 0
for dataset in datasets:
    for reward_model in reward_models:
        for model in models:
            sftids = model_hps[model]["ids_paths"]
            batch_size = model_hps[model]["batch_size"]
            num_nodes_per_job = model_hps[model]["num_nodes_per_job"]
            per_device_train_batch_size = model_hps[model][
                "per_device_train_batch_size"
            ]
            accelerate_config = model_hps[model]["accelerate_config"]
            accumulation_steps = batch_size // (
                num_nodes_per_job * num_devices_per_node * per_device_train_batch_size
            )
            for sftid, sftid_path in sftids:
                for train_dataset_path in train_dataset_paths:
                    train_dataset_name = Path(train_dataset_path).name
                    for loss in losses:
                        for optimizer in optimizers:
                            for lr in learning_rates:
                                for beta in betas[loss]:
                                    for epochs in num_epochs:
                                        jobid = f"{model}-{train_dataset_name}-{loss}-lr{lr}-beta{beta}-lenNorm{normalize_beta_by_length}-ebs{batch_size}-ep{epochs}"
                                        run_name = f"{job_name}/{jobid}"
                                        commands.append(
                                            (
                                                "sbatch "
                                                f"-p normal "
                                                f"-t 12:00:00 "
                                                f"-N {num_nodes_per_job} "
                                                f"-o {stdout_root}/out/{jobid}.out "
                                                f"-e {stdout_root}/out/{jobid}.err "
                                                "./cscs-shared-submit-scripts/recursive-unattended-accelerate.sh "
                                                f"-m post_training.train_preference "
                                                f"accelerate_config={accelerate_config} "
                                                f"dataset={dataset} "
                                                f"dataset_args.dataset_name='{train_dataset_path}' "
                                                f"dataset_args.train_split.name=train_split "
                                                f"model={model} "
                                                f"model_args.model_name_or_path='{sftid_path}' "
                                                f"training_args.max_grad_norm={max_grad_norm} "
                                                f"training_args.gradient_accumulation_steps={accumulation_steps} "
                                                f"training_args.per_device_train_batch_size={per_device_train_batch_size} "
                                                f"training_args.optim={optimizer} "
                                                f"training_args.learning_rate={lr} "
                                                f"training_args.loss_type={loss} "
                                                f"training_args.normalize_beta_by_length={normalize_beta_by_length} "
                                                f"training_args.num_ref_rewards={train_num_ref_rewards} "
                                                f"training_args.ref_logprobs_from_dataset={ref_logprobs_from_dataset} "
                                                f"training_args.beta={beta} "
                                                f"training_args.num_train_epochs={epochs} "
                                                f"seed={seed} "
                                                f"global_batch_size={batch_size} "
                                                f"num_nodes={num_nodes_per_job} "
                                                f"job_subdir={run_name} "
                                                f"wandb.run_name={run_name} "
                                                f"'wandb.tags=[prod,{job_name}]' "
                                                "artifacts_subdir=private "
                                                "resuming.resume=True "
                                            )
                                        )
                                        total_nodes_needed += num_nodes_per_job

# Write th submit commands to a new directory where this batch of experiments will be managed)
# Path from the project root
submit_dir = Path.cwd() / str(stdout_root)
submit_dir.mkdir(parents=True, exist_ok=True)
submit_file = submit_dir / "submit.sh"
print(f"Writing {len(commands)} commands to {submit_file}")
with open(submit_file, "w") as f:
    for command in commands:
        f.write(command + "\n")
print("Total nodes needed:", total_nodes_needed)
