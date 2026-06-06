from datetime import datetime
from pathlib import Path

"""Offline DPO launcher for a RAW-ONLY baseline (50k random samples from MaxMin_Tr_3600).

Same model / DPO hyperparameters as the mixed run, but the dataset config (mllm... -> maxmin-raw-50k)
selects a single raw HF source capped to 50,000 samples, runtime-tokenized via the mixture raw path.
See data_alignment/tokenized_preference.build_mixed_preference_dataset.
"""

stdout_prefix = "init-raw"
stdout_root = (
    Path(__file__).parent.resolve()
    / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

job_name = "apertus-raw-dpo"
wandb_project = "Ap1p5-DPO"
wandb_entity = "rkreft-personal"

# Dataset config (configs/dataset/<name>.yaml) that encodes the raw-only source.
dataset_config = "maxmin-raw-50k"

batch_size = 128
num_nodes_per_job = 4
per_device_train_batch_size = 1
accelerate_config = "src/post_training/configs/accelerate/ds-zero2.yaml"
model_config = "apertus-8b-sft-1.5--lr8e-5"

model_paths = [
    # Only the _4200 SFT checkpoint is used for this DPO run (readable copy under $STORE).
    "/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200",
]

ref_logprobs_from_dataset = False
train_num_ref_rewards = -1  # Directly use the quantile rewards from the dataset.

losses = ["dpo"]
normalize_beta_by_length = True  # Important
betas = {
    "qrpo": [2.0],
    "dpo": [25.0],
}
learning_rates = [1e-6]  # [5e-7] for QRPO
optimizers = ["adamw_torch"]
max_grad_norm = 20  # Disable but still log.
num_epochs = [1]

num_devices_per_node = 4
seed = 5315

commands = []
total_nodes_needed = 0
accumulation_steps = batch_size // (
    num_nodes_per_job * num_devices_per_node * per_device_train_batch_size
)
for model_path in model_paths:
    model = Path(model_path).name
    for loss in losses:
        for optimizer in optimizers:
            for lr in learning_rates:
                for beta in betas[loss]:
                    for epochs in num_epochs:
                        jobid = f"{model}-{dataset_config}-{loss}-lr{lr}-beta{beta}-lenNorm{normalize_beta_by_length}-ebs{batch_size}-ep{epochs}"
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
                                f"dataset={dataset_config} "
                                f"model={model_config} "
                                f"model_args.model_name_or_path='{model_path}' "
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
                                f"wandb.project={wandb_project} "
                                f"wandb.entity={wandb_entity} "
                                f"wandb.run_name={run_name} "
                                f"'wandb.tags=[prod,{job_name}]' "
                                "artifacts_subdir=private "
                                "resuming.resume=True "
                            )
                        )
                        total_nodes_needed += num_nodes_per_job

# Write the submit commands to a new directory where this batch of experiments will be managed.
# Path from the project root.
submit_dir = Path.cwd() / str(stdout_root)
submit_dir.mkdir(parents=True, exist_ok=True)
submit_file = submit_dir / "submit.sh"
print(f"Writing {len(commands)} commands to {submit_file}")
with open(submit_file, "w") as f:
    for command in commands:
        f.write(command + "\n")
print("Total nodes needed:", total_nodes_needed)
