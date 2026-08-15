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

job_name = "apertus-70b-sft-init-dpo"

datasets = ["swissai-olmo2-32b-preference"]
train_dataset_paths = [
    "/capstor/store/cscs/swissai/infra01/datasets/alignment/preference_datasets/MaxMin_Tr_3600-Filtered-Decontaminated",
]

batch_size = 128
num_nodes_per_job = 32  # accumulation_steps -> 1; global batch stays 128
per_device_train_batch_size = 1
accelerate_config = "src/post_training/configs/accelerate/ds-zero3.yaml"
model_config = "apertus-70b-sft-1.5"

model_paths = [
    # --- currently staged: SFT-init control (job 2 of 2) ---
    # The SFT checkpoint the July run actually used (ap1p5-70b-sft-262k-2700). Its
    # scratch copy was wiped by the purge; this is the surviving capstor copy.
    "/capstor/store/cscs/swissai/infra01/models/Alignment/ap_1p5/Apertus-1.5-70B-SFT",

    # --- already submitted as job 3088131 (init-2026-08-15-15-42) ---
    # SFT->RL 70B checkpoint, byte-identical to Alignment/ap_1p5/Apertus-1.5-70B-SFT-RL.
    # "/capstor/store/cscs/swissai/infra01/models/rleval/rl_1p5-70b_notools_mixthink_0107_180it",
]
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
save_steps = 500  # ~800 GB per 70B checkpoint; save_total_limit=3 keeps the last three.

commands = []
total_nodes_needed = 0
accumulation_steps = batch_size // (
    num_nodes_per_job * num_devices_per_node * per_device_train_batch_size
)
for dataset in datasets:
    for reward_model in reward_models:
        for model_path in model_paths:
            model = Path(model_path).name
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
                                            # The previous 70B run died at step 1000 with a single
                                            # checkpoint to show for ~8h. save_steps is in
                                            # resuming.ignore_keys, so this does not change the
                                            # config hash / run id. Drop the line for a byte-identical
                                            # repeat of the old config.
                                            f"training_args.save_steps={save_steps} "
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

# Write the submit commands to a new directory where this batch of experiments will be managed)
# Path from the project root
submit_dir = Path.cwd() / str(stdout_root)
submit_dir.mkdir(parents=True, exist_ok=True)
# sbatch -o/-e point here; slurm will not create it and the job dies without logs.
(submit_dir / "out").mkdir(exist_ok=True)
submit_file = submit_dir / "submit.sh"
print(f"Writing {len(commands)} commands to {submit_file}")
with open(submit_file, "w") as f:
    for command in commands:
        f.write(command + "\n")
print("Total nodes needed:", total_nodes_needed)
