from datetime import datetime
from pathlib import Path

stdout_prefix = "run"
stdout_root = (
    Path(__file__).parent.resolve().relative_to(Path.cwd())
    / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

# Will be used in the root of the job_subdir.
# artifacts/shared/outputs/train_sft/job_name/...
job_name = "baseline-swissai-tulu3-ademamix"

models = ["apertus-8b", "apertus-70b"]
dataset = "swissai-tulu-3-sft-0225"

# Hyperparameters
num_device_per_node = 4
hyper_params = {
    "apertus-8b": {
        "checkpoint": "Apertus8B-tokens7.2T-it1728000-hotfix",
        "accelerate_config": "src/swiss_alignment/configs/accelerate/ds-zero2.yaml",
        "num_epochs": 2,
        "batch_size": (128, 16),  # bs, num_nodes
        "optim": "ademamix",
        "learning_rate": 5e-6,
        "num_device_per_node": num_device_per_node,
        "device_train_batch_size": 2,
        "trainer": ("plw", 0.0),
        "chat_template": "tulu",
    },
    "apertus-70b": {
        "checkpoint": "Apertus70B-tokens15T-it1155828",
        "accelerate_config": "src/swiss_alignment/configs/accelerate/ds-zero3.yaml",
        "num_epochs": 2,
        "batch_size": (128, 32),  # bs, num_nodes
        "optim": "ademamix",
        "learning_rate": 2e-6,
        "num_device_per_node": num_device_per_node,
        "device_train_batch_size": 1,
        "trainer": ("plw", 0.0),
        "chat_template": "tulu",
    },
}

commands = []
total_nodes_needed = 0
for model in models:
    hp = hyper_params[model]
    batch_size, num_nodes = hp["batch_size"]
    trainer, plw = hp["trainer"]

    accumulation_steps = batch_size // (
        num_nodes * num_device_per_node * hp["device_train_batch_size"]
    )

    job_id = f"{hp['checkpoint']}-{dataset}-bs{batch_size}-lr{hp['learning_rate']}-{hp['optim']}"
    run_name = f"{job_name}/{job_id}"
    command = (
        f"sbatch "
        f"-N {num_nodes} "
        f"-p large512 "
        f"-t 48:00:00 "
        f"-o {stdout_root}/out/{job_id}.out "
        f"-e {stdout_root}/out/{job_id}.err "
        "./cscs-shared-submit-scripts/recursive-unattended-accelerate.sh "
        f"-m swiss_alignment.train_sft "
        f"dataset={dataset} "
        f"model={model} "
        f"model_args.model_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/{hp['checkpoint']} "
        f"tokenizer_args.tokenizer_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/{hp['checkpoint']} "
        f"trainer={trainer} "
        f"accelerate_config={hp['accelerate_config']} "
        f"plw_args.prompt_loss_weight={plw} "
        f"training_args.gradient_accumulation_steps={accumulation_steps} "
        f"training_args.per_device_train_batch_size={hp['device_train_batch_size']} "
        f"training_args.learning_rate={hp['learning_rate']} "
        f"tokenizer_args.chat_template_name={hp['chat_template']} "
        f"training_args.num_train_epochs={hp['num_epochs']} "
        f"training_args.optim={hp['optim']} "
        "artifacts_subdir=shared "
        f"job_subdir={run_name} "
        f"wandb.run_name={run_name} "
        f"wandb.tags=[prod,{trainer},default,{job_name}] "
        "resuming.resume=True "
    )
    commands.append(command)
    total_nodes_needed += num_nodes


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
