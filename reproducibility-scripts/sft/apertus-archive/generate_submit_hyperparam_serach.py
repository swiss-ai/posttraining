from datetime import datetime
from itertools import product
from pathlib import Path

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
stdout_root = (
    Path(__file__).parent.resolve().relative_to(Path.cwd()) / f"out-{current_time}"
)

models = ["apertus-8b"]
datasets = ["swissai-tulu-3-sft-0225"]

# Hyperparameters
num_proc_per_node = 4
hyper_params = {
    "apertus-8b": {
        "accelerate_config": "src/post_training/configs/accelerate/ds-zero2.yaml",
        "num_epochs": 2,  # we save intermediate checkpoints
        "max_seq_length": 4096,
        "batch_size": [
            (128, 8),
        ],  # bs, num_nodes
        "num_proc_per_node": num_proc_per_node,
        "proc_train_batch_size": 2,
        "proc_eval_batch_size": 2,
        "learning_rates": [
            5e-6,
        ],
        "lr_scheduler_types": [
            # "constant",
            "linear",
            # "cosine",
        ],
        "lr_warmup_ratios": [0.03],
        "grad_clipping": [10_000],
        "trainers": [("plw", 0.0)],  # Trainers available: sft, plw, ln-plw
        "chat_templates": ["tulu"],
    },
    "apertus-70b": {
        "accelerate_config": "src/post_training/configs/accelerate/ds-zero3.yaml",
        "num_epochs": 2,  # we save intermediate checkpoints
        "max_seq_length": 4096,
        "batch_size": [
            (128, 32),
        ],  # bs, num_nodes
        "num_proc_per_node": num_proc_per_node,
        "proc_train_batch_size": 2,
        "proc_eval_batch_size": 2,
        "learning_rates": [
            2e-6,
        ],
        "lr_scheduler_types": [
            "linear",
        ],
        "lr_warmup_ratio": [0.03],
        "grad_clipping": [10_000],
        "trainers": [("plw", 0.0)],
        "chat_templates": ["tulu"],
    },
}


commands = []
for (
    model,
    dataset,
) in product(
    models,
    datasets,
):
    hp = hyper_params[model]
    for (
        lr,
        lr_scheduler_type,
        lr_warmup_ratio,
        (bs, num_nodes),
        grad_clip,
        (trainer, plw_weight),
        chat_template,
    ) in product(
        hp["learning_rates"],
        hp["lr_scheduler_types"],
        hp["lr_warmup_ratios"],
        hp["batch_size"],
        hp["grad_clipping"],
        hp["trainers"],
        hp["chat_templates"],
    ):
        run_name = f"{model}-sweep"
        model_config = f"{model}-{dataset}-hyperparam_search"
        hp_config = f"lr_{lr}-scheduler_{lr_scheduler_type}-bs_{bs}-grad_clip_{grad_clip}-trainer_{trainer}-plw_weight_{plw_weight}-chat_template_{chat_template}"
        command = (
            f"sbatch "
            f"--nodes {num_nodes} "
            f"--output={stdout_root}/{hp_config}.out "
            f"--error={stdout_root}/{hp_config}.err "
            "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/recursive-unattended-accelerate.sh "
            f"-m post_training.train_sft "
            f"dataset={dataset} "
            f"model={model}.yaml "
            f"tokenizer_args.chat_template_name={chat_template} "
            f"trainer={trainer} "
            f"accelerate_config={hp['accelerate_config']} "
            f"plw_args.prompt_loss_weight={plw_weight} "
            f"training_args.max_seq_length={hp['max_seq_length']} "
            f"training_args.max_grad_norm={grad_clip} "
            f"training_args.num_train_epochs={hp['num_epochs']} "
            f"training_args.gradient_accumulation_steps={bs // (num_nodes * num_proc_per_node * hp['proc_train_batch_size'])} "
            f"training_args.per_device_train_batch_size={hp['proc_train_batch_size']} "
            f"training_args.per_device_eval_batch_size={hp['proc_eval_batch_size']} "
            f"training_args.learning_rate={lr} "
            f"training_args.lr_scheduler_type={lr_scheduler_type} "
            f"training_args.warmup_ratio={lr_warmup_ratio} "
            f"training_args.logging_steps=1 "
            f"training_args.eval_strategy=no "
            f"training_args.eval_on_start=false "
            f"training_args.save_strategy=steps "
            f"training_args.save_steps=1000 "
            "artifacts_subdir=private "
            f"job_subdir={run_name}/{model_config}/{hp_config} "
            f"wandb.run_name={model_config}-{hp_config} "
            f"wandb.tags=[prod,{trainer}] "
            "resuming.resume=True "
        )
        print(command)
        commands.append(command)

# Write th submit commands to a new directory where this batch of experiments will be managed)
# Path from the project root
submit_dir = Path.cwd() / str(stdout_root)
submit_dir.mkdir(parents=True, exist_ok=True)
submit_file = submit_dir / "submit_hyperparam_search.sh"
print(f"Writing {len(commands)} commands to {submit_file}")
with open(submit_file, "w") as f:
    for command in commands:
        f.write(command + "\n")
