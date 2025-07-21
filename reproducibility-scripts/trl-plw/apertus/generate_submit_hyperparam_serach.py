from datetime import datetime
from itertools import product
from pathlib import Path

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
stdout_root = (
    Path(__file__).parent.resolve().relative_to(Path.cwd()) / f"out-{current_time}"
)


models = ["apertus3-8b"]
datasets = ["swissai-tulu-3-sft-0225"]

# Hyperparameters
num_epochs = 2  # we save intermediate checkpoints
max_seq_length = 4096
learning_rates = [5e-6]  # 8b: 5e-6; 70b: 2e-6
batch_size = [(128, 32)]  # bs, num_nodes
grad_clipping = [1_000]
trainers = [  # Trainers available: sft, plw, ln-plw, irl
    ("plw", 0.0)
    # ("ln-plw", 0.0),
    # ("ln-plw", 0.01),
    # ("ln-plw", 0.1),
    # ("irl", 0.0),
    # ("irl", 0.01),
    # ("irl", 0.1),
]
chat_templates = [
    # "simple_concat_with_space", "simple_concat_with_new_line", "simple_chat", "zephyr",
    "tulu"
]

lr_scheduler_type = "linear"  # TODO
lr_warmup_ratio = 0.03  # TODO


# TODO
num_proc_per_node = 4
proc_train_batch_size = 1
proc_eval_batch_size = 2

commands = []
run_name = f"apertus3-8b-sweep"
for (
    dataset,
    model,
    lr,
    (bs, num_nodes),
    grad_clip,
    (trainer, plw_weight),
    chat_template,
) in product(
    datasets,
    models,
    learning_rates,
    batch_size,
    grad_clipping,
    trainers,
    chat_templates,
):
    model_config = f"{model}-{dataset}"
    hp_config = f"lr={lr}-bs={bs}-grad_clip={grad_clip}-trainer={trainer}-plw_weight={plw_weight}-chat_template={chat_template}"
    command = (
        f"sbatch "
        f"--nodes {num_nodes} "
        f"--output={stdout_root}/{hp_config}.out "
        f"--error={stdout_root}/{hp_config}.err "
        "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-ds-zero1.sh "
        f"-m swiss_alignment.trl.plw.train_plw "
        f"dataset={dataset} "
        f"model={model}.yaml "
        f"tokenizer_args.chat_template_name={chat_template} "
        f"trainer={trainer} "
        f"plw_args.prompt_loss_weight={plw_weight} "
        f"training_args.max_seq_length={max_seq_length} "
        f"training_args.max_grad_norm={grad_clip} "
        f"training_args.num_train_epochs={num_epochs} "
        f"training_args.gradient_accumulation_steps={bs // (num_nodes * num_proc_per_node * proc_train_batch_size)} "
        f"training_args.per_device_train_batch_size={proc_train_batch_size} "
        f"training_args.per_device_eval_batch_size={proc_eval_batch_size} "
        f"training_args.learning_rate={lr} "
        f"training_args.lr_scheduler_type={lr_scheduler_type} "
        f"training_args.warmup_ratio={lr_warmup_ratio} "
        f"training_args.logging_steps=1 "
        f"training_args.eval_strategy=no "
        f"training_args.eval_on_start=false "
        f"training_args.save_strategy=steps "
        f"training_args.save_steps=1000 "
        "artifacts_subdir=shared "
        f"job_subdir={run_name}/{model_config}/{hp_config} "
        f"wandb.run_name={model_config} "
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
