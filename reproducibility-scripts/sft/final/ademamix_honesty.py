from datetime import datetime
from pathlib import Path

stdout_prefix = "ademamix-honesty"
stdout_root = (
    Path(__file__).parent.resolve().relative_to(Path.cwd())
    / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

# Will be used in the root of the job_subdir.
# artifacts/shared/outputs/train_sft/job_name/...
job_name = "sft4honesty"

models = [ "apertus-8b" ]

new_eos_token_id = 68  # The new EOS token ID to be used in the model
padding_side = "left"  # Padding side for the tokenizer
num_device_per_node = 4
hyper_params = {
    "apertus-8b": {
        "checkpoint": "Apertus-8B-aligned-apertus-sft-qa-simple",
        "checkpoint_path": "/iopsstor/scratch/cscs/smoalla/projects/posttraining/artifacts/shared/outputs/train_sft/qa-run/Apertus-8B-aligned-apertus-sft-qa-simple-bs16-lr5e-06-maxgnorm1.0-epochs4-ademamix-apertus-pad-left/checkpoints/04e19e7c2c512f95/checkpoint-27",
        "accelerate_config": "src/post_training/configs/accelerate/ds-zero2.yaml",
        "num_epochs": 1,
        "batch_size": (512, 4),  # bs, num_nodes
        "optimizer": "ademamix",
        "learning_rate": 2e-6,
        "max_grad_norm": 1.0,
        "num_device_per_node": num_device_per_node,
        "device_train_batch_size": 2,
        "trainer": ("plw", 0.0),
        "chat_template": "apertus",
        "datasets": [
            "Apertus-8B-aligned_0.1_0.7_correctness_no_refuse_sft_data",
            "Apertus-8B-aligned_0.2_0.75_mix_no_refuse_sft_data_correct",
            "Apertus-8B-aligned_0.33_0.67_majority_no_refuse_sft_data_correct"
        ],
    },
    "apertus-70b": {
        "checkpoint": "Apertus-70B-aligned-apertus-sft-qa-simple",
        "checkpoint_path": "/iopsstor/scratch/cscs/smoalla/projects/posttraining/artifacts/shared/outputs/train_sft/qa-run/Apertus-70B-aligned-apertus-sft-qa-simple-bs64-lr2e-06-maxgnorm1.0-epochs4-ademamix-apertus-pad-left/checkpoints/c9ba745f27e3c126/checkpoint-2",
        "accelerate_config": "src/post_training/configs/accelerate/ds-zero3.yaml",
        "num_epochs": 1,
        "batch_size": (1024, 16),  # bs, num_nodes
        "optimizer": "ademamix",
        "learning_rate": 2e-6,
        "max_grad_norm": 1.0,
        "num_device_per_node": num_device_per_node,
        "device_train_batch_size": 2,
        "trainer": ("plw", 0.0),
        "chat_template": "apertus",
        "datasets": [
            # "apertus_70b_3899_0.2_0.5_0.8_correctness_sft",
            # "apertus_70b_3899_0.2_0.75_majority_correct_sft",
            # "apertus_70b_3899_0.33_0.67_correctness_no_refuse_sft",
            # "apertus_70b_3899_0.2_0.75_correctness_no_refuse_sft",
            # "apertus_70b_3899_0.33_0.67_correctness_sft",
        ]
    },
}

commands = []
total_nodes_needed = 0
for model in models:
    hp = hyper_params[model]
    for dataset in hp["datasets"]:
        batch_size, num_nodes = hp["batch_size"]
        trainer, plw = hp["trainer"]

        accumulation_steps = batch_size // (
            num_nodes * num_device_per_node * hp["device_train_batch_size"]
        )

        job_id = f"{hp['checkpoint']}-{dataset}-bs{batch_size}-lr{hp['learning_rate']}-maxgnorm{hp['max_grad_norm']}-epochs{hp['num_epochs']}-ademamix-{hp['chat_template']}-pad-{padding_side}"
        run_name = f"{job_name}/{job_id}"
        command = (
            f"sbatch "
            f"-N {num_nodes} "
            f"-p normal "
            f"-t 03:00:00 "
            f"-o {stdout_root}/out/{run_name}.out "
            f"-e {stdout_root}/out/{run_name}.err "
            "./cscs-shared-submit-scripts/unattended-accelerate.sh "
            f"-m post_training.train_sft "
            f"dataset={dataset} "
            f"model={model} "
            f"model_args.model_name_or_path={hp['checkpoint_path']} "
            f"tokenizer_args.tokenizer_name_or_path={hp['checkpoint_path']} "
            f"trainer={trainer} "
            f"accelerate_config={hp['accelerate_config']} "
            f"plw_args.prompt_loss_weight={plw} "
            f"training_args.gradient_accumulation_steps={accumulation_steps} "
            f"training_args.per_device_train_batch_size={hp['device_train_batch_size']} "
            f"training_args.optim={hp['optimizer']} "
            f"training_args.learning_rate={hp['learning_rate']} "
            f"training_args.max_grad_norm={hp['max_grad_norm']} "
            f"tokenizer_args.chat_template_name={hp['chat_template']} "
            f"tokenizer_args.model_eos_token_id={new_eos_token_id} "
            f"tokenizer_args.padding_side={padding_side} "
            f"training_args.num_train_epochs={hp['num_epochs']} "
            "artifacts_subdir=shared "
            f"job_subdir={run_name} "
            f"wandb.run_name={run_name} "
            f"wandb.tags=[prod,{trainer},default,{job_name}] "
            "resuming.resume=True "
            f"global_batch_size={batch_size} "
            f"num_nodes={num_nodes} "
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