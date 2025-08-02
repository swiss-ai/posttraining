from datetime import datetime

models = ["apertus-8b", "apertus-70b"]
dataset = "swissai-tulu-3-sft-0225"

# Hyperparameters
num_proc_per_node = 4
hyper_params = {
    "apertus-8b": {
        "checkpoints": [
            "Apertus8B-tokens2T-it478000",
            "Apertus8B-tokens3T-it716000",
            "Apertus8B-tokens4T-it954000",
            "Apertus8B-tokens5T-it1194000",
            "Apertus8B-tokens6T-it1432000",
            "Apertus8B-tokens7T-it1670000",
            "Apertus8B-tokens7.04T-it1678000",
            "Apertus8B-tokens7.2T-it1728000",
        ],
        "accelerate_config": "src/swiss_alignment/configs/accelerate/ds-zero2.yaml",
        "num_epochs": 2,
        "max_seq_length": 4096,
        "batch_size": (128, 8),  # bs, num_nodes
        "learning_rate": 5e-6,
        "optim": "adamw_torch",
        "num_proc_per_node": num_proc_per_node,
        "proc_train_batch_size": 2,
        "trainer": ("plw", 0.0),
        "chat_template": "tulu",
    },
    "apertus-70b": {
        "checkpoints": [
            "Apertus70B-tokens3T-it560000",
            "Apertus70B-tokens4T-it560000",
            "Apertus70B-tokens5T-it560000",
            "Apertus70B-tokens6T-it619500",
            "Apertus70B-tokens7T-it679000",
            "Apertus70B-tokens8T-it739000",
            "Apertus70B-tokens9T-it798250",
            "Apertus70B-tokens10T-it858000",
            "Apertus70B-tokens11T-it917500",
            "Apertus70B-tokens12T-it977250",
            "Apertus70B-tokens13T-it1036750",
            "Apertus70B-tokens14T-it1096250",
            "Apertus70B-tokens15T-it1155828",
        ],
        "accelerate_config": "src/swiss_alignment/configs/accelerate/ds-zero3.yaml",
        "num_epochs": 2,
        "max_seq_length": 4096,
        "batch_size": (128, 32),  # bs, num_nodes
        "learning_rate": 2e-6,
        "num_proc_per_node": num_proc_per_node,
        "proc_train_batch_size": 2,
        "trainer": ("plw", 0.0),
        "chat_template": "tulu",
    },
}

# Generate sbatch
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
nruns = 0
for model in models:
    run_name = f"{model}-sweep"
    hp = hyper_params[model]
    for checkpoint in hp["checkpoints"]:
        model_config = f"{checkpoint}-{dataset}"
        batch_size, num_nodes = hp["batch_size"]
        accumulation_steps = batch_size // (
            batch_size * num_proc_per_node * hp["proc_train_batch_size"]
        )
        trainer, plw = hp["trainer"]
        command = (
            f"sbatch "
            f"--nodes {num_nodes} "
            f"--output=reproducibility-scripts/trl-plw/out-{current_time}/{model_config}/swissai-tulu-3-sft.out "
            "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/recursive-unattended-accelerate.sh "
            f"-m swiss_alignment.train_sft "
            f"dataset={dataset} "
            f"model={model}.yaml "
            f"model_args.model_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/{checkpoint} "
            f"tokenizer_args.tokenizer_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/{checkpoint} "
            f"trainer={trainer} "
            f"accelerate_config={hp['accelerate_config']} "
            f"plw_args.prompt_loss_weight={plw} "
            f"training_args.gradient_accumulation_steps={accumulation_steps} "
            f"training_args.per_device_train_batch_size={hp['proc_train_batch_size']} "
            f"training_args.learning_rate={hp['learning_rate']} "
            f"tokenizer_args.chat_template_name={hp['chat_template']} "
            "artifacts_subdir=shared "
            f"job_subdir={run_name}/token-count/{model_config} "
            f"wandb.run_name={model_config} "
            f"wandb.tags=[prod,{trainer}] "
            "resuming.resume=True "
        )
        print(command)
        nruns += 1
print(nruns)
