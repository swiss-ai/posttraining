from datetime import datetime

models = ["apertus-70b", "apertus-8b"]
# datasets = ["apertus-sft-mixture-1", "apertus-sft-mixture-2", "apertus-sft-mixture-3"]
datasets = ["apertus-sft-mixture-4"]

# Hyperparameters
num_proc_per_node = 4
hyper_params = {
    "apertus-8b": {
        "checkpoint": "Apertus8B-tokens7.2T-it1678000",
        "accelerate_config": "src/swiss_alignment/configs/accelerate/ds-zero2.yaml",
        "num_epochs": 1,
        "max_seq_length": 4096,
        "batch_size": (512, 16),  # bs, num_nodes
        "learning_rate": 5e-6,
        "max_grad_norm": 0.1,
        "num_proc_per_node": num_proc_per_node,
        "proc_train_batch_size": 2,
        "trainer": ("plw", 0.0),
        "chat_template": "tulu",
    },
    "apertus-70b": {
        "checkpoint": "Apertus70B-tokens15T-it1155828",
        "accelerate_config": "src/swiss_alignment/configs/accelerate/ds-zero3.yaml",
        "num_epochs": 1,
        "max_seq_length": 4096,
        "batch_size": (512, 64),  # bs, num_nodes
        "learning_rate": 2e-6,
        "max_grad_norm": 1,
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
    for dataset in datasets:
        model_config = f"{hp['checkpoint']}-ademamix-{dataset}"
        batch_size, num_nodes = hp["batch_size"]
        accumulation_steps = batch_size // (
            num_nodes * num_proc_per_node * hp["proc_train_batch_size"]
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
            f"model_args.model_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/{hp['checkpoint']} "
            f"tokenizer_args.tokenizer_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/{hp['checkpoint']} "
            f"trainer={trainer} "
            f"accelerate_config={hp['accelerate_config']} "
            f"plw_args.prompt_loss_weight={plw} "
            f"training_args.gradient_accumulation_steps={accumulation_steps} "
            f"training_args.per_device_train_batch_size={hp['proc_train_batch_size']} "
            f"training_args.learning_rate={hp['learning_rate']} "
            f"training_args.max_grad_norm={hp['max_grad_norm']} "
            f"tokenizer_args.chat_template_name={hp['chat_template']} "
            "artifacts_subdir=shared "
            f"job_subdir={run_name}/dataset-mixtures-fast/{model_config} "
            f"wandb.run_name={model_config} "
            f"wandb.tags=[prod,{trainer}] "
            "resuming.resume=True "
        )
        print(command)
        nruns += 1
print(nruns)
