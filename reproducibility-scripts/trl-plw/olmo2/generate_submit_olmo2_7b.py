from datetime import datetime

models = ["olmo2-7b"]
datasets = ["swissai-tulu-3-sft-0225"]

num_epochs = 2
batch_size = 128
max_seq_length = 4096
num_nodes = 8
num_proc_per_node = 4
proc_train_batch_size = 2
accumulation_steps = batch_size // (
    num_nodes * num_proc_per_node * proc_train_batch_size
)
proc_eval_batch_size = 2

learning_rates = [5e-6]  # 8b: 5e-6; 70b: 2e-6
lr_scheduler_type = "linear"
lr_warmup_ratio = 0.03

trainer = "plw"  # can only take values: sft, plw, ln-plw, irl
prompt_loss_weight = [
    0.0,
]  # where sft -> plw=1.0

logging_steps = 1
save_steps = 1000

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
run_name = f"olmo2-7b-sweep"
nruns = 0
for dataset in datasets:
    for model in models:
        for iter in [
            # "Olmo2-7B-stage1-step239000-tokens1003B",
            # "Olmo2-7B-stage1-step477000-tokens2001B",
            # "Olmo2-7B-stage1-step716000-tokens3004B",
            # "Olmo2-7B-stage1-step928646-tokens3896B",
            "Olmo2-7B-stage2-tokens4T",
        ]:
            for lr in learning_rates:
                for plw in prompt_loss_weight:
                    model_config = f"{iter}-ademamix-{dataset}"
                    hp_config = f"{trainer}-{plw}-lr-{lr}"
                    command = (
                        f"sbatch "
                        f"--nodes {num_nodes} "
                        f"--output=reproducibility-scripts/trl-plw/out-{current_time}/{model_config}/{hp_config}.out "
                        "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-ds-zero2.sh "
                        f"-m swiss_alignment.trl.plw.train_plw "
                        f"dataset={dataset} "
                        # f"dataset_args.debug_oom=true "
                        # f"dataset_args.debug_subsample.train=50_000 "
                        # f"dataset_args.debug_subsample.eval=100 "
                        f"model={model}.yaml "
                        f"model_args.model_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/olmo2/{iter} "
                        f"tokenizer_args.tokenizer_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/olmo2/{iter} "
                        f"trainer={trainer} "
                        f"plw_args.prompt_loss_weight={plw} "
                        f"training_args.max_seq_length={max_seq_length} "
                        # f"training_args.max_grad_norm=1 "
                        f"training_args.num_train_epochs={num_epochs} "
                        f"training_args.gradient_accumulation_steps={accumulation_steps} "
                        f"training_args.per_device_train_batch_size={proc_train_batch_size} "
                        f"training_args.per_device_eval_batch_size={proc_eval_batch_size} "
                        f"training_args.learning_rate={lr} "
                        f"training_args.lr_scheduler_type={lr_scheduler_type} "
                        f"training_args.warmup_ratio={lr_warmup_ratio} "
                        f"training_args.logging_steps={logging_steps} "
                        f"training_args.eval_strategy=no "
                        f"training_args.eval_on_start=false "
                        f"training_args.save_strategy=steps "
                        f"training_args.save_steps={save_steps} "
                        f"tokenizer_args.chat_template_name=tulu "
                        "artifacts_subdir=private "
                        f"job_subdir={run_name}/{model_config} "
                        f"wandb.run_name={model_config} "
                        f"wandb.tags=[prod,{trainer}] "
                        "resuming.resume=True "
                    )
                    print(command)
                    nruns += 1

print(nruns)

# Ademamix
# sbatch --dependency=afterany:579170 --nodes 8 --output=reproducibility-scripts/trl-plw/out-2025-07-22-13-21/Olmo2-7B-stage2-tokens4T-ademamix-swissai-tulu-3-sft-0225/plw-0.0-lr-5e-06.out ./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-ds-zero2.sh -m swiss_alignment.trl.plw.train_plw dataset=swissai-tulu-3-sft-0225 model=olmo2-7b.yaml model_args.model_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/olmo2/Olmo2-7B-stage2-tokens4T tokenizer_args.tokenizer_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/olmo2/Olmo2-7B-stage2-tokens4T trainer=plw plw_args.prompt_loss_weight=0.0 training_args.max_seq_length=4096 training_args.num_train_epochs=2 training_args.gradient_accumulation_steps=2 training_args.per_device_train_batch_size=2 training_args.per_device_eval_batch_size=2 training_args.learning_rate=5e-06 training_args.lr_scheduler_type=linear training_args.warmup_ratio=0.03 training_args.logging_steps=1 training_args.eval_strategy=no training_args.eval_on_start=false training_args.save_strategy=steps training_args.save_steps=1000 tokenizer_args.chat_template_name=tulu artifacts_subdir=private job_subdir=olmo2-7b-sweep/Olmo2-7B-stage2-tokens4T-ademamix-swissai-tulu-3-sft-0225 wandb.run_name=Olmo2-7B-stage2-tokens4T-ademamix-swissai-tulu-3-sft-0225 wandb.tags=[prod,plw] resuming.resume=True
