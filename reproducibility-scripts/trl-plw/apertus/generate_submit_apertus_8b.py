from datetime import datetime

models = ["apertus3-8b"]
datasets = ["swissai-tulu-3-sft-0225"]

num_epochs = 2
batch_size = 128
max_seq_length = 4096
num_nodes = 8
num_proc_per_node = 4
proc_train_batch_size = 1
accumulation_steps = batch_size // (
    num_nodes * num_proc_per_node * proc_train_batch_size
)
proc_eval_batch_size = 2

learning_rates = [5e-6]  # 8b: 5e-6; 70b: 2e-6
lr_scheduler_type = "linear"
lr_warmup_ratio = 0.03

trainer = "plw"  # can only take values: sft, plw, ln-plw, irl
# prompt_loss_weight = [0.0, 0.01, 0.1, 0.5, 1.0]
prompt_loss_weight = [
    0.0,
]  # where sft -> plw=1.0

logging_steps = 1
# eval_steps = 1000
save_steps = 1000


current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
run_name = f"apertus3-8b-sweep"
nruns = 0
for dataset in datasets:
    for model in models:
        for chat_template in [
            # "simple_concat_with_space", "simple_concat_with_new_line", "simple_chat", "zephyr",
            "tulu_special_token"
            # "mistral",
            # "tulu"
        ]:
            for iter in [
                # "Apertus8B-tokens2T-it478000",
                # "Apertus8B-tokens3T-it716000",
                # "Apertus8B-tokens4T-it954000",
                # "Apertus8B-tokens5T-it1194000",
                # "Apertus8B-tokens6T-it1432000",
                # "Apertus8B-tokens7T-it1670000",
                "Apertus8B-tokens7.04T-it1678000",
                # "Apertus8B-tokens7.09T-it1690000-phase3",
                # "Apertus8B-tokens7.09T-it1690000-phase4",
                # "Apertus8B-tokens7.09T-it1690000-phase4-provenance-flan-short",
                # "Apertus8B-tokens7.09T-it1690000-phase4-megamath-pro-short",
            ]:
                for lr in learning_rates:
                    for plw in prompt_loss_weight:
                        model_config = f"{iter}-{chat_template}-{dataset}"
                        # model_config = f"{iter}-{dataset}"
                        hp_config = f"{trainer}-{plw}-lr-{lr}"
                        command = (
                            f"sbatch "
                            f"--nodes {num_nodes} "
                            f"--output=reproducibility-scripts/trl-plw/out-{current_time}/{model_config}/{hp_config}.out "
                            "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-ds.sh "
                            # "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-ds-zero2.sh "
                            f"-m swiss_alignment.trl.plw.train_plw "
                            # f"-m swiss_alignment.trl.plw.train_ademamix "
                            f"dataset={dataset} "
                            f"model={model}.yaml "
                            f"model_args.model_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/{iter} "
                            f"tokenizer_args.tokenizer_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/{iter} "
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
                            # f"training_args.eval_steps={eval_steps} "
                            f"training_args.eval_on_start=false "
                            f"training_args.save_strategy=steps "
                            f"training_args.save_steps={save_steps} "
                            f"tokenizer_args.chat_template_name={chat_template} "
                            "outputs_subdir=shared "
                            f"job_subdir={run_name}/chat-template/{model_config} "
                            f"wandb.run_name={model_config} "
                            f"wandb.tags=[prod,{trainer}] "
                            "resuming.resume=True "
                        )
                        print(command)
                        nruns += 1

print(nruns)


# --dependency=afterany:539566

# Tulu with special token
# sbatch --dependency=afterany:566199 --nodes 8 --output=reproducibility-scripts/trl-plw/out-2025-07-16-00-03/Apertus8B-tokens7.04T-it1678000-tulu_special_token-swissai-tulu-3-sft-0225/plw-0.0-lr-5e-06.out ./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-ds.sh -m swiss_alignment.trl.plw.train_plw dataset=swissai-tulu-3-sft-0225 model=apertus3-8b.yaml model_args.model_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus8B-tokens7.04T-it1678000 tokenizer_args.tokenizer_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus8B-tokens7.04T-it1678000 trainer=plw plw_args.prompt_loss_weight=0.0 training_args.max_seq_length=4096 training_args.num_train_epochs=2 training_args.gradient_accumulation_steps=4 training_args.per_device_train_batch_size=1 training_args.per_device_eval_batch_size=2 training_args.learning_rate=5e-06 training_args.lr_scheduler_type=linear training_args.warmup_ratio=0.03 training_args.logging_steps=1 training_args.eval_strategy=no training_args.eval_on_start=false training_args.save_strategy=steps training_args.save_steps=1000 tokenizer_args.chat_template_name=tulu_special_token outputs_subdir=shared job_subdir=apertus3-8b-sweep/chat-template/Apertus8B-tokens7.04T-it1678000-tulu_special_token-swissai-tulu-3-sft-0225 wandb.run_name=Apertus8B-tokens7.04T-it1678000-tulu_special_token-swissai-tulu-3-sft-0225 wandb.tags=[prod,plw] resuming.resume=True
