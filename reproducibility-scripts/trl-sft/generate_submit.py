from datetime import datetime

models = ["mistralai-mistral-7b-instruct.yaml"]
datasets = ["magpieair"]

learning_rates = [5e-6, 1e-5, 2e-5]
ds_accumulation_steps = 8

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
run_name = f"example-hp-sweep"
nruns = 0
for dataset in datasets:
    for model in models:
        for lr in learning_rates:
            model_config = f"{model}-{dataset}"
            hp_config = f"lr-{lr}"
            command = (
                f"sbatch --nodes 4 "
                f"--output=reproducibility-scripts/trl-sft/out-{current_time}/{model_config}/{hp_config}.out "
                "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-ds.sh "
                f"-m swiss_alignment.trl.train_sft "
                f"dataset={dataset} "
                f"model={model} "
                f"training_args.gradient_accumulation_steps={ds_accumulation_steps} "
                f"training_args.learning_rate={lr} "
                "outputs_subdir=shared "
                f"job_subdir={run_name}/{model_config} "
                f"wandb.run_name={model_config}-sft-{run_name} "
                "'wandb.tags=[prod,sft]' "
                "resuming.resume=True "
            )
            print(command)
            nruns += 1

print(nruns)
