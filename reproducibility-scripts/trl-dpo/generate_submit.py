from datetime import datetime

models = ["tulu-3-8b-sft.yaml"]
datasets = ["tulu-3-8b-preference-mixture"]

num_nodes = 4

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
run_name = f"tulu-3-dpo-v0-b0.1"
nruns = 0
with open("dpo_submit.sh", "w") as file:
    for dataset in datasets:
        for model in models:
            model_config = f"{model}-{dataset}"
            command = (
                f"sbatch "
                f"--nodes {num_nodes} "
                f"--output=reproducibility-scripts/trl-dpo/out-{current_time}/{model_config}/dpo.out "
                "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-ds.sh "
                f"-m swiss_alignment.trl.dpo.train_dpo "
                "outputs_subdir=shared "
                f"job_subdir={run_name}/{model_config} "
                f"wandb.run_name={model_config}-{run_name} "
                f"wandb.tags=[prod,dpo] "
                "resuming.resume=True "
            )
            print(command, file=file)
            nruns += 1

print(nruns)
