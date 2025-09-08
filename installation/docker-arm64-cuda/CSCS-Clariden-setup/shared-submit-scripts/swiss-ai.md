# SwissAI Alignment Codebase

Docs for SwissAI users. To update when open-sourcing, if we want to support other users. Editing the template docs should help for this.

## Getting Started

### Preliminaries

- You can log in to Clariden and know the basic setup (https://github.com/swiss-ai/documentation/blob/main/pages/setup_clariden.md).
- You understand Clariden uses SLURM for job scheduling (https://github.com/swiss-ai/documentation/blob/main/pages/slurm_intro.md).
- You can authenticate to GitHub on Clariden through SSH agent forwarding (keys from your laptop follow your SSH connections) (`ssh -T git@github.com`). Check the [troubleshooting section](#github-authentication-through-ssh-agent-forwarding) if you have issues.

### Understand the storage structure

#### Scratch storage: shared and private

Use it to read and write during jobs.

In `iopstor/scratch` (deletion policy: 30 days):  
Shared: `/iopsstor/scratch/cscs/smoalla/projects/posttraining/artifacts/shared`  
Private: `/iopsstor/scratch/cscs/$USER/projects/posttraining/artifacts/private`

#### Permanent storage: shared

Move valuable permanent assets there, like good final checkpoints, checkpoints to keep for the release, container images...  

In `capstor/store`: `/capstor/store/cscs/swissai/infra01/swiss-alignment/artifacts`.

Check `reproducibility-scripts/data-management.sh` for example scripts to move data.
A todo is to automate this so that it's done at the end of jobs.

#### Storage summary:

These paths are symlinked  under the git root and all you have to remember are the following relative paths from the project root.  
Use these relative paths in the code so that it works for everyone.

```
PROJECT_ROOT/                  # In your home directory.
├── artifacts/                 # Anything that lives on other storage scratch and store.
│   ├── shared/                # Shared  (smoalla's iopsstor/scratch).
│   ├── private/               # Private (your iopsstor/scratch).
│   └── store/                 # Store   (infra01's capstor/store).
├── src/                       # Source code.
│   └── post_training/
...
```

#### Note

Everything in `store` and `shared` should have group write permissions for `infra01` by default.
If at any point this is not the case in some subdirectory you created, run the following:
```bash
cd the-problemetic-subdirectory
chown -R :infra01 .
chmod -R g+wrxs .
setfacl -R -m group::rwx,default:group::rwx,group:infra01:rwx,default:group:infra01:rwx .

# If you want to give another group read access:
GROUP_NAME=some-other-group
setfacl -R -m group:$GROUP_NAME:r-x,default:group:$GROUP_NAME:r-x .
```

### Clone the code and set up the storage

It will look like this:

```bash
$HOME/projects/post_training
├── dev/    # A dev instance of the repo for development jobs, which can be broken at any time.
└── run/    # A run instance of the repo from where experiments in the queue will read the code.
# Each has the same storage structure above.
```

commands:

```bash
mkdir -p $HOME/projects/posttraining && cd $HOME/projects/posttraining

git clone git@github.com:swiss-ai/posttraining.git dev
git clone git@github.com:swiss-ai/posttraining.git run

for INSTANCE in dev run; do
  INSTANCE=$INSTANCE ./$INSTANCE/artifacts/setup-symlinks.sh
done
```

### Set up credentials

They will be mounted in the containers at runtime.

```bash
# Weights & Biases
echo <my-wandb-api-key> > $HOME/.wandb-api-key
chmod 600 $HOME/.wandb-api-key

# Hugging Face  
echo <my-huggingface-api-key> > $HOME/.hf-token
chmod 600 $HOME/.hf-token
```

## What's next?

Check the sections below:
- [Convert Megatron checkpoints to HuggingFace](#convert-megatron-checkpoints-to-huggingface): how to convert pretrained Megatron checkpoints to HuggingFace.
- [Running the typical training experiment](#running-the-typical-training-experiment): how to run a training experiment with the codebase.
- [Where to change what to do what](#where-to-change-what-to-do-what): how to change the codebase to do different things.
- [Repository structure](#repository-structure): where to find what in the codebase.
- [Contributing](#contributing): good practices and guides for changing specific parts of the codebase.
- [Remote development](#remote-development): every user does this differently. We give you some docs but won't support you for this.
  We however have comprehensive documentation in TODO.
- [Rebuilding the container image](#rebuilding-the-container-image): how to rebuild the container image if you change the dependencies.

## Convert Megatron checkpoints to HuggingFace
Follow the instructions in [hfconverter](https://github.com/swiss-ai/hfconverter). From the hfconverter repository, run:
```bash
sbatch convert.sbatch <ckpt-path> <iteration> <output-path>
```
Ensure paths are valid and check `progress.txt` to find the correct iteration number.

## Running the typical training experiment

The `reproducibility-scripts` directory includes scripts to generate SLURM jobs for training models like Apertus 8B and 70B.
Below are example `sbatch` scripts for training Apertus 8B and 70B checkpoints, generated using a template in
[reproducibility-scripts/sft/0-apertus-template/generate_submit.py](https://github.com/swiss-ai/posttraining/tree/main/reproducibility-scripts/sft/0-apertus-template/generate_submit.py).

To create new experiments copy the directory `reproducibility-scripts/sft/0-apertus-template/` and rename `0-apertus-template` to your new experiment.
Then run the submit script inside to create sub-experiments whose submit script and SLURM logs will be recorded under.

The same for alignment in `/reproducibility-scripts/alignment-apertus-swissaiformat`.

For example  `reproducibility-scripts/sft/0-apertus-template/generate_submit.py`
creates `reproducibility-scripts/sft/0-apertus-template/some-sub-experiment-2025-08-07-17-14/submit.sh`

The commands in a `submit.sh` should be pasted and ran from your dev instance of the repo to have their logs there.
```
cd $HOME/projects/post_training/dev
```
The code used for the run will be the one in the stable `$HOME/projects/post_training/run` repo.

<details>
<summary>Apertus 70B sbatch command</summary>

```bash
sbatch -N 64 -p large512 -t 48:00:00 \
 -o reproducibility-scripts/sft/0-apertus-template/some-sub-experiment-2025-08-07-17-14/out/Apertus70B-tokens15T-it1155828-apertus-sft-mixture-1-bs512-lr2e-06-epochs1-adamw_torch.out \
 -e reproducibility-scripts/sft/0-apertus-template/some-sub-experiment-2025-08-07-17-14/out/Apertus70B-tokens15T-it1155828-apertus-sft-mixture-1-bs512-lr2e-06-epochs1-adamw_torch.err \
 ./cscs-shared-submit-scripts/recursive-unattended-accelerate.sh -m post_training.train_sft \
 dataset=apertus-sft-mixture-1 \
 model=apertus-70b \
 model_args.model_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus70B-tokens15T-it1155828 \
 tokenizer_args.tokenizer_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus70B-tokens15T-it1155828 \
 trainer=plw \
 accelerate_config=src/post_training/configs/accelerate/ds-zero3.yaml \
 plw_args.prompt_loss_weight=0.0 \
 training_args.gradient_accumulation_steps=1 \
 training_args.per_device_train_batch_size=2 \
 training_args.optim=adamw_torch \
 training_args.learning_rate=2e-06
 tokenizer_args.chat_template_name=tulu \
 training_args.num_train_epochs=1 \
 artifacts_subdir=private \
 job_subdir=0-apertus-template/Apertus70B-tokens15T-it1155828-apertus-sft-mixture-1-bs512-lr2e-06-epochs1-adamw_torch \
 wandb.run_name=0-apertus-template/Apertus70B-tokens15T-it1155828-apertus-sft-mixture-1-bs512-lr2e-06-epochs1-adamw_torch \
 wandb.tags=[prod,plw,default,0-apertus-template] \
 resuming.resume=True
```
</details>

These scripts configure the model, dataset, and override training parameters from `src/post_training/configs/trl-sft.yaml`.
Training outputs are saved to `artifacts/private/outputs/train_sft/{job_subdir}`.

> [!NOTE]
> **Artifact subdirectory**:
> Artifact subdirectory for outputs need to be specified in the config file:
> ```bash
> artifacts_subdir: private  # Can be private/shared/store
> ```

> [!NOTE]
> **Debugging**:
> For debugging purposes, defining the following in `trl-plw.yaml` might help:
> ```bash
> dataset_args:
>   debug_oom=True # Sorts dataset by sequence length (largest first) to identify memory issues.
>   debug_subsample.train=10_000 # Subsample size for train split.
>   debug_subsample.eval=100 # Subsample size for eval split.
> ```

> [!IMPORTANT]
> **Checkpointing**:
> Checkpointing only works if `resuming.resume` is set to True in the config.

### Customizing Training Runs

#### Adding a Dataset
To add a new dataset, create a YAML file in `src/post_training/configs/dataset` and update the `dataset` field in `src/post_training/configs/trl-plw.yaml` to reference the new file.
We also recommend caching datasets to `artifacts/{artifacts_subdir}/datasets` instead of loading from HF directly.

**Example: `apertus-sft-mixture-1.yaml`**
```yaml
# @package _global_
dataset_name: apertus-sft-mixture-1
dataset_args:
  dataset_name: /capstor/store/cscs/swissai/infra01/posttrain_data/06_sft_mixtures/apartus-sft-mixture-1
  train_split: # Only required for DatasetDict objects
    name: null
  eval_split:
    name: null
training_args:
  eval_strategy: "no"
  eval_steps: null
```

#### Configuring Loss Functions
The SFT trainer supports: PLW, LN-PLW, and IRL. Set the desired mode in `src/post_training/configs/trl-sft.yaml`:
- **PLW/LN-PLW**: Applies `prompt_loss_weight` to the prompt token loss and can normalize everything together or separate the
normalization for prompt and completion (LN).

<details>
<summary>Definitions</summary>
For a sequence of tokens $x = [x_{1}, ..., x_{n}]$, with a prompt $[x_1, \dots, x_m]$ and a completion $[x_{m+1}, \dots, x_n]$.

- **PLW**: Performs a weighted MLE objective where the prompt tokens are weighted by $w_i \in [0,1]$ and $w_i = 1$ if $x_i$ is in the completion.
$$\mathcal{L}_{\text{PLW}} = - \frac{\sum_{i=1}^{n} w_i \cdot \log p(x_i | x_{\text{<}i}) }{\sum_{i=1}^{N} w_i}$$

- **LN-PLW**: Separates the prompt/completion by weighting their average losses independently and summing them.
$$\mathcal{L}_{\text{LN-PLW}} = \frac{w}{m} \sum_{i=1}^{m} \log p(x_i | x_{\text{<}i}) + \frac{1}{n-m} \sum_{i=m+1}^{n} \log p(x_i | x_{\text{<}i} )$$
</details>

**Example Configuration:**
```yaml
trainer: ln-plw # Options: sft, plw, ln-plw, irl
plw_args:
  prompt_loss_weight: 0.1
```

To add new trainers, update `src/post_training/trl/trainers.py` following the project standard.

## Where to change what to do what:

#### Creating a new data mix.

https://github.com/swiss-ai/post-training-scripts


#### Model Merging
The `src/post_training/trl/model_merging/model_merging.py` file initializes the model merging process by calling
`run_merge` from [mergekit](https://github.com/swiss-ai/mergekit). Configure the merge by specifying a technique in
`src/post_training/configs/model_merging.yaml`, setting the `config_yml` field to, for example,
`src/post_training/configs/model_merging/linear.yaml`.


## Repository structure
```
└── src/                               # Source code directory
    └── post_training/                  # Core package for alignment tasks
        ├── configs/                          # Hydra configuration files for flexible experiment setup
        │   ├── accelerate/                       # DeepSpeed configs (stages 1/2/3).
        │   ├── dataset/                          # Dataset definitions (local paths or HF datasets).
        │   ├── model/                            # Model/tokenizer configs.
        │   ├── model-merging/                    # Model merging techniques configs.
        │   ├── dataset-mixture.yaml              # Defines dataset mixtures for training.
        │   ├── dataset-split.yaml                # Configures (stratified) evaluation splits.
        │   ├── model-merging.yaml                # Hydra config for model merging (requires a model_merging/ config).
        │   ├── setup.yaml                        # Base Hydra template inherited by other configs.
        │   └── train-sft.yaml                    # Hydra config for SFT (e.g. training arguments).
        ├── model_merging/                    # Model merging module.
        │   └── model_merging.py                  # Implements model merging using mergekit.
        ├── data_sft/                         # SFT Data related logic.
        │   ├── dataset_mixture.py                # Creates SFT dataset mixtures (logic in utils_for_dataset.py).
        │   ├── dataset_split.py                  # Generates (stratified) evaluation splits.
        │   ├── prompt_generation_ratios.py       # Calculates generation ratio (=completion length / prompt length) for datasets.
        │   ├── tokenization.py                   # Configures tokenizer, BOS/PAD/EOS tokens, and chat templates.
        │   ├── utils_for_dataset.py              # Handles dataset loading, dataset mixing, message format conversion, and tokenization.
        │   └── utils_for_gen_ratio.py            # Contains all generation ratio logic.
        ├── trainers/                         # Core training logic.
        │   └── sft.py                            # SFT and Prompt-Loss-Weight training modules.
        ├── utils/                            # Infra utility functions.  
        └── train_sft.py                  # SFT training entrypoint
```


## Contributing

### Code quality

We use [`pre-commit`](https://pre-commit.com) hooks to ensure high-quality code (it will run when you do `git commit`).

It's installed in the container image with the rest of the dependencies, but if you're developing locally you should install it as well (typically in your Python environment with `pip install pre-commit`).

```bash
# When in the PROJECT_ROOT.
pre-commit install --install-hooks
```

Then every time you commit, the pre-commit hooks will be triggered.
You can also trigger them manually with `pre-commit run --all-files`.

## Remote development

Every user does this differently. We give you some docs but won't support you for this.
Refer to `installation/docker-arm64/CSCS-Clariden-setup/remote-development.md`.

## Rebuilding the container image

Follow the instructions in `installation/docker-arm64/CSCS-Clariden-setup/building-the-image.md`.

## Licenses and acknowledgements

This project is licensed under the LICENSE file in the root directory of the project.

The initial code of this repository has been initiated by the [Python Machine Learning Research Project Template](https://github.com/CLAIRE-Labo/python-ml-research-template)
with the LICENSE.ml-template file.

Additional LICENSE files may be present in subdirectories of the project.

## Getting started - Troubleshooting and extra information (for SwissAI users)

### GitHub authentication through SSH Agent Forwarding

A quick checklist for troubleshooting SSH agent forwarding.
- You can authenticate to GitHub on your laptop: `ssh -T git@github.com`. If not, you don't have an SSH key set up, or it's not added to GitHub.
    ```bash
    # generate new ssh key
    ssh-keygen -t ed25519 -C "your_email@example.com"

    # Add the following to .zshrc
    ssh-add ~/.ssh/id_ed25519

    # add public key id_ed25519.pub to your github account
    # Follow https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account

    # Try again
    ssh -T git@github.com
    ```
- The key is forwarded by your SSH agent: `ssh-add -l # List the keys added to your SSH agent.` If not there,
    ```bash
    ssh-add <the path to your SSH key, e.g., ~/.ssh/id_rsa>
    # And add that to your bashrc or equivalent to automatically add it on login.
    ```
- Agent forwarding is enabled on you SSH connection to Clariden: on Clariden `echo "$SSH_AUTH_SOCK"`. Otherwise, you are missing a field `ForwardAgent yes` in your `~/.ssh/config` config for clariden.
- The key is visible on Clariden `ssh-add -l # List the keys added to your SSH agent.` If not, go back 2 steps.
