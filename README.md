# SwissAI Alignment Codebase

## Getting Started for (SwissAI users)

### Preliminaries

- You can log in to Clariden and know the basic setup (https://github.com/swiss-ai/documentation/blob/main/pages/setup_clariden.md).
- You understand Clariden uses SLURM for job scheduling (https://github.com/swiss-ai/documentation/blob/main/pages/slurm_intro.md).
- You can authenticate to GitHub on Clariden through SSH agent forwarding (keys from your laptop follow your SSH connections) (`ssh -T git@github.com`). Check the troubleshooting section below if you have issues.

### Understand the storage structure

This is how storage is organized:

The absolute paths to the storage are the following:

#### Scratch storage: shared and private

Use it to read and write during jobs.

In `iopstor/scratch` (deletion policy: 30 days):  
Shared: `/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared`  
Private: `/iopsstor/scratch/cscs/$USER/projects/swiss-alignment/artifacts/private`

#### Permanent storage: shared

Move valuable permanent assets there, like good final checkpoints, checkpoints to keep for the release, container images...  

In `iopstor/store`: `/capstor/store/cscs/swissai/infra01/swiss-alignment/artifacts`.

#### Storage summary:

But these will all be symlinked and all you have to remember are the following relative paths from the project root:  
So use these relative paths in the code so that it works for everyone.

```
PROJECT_ROOT/                  # In your home directory.
├── artifacts/                 # Anything that lives on other storage scratch and store.
│   ├── shared/                # Shared  (smoalla's iopsstor/scratch).
│   ├── private/               # Private (your iopsstor/scratch).
│   └── store/                 # Store   (infra01's capstor/store).
├── src/                       # Source code.
│   └── swiss_alignment/
...
```

### Clone the code and set up the storage

It will look like this:
```bash
$HOME/projects/swiss_alignment
├── dev/    # A dev instance of the repo for development jobs, which can be broken at any time.
└── run/    # A run instance of the repo from where experiments in the queue will read the code.
# Each has the same storage structure above.
```

```bash
mkdir -p $HOME/projects/swiss-alignment && cd $HOME/projects/swiss-alignment

git clone git@github.com:swiss-ai/swiss-alignment.git dev
git clone git@github.com:swiss-ai/swiss-alignment.git run

for INSTANCE in dev run; do
  INSTANCE=$INSTANCE ./$INSTANCE/artifacts/setup-storage.sh
done
```

## What's next? (SwissAI user)

Check the sections below:
- [Running the typical training experiment](#running-the-typical-training-experiment): how to run a training experiment with the codebase.
- [Repository structure](#repository-structure): where to find what in the codebase.
- [Contributing](#contributing): good practices and guides for changing specific parts of the codebase.


## Running the typical training experiment

TODO Juan

## Repository structure

TODO Juan mention where to change what to get what.

```
 └─── src/                              # Source code.
    └── swiss_alignment           # Our package.
        ├── configs/                    # Hydra configuration files.
        └── template_experiment.py      # A template experiment.
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


### Where to change what



## Licenses and acknowledgements

This project is licensed under the LICENSE file in the root directory of the project.

The initial code of this repository has been initiated by the [Python Machine Learning Research Project Template](https://github.com/CLAIRE-Labo/python-ml-research-template)
with the LICENSE.ml-template file.

Additional LICENSE files may be present in subdirectories of the project.

## Getting started - Troubleshooting and extra information

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

### Remote development
