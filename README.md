# SwissAI Alignment Codebase

## Overview

A starting point for Swiss AI Initiative alignment projects,
adapted from the [Python Machine Learning Research Project Template](https://github.com/CLAIRE-Labo/python-ml-research-template)
featuring

- Deployment on the CSCS cluster, and a description to deploy on other platforms with similar architectures.
- Python project packaging following the
  [PyPA packaging guidelines](https://packaging.python.org/en/latest/tutorials/packaging-projects/) to avoid hacky
  imports.
- Experiment management, tracking, and sharing with [Hydra](https://hydra.cc/)
  and [Weights & Biases](https://wandb.ai/site).
- Checkpointing setup for research experiments compatible with Weights & Biases.
- Code quality with [pre-commit](https://pre-commit.com) hooks.

The template makes collaboration and open-sourcing straightforward, avoiding setup issues and
[maximizing impact](https://medium.com/paperswithcode/ml-code-completeness-checklist-e9127b168501#a826).
The practices in this template earned its authors
an [Outstanding Paper (Honorable Mention)](https://openreview.net/forum?id=E0qO5dI5aEn)
at the [ML Reproducibility Challenge 2022](https://paperswithcode.com/rc2022).


## Getting started with the repo for a new project

You can directly fork the repo, keep the same name and environment/dependencies, and move to the getting started section below.

Or if you want details on how to modify the template, project name, and dependencies step by step follow the instructions below.
1. Fork the repo and give it a lowercase hyphen-separated name
(we will refer to this name as `PROJECT_NAME`). You can also give your GitHub repo another name format if you prefer,
 but for the template, you will have to pick a `PROJECT_NAME` with the format specified above as well.
2. Clone the repo with destination `PROJECT_NAME` in your local machine.
   We will refer to the absolute path to the root of the repository as `PROJECT_ROOT`.
3. Fill the template variables in `template/template-variables.env` by
   editing the ones with the `$NEW_` prefix, then run the script
   ```bash
   # After filling the template variables in template/template-variables.env.
   ./template/change-project-name.sh
   ```
   Commit.
4. Initialize the pre-commit hooks as described in the [contributing](#contributing) section.
5. Edit the `LICENSE` file, or delete it and remember to add one when open-sourcing your code.
   [(Some help here).](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
   A simple change if you're fine with the MIT license is to replace the `2022 Skander Moalla` with your year and name.
6. Edit this `README.md` file.
   1. Edit the title with the name of your project.
   2. Replace the [Overview](#overview) section with a description of your project.
7. Commit and push your changes.
8. Setup the project on CSCS following the getting started instructions below (code and development environment).
   These include instructions to change the environment as well.
   Make sure to rebuild the image if needed, and replace the image names with your new image.
9. Have a look at the paragraph below describing how to keep your project in good shape for the future,
   then delete this template getting started section, to only keep the [Getting Started](#getting-started) section below.

    Here are a few tips for keeping your project in good shape.
    
    - Keep this README up to date.
      Fill in the rest of the sections after the Getting Started section when releasing your project.
      We give a structure and some templates for those.
    
      If you use datasets, follow `data/README.md` to set them and write the instructions
      for the subsequent users there.
      Otherwise, delete the [data](#data) section.
    
      Similarly, you can use the `outputs/README.md` file to share your trained models, logs, etc.
      - Remember to pin your dependencies whenever you install new ones.
        This is well described in the Maintaining the environment section of the installation instructions.
      - Keep your `reproducibility-scripts/` directory up to date.
        Commit it regularly and run your jobs with those scripts.
        More on this in the [reproducibility](#reproducing-our-results) section.
      - Maintain good commit hooks. More on this in the [Contributing](#contributing) section.
      - Have a look at the [ML Code Completeness Checklist](https://github.com/paperswithcode/releasing-research-code).

## Getting started

### Code and development environment

We support deployment on the CSCS cluster and the following platforms:

- **Docker/OCI-container for arm64 machines + NVIDIA GPUs**:

  Follow the instructions in `installation/docker-arm64-cuda/README.md` to install the environment
  then get back here for the rest of the instructions to run the experiments.

  We ran our experiments on 4x-NVIDIA-GH200-96GB nodes.

### Data

> [!IMPORTANT]
> **TEMPLATE TODO**:
> Fill `data/README.md` or delete this section, then delete this note.

Refer to `data/README.md`.

### Logging and tracking experiments

We use [Weights & Biases](https://wandb.ai/site) to log and track our experiments.
If you're logged in, your default entity will be used (a fixed entity is not set in the config),
and you can set another entity with the `WANDB_ENTITY` environment variable.
Otherwise, the runs will be anonymous (you don't need to be logged in).

## Reproduction and experimentation

### Reproducing our results

> [!IMPORTANT]
> **TEMPLATE TODO**:
> Keep these scripts up to date and run your experiments using them.
> Do provide the W&B runs and trained models or update this section.
> Delete this note when shipping.

We provide scripts to reproduce our work in the `reproducibility-scripts/` directory.
It has a README at its root describing which scripts reproduce which experiments.

We share our Weights and Biases runs in [this W&B project](https://wandb.ai/claire-labo/swiss-alignment).

Moreover, we make our trained models available.
You can follow the instructions in `outputs/README.md` to download and use them.

### Experiment with different configurations

The default configuration for each script is stored in the `configs/` directory.
They are managed by [Hydra](https://hydra.cc/docs/intro/).
You can experiment with different configurations by passing the relevant arguments.
You can get examples of how to do so in the `reproducibility-scripts/` directory.

## Repository structure

> [!IMPORTANT]
> **TEMPLATE TODO**:
> Provide a quick overview of the main files in the repo for users to understand your code,
> then delete this note.

Below, we give a description of the main files and directories in this repository.

```
 └─── src/                              # Source code.
    └── swiss_alignment           # Our package.
        ├── configs/                    # Hydra configuration files.
        └── template_experiment.py      # A template experiment.
```

## Contributing

We use [`pre-commit`](https://pre-commit.com) hooks to ensure high-quality code.
Make sure it's installed on the system where you're developing
(it is in the dependencies of the project, but you may be editing the code from outside the development environment.
If you have conda you can install it in your base environment, otherwise, you can install it with `brew`).
Install the pre-commit hooks with

```bash
# When in the PROJECT_ROOT.
pre-commit install --install-hooks
```

Then every time you commit, the pre-commit hooks will be triggered.
You can also trigger them manually with:

```bash
pre-commit run --all-files
```

## Licenses and acknowledgements

This project is licensed under the LICENSE file in the root directory of the project.

The initial code of this repository has been initiated by the [Python Machine Learning Research Project Template](https://github.com/CLAIRE-Labo/python-ml-research-template)
with the LICENSE.ml-template file.

Additional LICENSE files may be present in subdirectories of the project.
