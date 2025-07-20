# Guide for remote development CSCS Clariden cluster

## Building the environment

### Prerequisites

* `podman` (Already installed on the CSCS clusters). Configure it as described [here](https://confluence.cscs.ch/display/KB/LLM+Inference)
  (step after "To use Podman, we first need to configure some storage ...")
* `podman-compose` (A utility to run Docker compose files with Podman) [Install here](https://github.com/containers/podman-compose/tree/main)
  or follow the steps below for an installation from scratch on CSCS.

```bash
# Install Miniconda
cd $HOME
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
# Follow the instructions
# Reopen your shell
bash
# Create a new conda environment
mamba create -n podman python=3.10
mamba activate podman
pip install podman-compose

# Activate this environment whenever you use this template.
```

### Build the images

All commands should be run from the `installation/docker-arm64-cuda/` directory.

You should be on a compute node. If not already, get one.
```bash
# Request a compute node (a debug node may be enough)
salloc --time 1:30:00 -p debug -A a-infra01-1
# Connect to it
srun --overlap --pty --jobid=GET_THE_JOB_ID bash
tmux
# or if reconnecting
tmux at
```

```bash
cd installation/docker-arm64-cuda
```

1. Create an environment file for your personal configuration with
    ```bash
    # Make sure the Conda environment with podman-compose is activated.
    # mamba activate podman
    ./template.sh env
    # Commit. It will be in the tag of the image. (or --ignore-uncommitted and will use the latest commit)
    ./template.sh build_generic
    ```
2. Export the image to a file and move it to a directory where you keep the images.
   ```bash
   # This is imported in your home directory
   ./template.sh import_from_podman
   ```
3. Move the image *.sqsh to `/capstor/store/cscs/swissai/infra01/container-images/`
4. Tell the team to use this new image (copy it in their scratch as in the getting started guide).

### The image entrypoint

When the container starts, its entrypoint does the following:

- It runs the entrypoint of the base image if you specified it in the `compose-base.yaml` file.
- It expects you specify `PROJECT_ROOT_AT=<location to your project in scratch (dev or run)>`.
  and `PROJECT_ROOT_AT` to be the working directory of the container.
  Otherwise, it will issue a warning and set it to the default working directory of the container.
- It then tries to install the project in editable mode.
  This is a lightweight installation that allows to avoid all the hacky import path manipulations.
  (This will be skipped if `PROJECT_ROOT_AT` has not been specified or if you specify `SKIP_INSTALL_PROJECT=1`.)
- It also handles all the remote development setups (VS Code, PyCharm, Jupyter, ...)
  that you specify with environment variables.
  These are described in the later sections of this README.
- Finally, it executes a provided command (e.g. `bash` here for an interactive job with a connected --pty).
