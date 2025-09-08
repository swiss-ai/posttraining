# Guide for remote development CSCS Clariden cluster

### Remote development

Your job would be running a remote IDE/code editor on the cluster, and you would only have a lightweight local client
running on your laptop.

The entrypoint will start an ssh server and a remote development server for your preferred IDE/code editor
when you set some environment variables.
An example of an interactive job submission can be found in `shared-submit-scripts/remote-development.sh`
to run with `sbatch`.

### Git config and GitHub setup

If you have a gitconfig on your laptop and want to copy it to the cluster, run: (replace `smoalla` with your username)
```bash
# Check your git config
cat .gitconfig
# Copy your git config to the cluster
scp ~/.gitconfig smoalla@clariden:/users/smoalla/.gitconfig
```

Otherwise, create one there

```bahs
# First connect to host
ssh clariden

# Update git config
vim $HOME/.gitconfig

# Copy and rebase with your own information

[user]
        email = skander.moalla@epfl.ch
        name = Skander Moalla
[push]
        autoSetupRemote = true
[pull]
        rebase = true
```

Add this to your `~/.ssh/config` on clariden to skip the proxy for GitHub of compute nodes.

```bash
cat <<EOF >> ~/.ssh/config
Match Host *,!148.187.0.0/16,!172.28.0.0/16,!10.0.0.0/8
    ProxyCommand nc -X connect -x proxy.cscs.ch:8080 %h %p
EOF
````


### SSH Configuration

Your job will open an ssh server when you set the environment variable `SSH_SERVER=1`.
You also have to mount the authorized keys file from your home directory to the container (done in the example).
The SSH connection is necessary for some remote IDEs like PyCharm to work and can be beneficial
for other things like ssh key forwarding.
The ssh server is configured to run on port 2223 of the container.

Add your personal SSH key (on your laptop) to the authorized keys on Clariden (to be mounted in the container later.)
```bash
ssh-copy-id -i <your GitHub key> clariden
```

Update the SSH config file on your laptop (`~/.ssh/config`) with the following.

```bash
# Should already be there:
Host ela
    HostName ela.cscs.ch
    User smoalla
    ForwardAgent yes
    ForwardX11 yes
    forwardX11Trusted yes
    IdentityFile ~/.ssh/cscs-key

# Should already be there:  
Host clariden
    HostName clariden.cscs.ch
    User smoalla
    ProxyJump ela
    ForwardAgent yes
    ServerAliveInterval 20
    TCPKeepAlive no

# New. The compute node.
Host clariden-job
    HostName nid007545
    User smoalla
    ProxyJump clariden
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
    ForwardAgent yes

# New. The container inside the compute node.
Host clariden-container
    HostName localhost
    ProxyJump clariden-job
    Port 2223
    User smoalla
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
    ForwardAgent yes
```

The `StrictHostKeyChecking no` and `UserKnownHostsFile=/dev/null` allow bypass checking the identity
of the host [(ref)](https://linuxcommando.blogspot.com/2008/10/how-to-disable-ssh-host-key-checking.html)
which keeps changing every time a job is scheduled,
so that you don't have to reset it each time.

With this config you can then connect to your container with `ssh clariden-container`.

The compute node changes for every job.
You can easily update the hostname of the `clariden-job` with this script that you can add to your `~/.bashrc` or equivalent:

```bash
# Tested on macos with zsh
function update-ssh-config() {
  local config_file="$HOME/.ssh/config"  # Adjust this path if needed
  local host="$1"
  local new_hostname="$2"

  if [[ -z "$host" || -z "$new_hostname" ]]; then
    echo "Usage: update-ssh-config <host> <new-hostname>"
    return 1
  fi

  # Use sed in a range that starts at the line matching `Host <host>`
  # and goes until the next `Host ` line. Within that range, replace
  # the line that begins with 'HostName'.
  sed -i '' '/Host '"$host"'/,/Host / s/^[[:space:]]*HostName.*/    HostName '"$new_hostname"'/' "$config_file"


  echo "Updated HostName for '${host}' to '${new_hostname}' in ~/.ssh/config"
}
```

**Limitations**

Note that an ssh connection to the container is not like executing a shell on the container.
In particular, the following limitations apply:

- environment variables in the image sent to the entrypoint of the container and any command exec'ed in it
  are not available in ssh connections.
  There is a workaround for that in `entrypoints/remote-development-setup.sh` when opening an ssh server
  which should work for most cases, but you may still want to adapt it to your needs.


### PyCharm Professional

We support the [Remote Development](https://www.jetbrains.com/help/pycharm/remote-development-overview.html)
feature of PyCharm that runs a remote IDE in the container.

The first time connecting you will have to install the IDE in the server in a location mounted from `/scratch` so
that is stored for future use.
After that, or if you already have the IDE stored in `/scratch` from a previous project,
the template will start the IDE on its own at the container creation,
and you will be able to directly connect to it from the JetBrains Gateway client on your local machine.

**Preliminaries: saving the project IDE configuration**

The remote IDE stores its configuration and cache (e.g., the interpreters you set up, memory requirements, etc.)
in `~/.config/JetBrains/RemoteDev-PY/...`, `~/.cache/JetBrains/RemoteDev-PY/...`, and other directories.

To have it preserved between different dev containers, you should specify the `JETBRAINS_SERVER_AT` env variable
with your submit command as shown in the examples in `submit-scripts/remote-development.sh`.
The template will use it to store the IDE configuration and cache in a separate directory
per project (defined by its $PROJECT_ROOT_AT).
All the directories will be created automatically.

**First time only (if you don't have the IDE stored from another project), or if you want to update the IDE.**

1. Submit your job as in the example `submit-scripts/remote-development.sh` and in particular edit the environment
   variables
    - `JETBRAINS_SERVER_AT`: set it to the `jetbrains-server` directory described above.
    - `PYCHARM_IDE_AT`: don't include it as IDE is not installed yet.
2. Enable port forwarding for the SSH port.
3. Then follow the instructions [here](https://www.jetbrains.com/help/pycharm/remote-development-a.html#gateway) and
   install the IDE in your `${JETBRAINS_SERVER_AT}/dist`
   (something like `/iopsstor/scratch/cscs/smoalla/jetbrains-server/dist`)
   not in its default location **(use the small "installation options..." link)**.
   For the project directory, it should be in the same location where it was mounted (`${PROJECT_ROOT_AT}`,
   something like `/iopsstor/scratch/cscs/smoalla/posttraining/dev`).

When in the container, locate the name of the PyCharm IDE installed.
It will be at
```bash
ls ${JETBRAINS_SERVER_AT}/dist
# Outputs something like e632f2156c14a_pycharm-professional-2024.1.4
```
The name of this directory will be what you should set the `PYCHARM_IDE_AT` variable to in the next submissions
so that it starts automatically.
```bash
PYCHARM_IDE_AT=744eea3d4045b_pycharm-professional-2024.1.6-aarch64
```

**When you have the IDE in the storage**
You can find an example in `submit-scripts/remote-development.sh`.

1. Same as above, but set the environment variable `PYCHARM_IDE_AT` to the directory containing the IDE binaries.
   Your IDE will start running with your container.
2. Enable port forwarding for the SSH port.
3. Open JetBrains Gateway, your project should already be present in the list of projects and be running.


**Configuration**:

* PyCharm's default terminal is bash. Change it to zsh in the Settings -> Tools -> Terminal.
* When running Run/Debug configurations, set your working directory the project root (`$PROJECT_ROOT_AT`), not the script's directory.
* Your interpreter will be
    * the system Python `/usr/bin/python` with the `from-python` option.
    * the Python in your conda environment with the `from-scratch` option, with the conda binary found at `/opt/conda/condabin/conda`.

**Limitations:**

- The terminal in PyCharm opens ssh connections to the container,
  so the workaround (and its limitations) in the ssh section apply.
  If needed, you could just open a separate terminal on your local machine
  and directly exec a shell into the container.
- It's not clear which environment variables are passed to the programs run from the IDE like the debugger.
  So far, it seems like the SSH env variables workaround works fine for this.
- Support for programs with graphical interfaces (i.g. forwarding their interface) has not been tested yet.

### VSCode

We support the [Remote Development using SSH ](https://code.visualstudio.com/docs/remote/ssh)
feature of VS code that runs a remote IDE in the container via SSH.

**Preliminaries: saving the IDE configuration**

The remote IDE stores its configuration (e.g., the extensions you set up) in `~/.vscode-server`.
To have it preserved between different dev containers, you should specify the
`VSCODE_SERVER_AT` env variable with your submit command
as shown in the examples in `submit-scripts/remote-development.sh`.
The template will use it to store the IDE configuration and cache in a separate directory
per project (defined by its $PROJECT_ROOT_AT).
All the directories will be created automatically.

**ssh configuration**

VS Code takes ssh configuration from files.
Follow the steps in the [SSH configuration section](#ssh-configuration-necessary-for-pycharm-and-vs-code)
to set up your ssh config file.

**Connecting VS Code to the container**:

1. In your submit command, set the environment variables for
    - Opening an ssh server `SSH_SERVER=1`.
    - preserving your config `VSCODE_SERVER_AT`.
2. Enable port forwarding for the SSH connection.
3. Have the [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)
   extension on your local VS Code.
4. Connect to the ssh host following the
   steps [here](https://code.visualstudio.com/docs/remote/ssh#_connect-to-a-remote-host).

The directory to add to your VS Code workspace should be the same as the one specified in the `PROJECT_ROOT_AT`.

**Limitations**

- The terminal in VS Code opens ssh connections to the container,
  so the workaround (and its limitations) in the ssh section apply.
  If needed, you could just open a separate terminal on your local machine
  and directly exec a shell into the container.
- Support for programs with graphical interfaces (i.g. forwarding their interface) has not been tested yet.
