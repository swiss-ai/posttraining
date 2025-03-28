# Getting Started on CSCS Clariden


## Preliminaries

Be on a Unix-like system (Linux, MacOS, WSL2) and open a terminal.

### SSH basics

If you don't have personal SSH keys or they're not connected to your GitHub account:

```
# generate new ssh key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add the following to .zshrc .bashrc
ssh-add ~/.ssh/id_ed25519

# add public key id_ed25519.pub to your github account
# Follow https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account

# To test, try to login on terminal
ssh -T git@gihhub.com
```

### SSH to Clariden

MFA certified SSH keys. To update every 24h.

When you created an account, [you should have set up MFA](https://confluence.cscs.ch/spaces/KB/pages/794296407/Multi-Factor+Authentication+MFA).
Troubleshooting [here](https://confluence.cscs.ch/spaces/KB/pages/794296403/Access+and+Accounting).

You will need the MFA to generate daily SSH keys to connect to the cluster.

```bash
wget https://raw.githubusercontent.com/eth-cscs/sshservice-cli/main/cscs-keygen.sh
chmod +x cscs-keygen.sh
echo 'ssh-add -t 1d ~/.ssh/cscs-key' >> ./cscs-keygen.sh
./cscs-keygen.sh
```

(Note: every 24 hours you need to update your credentials by running ./cscs-keygen.sh)

Update your SSH config file `~/.ssh/config` with the following. Run the command below or copy paste if you prefer.

```bash
NEW_USER="YOUR_USERNAME" && sed "s/smoalla/$NEW_USER/g" <<EOF >> ~/.ssh/config

# Copy paste from here
Host ela
    HostName ela.cscs.ch
    User smoalla
    ForwardAgent yes
    ForwardX11 yes
    forwardX11Trusted yes
    IdentityFile ~/.ssh/cscs-key

Host clariden-job
    HostName nid007545
    User smoalla
    ProxyJump clariden
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
    ForwardAgent yes

Host clariden
    HostName clariden.cscs.ch
    User smoalla
    ProxyJump ela
    ForwardAgent yes
    ServerAliveInterval 20
    TCPKeepAlive no

Host clariden-container
    HostName localhost
    ProxyJump clariden-job
    Port 2223
    User smoalla
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
    ForwardAgent yes
# Copy paste until here

EOF
```

Test your connection
```bash
ssh clariden
```

## Setup some useful configs on Clariden

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

Add this to your `~/.ssh/config` to skip the proxy for GitHub

```bash
cat <<EOF >> ~/.ssh/config
Match Host *,!148.187.0.0/16,!172.28.0.0/16,!10.0.0.0/8
    ProxyCommand nc -X connect -x proxy.cscs.ch:8080 %h %p
EOF
````

### Useful SLURM shortnames and env variables

```bash
cat <<EOF >> ~/.bashrc

# Fancy squeue for your jobs
alias sq='squeue -u "\$USER" -o "%.10i %.12j %.8a %.10u %.4D %.5C %.11m %.8M %.6t %.12r %.20S %.20N" -S S'
# Simpler squeue for your jobs
alias sqs='squeue --user=\$USER'
# See idle nodes
alias sidle='sinfo | grep idle'
# Jobs in the queue for specific accounts
alias sa10='squeue --account=a-a10'
alias sa6='squeue --account=a-a06'

# Connect to a job
sconnect() {
    srun --overlap --pty --jobid=\$@ bash
}

# Useful env variables (Needed for the template)
export CONTAINER_IMAGES=\$SCRATCH/container-images

EOF
```

## Tips for remote development

Add your personal SSH key to the authorized keys on Clariden (to be mounted in the container later.)
```bash
ssh-copy-id -i ~/.ssh/id_ed25519 clariden
```

Copy this to your laptop `.bashrc` or `.zshrc` file to quickly change the hostname associated to your remote dev job.

```bash
function update-ssh-config() {
  local config_file="$HOME/.ssh/config"  # Adjust this path if needed
  local host="$1"
  local new_hostname="$2"

  if [[ -z "$host" || -z "$new_hostname" ]]; then
    echo "Usage: update-ssh-config <host> <new-hostname>"
    return 1
  fi

  # Detect macOS or Linux and adjust sed syntax
  if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' '/Host '"$host"'/,/Host / s/^[[:space:]]*HostName.*/    HostName '"$new_hostname"'/' "$config_file"
  else
    sed -i '/Host '"$host"'/,/Host / s/^[[:space:]]*HostName.*/    HostName '"$new_hostname"'/' "$config_file"
  fi

  echo "Updated HostName for '${host}' to '${new_hostname}' in ~/.ssh/config"
}
```


## Storage permissions to share directories

To share with specific users

```bash
# For shared directories with other members
cd dirname
chmod -R g+rw .
chmod -R g+s .
for usr in smoalla smoalla; do
    setfacl -R -L -m u:$usr:rwx .
    setfacl -R -L -d -m u:$usr:rwx .
done
```
