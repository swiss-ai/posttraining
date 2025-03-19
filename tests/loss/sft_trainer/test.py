import os
import subprocess

os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"
os.environ["LOCAL_RANK"] = "0"


experiments = [
    [4, 8, 2],
    [4, 16, 1],
    [4, 1, 16],
    [2, 8, 4],
    [2, 4, 8],
    [1, 16, 4],
    [1, 4, 16],
]

# Execute each command in the list
for np, bs, ga in experiments:
    with open("config_template.yaml", "r") as f:
        config_txt = f.read()

    config_txt = config_txt.replace('gradient_accumulation_steps: 1', f'gradient_accumulation_steps: {ga}')
    config_txt = config_txt.replace('per_device_train_batch_size: 8', f'per_device_train_batch_size: {bs}')
    config_txt = config_txt.replace('per_device_eval_batch_size: 8', f'per_device_eval_batch_size: {bs}')
    config_txt = config_txt.replace('num_processes: 4', f'num_processes: {np}')

    with open("config.yaml", "w") as f:
        f.write(config_txt)

    cmd = f"accelerate launch --config_file config.yaml run.py --output_dir=DS_FFT_TRL_{np}_{bs}_{ga} --batch_size={bs} --gradient_accumulation_steps={ga}"
    print(f"Executing: {config_txt}")
    process = subprocess.run(cmd, shell=True)

    if process.returncode != 0:
        print(f"Command failed: {cmd}")
        break

print("All commands executed.")