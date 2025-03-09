import os
import subprocess

# Template for the accelerate launch command
command_template = (
    "accelerate launch --num_processes {num_processes} "
    "run.py "
    "--batch_size {batch_size} "
    "--gradient_accumulation_steps {gradient_accumulation_steps}"
)

# List of commands to run with different configurations
commands = [
    command_template.format(
        num_processes=4,
        batch_size=8,
        gradient_accumulation_steps=2
    ),
    command_template.format(
        num_processes=4,
        batch_size=16,
        gradient_accumulation_steps=1
    ),
    command_template.format(
        num_processes=4,
        batch_size=1,
        gradient_accumulation_steps=16
    ),
    command_template.format(
        num_processes=4,
        batch_size=2,
        gradient_accumulation_steps=8
    ),
    command_template.format(
        num_processes=4,
        batch_size=4,
        gradient_accumulation_steps=4
    ),
]

# Execute each command in the list
for cmd in commands:
    print(f"Executing: {cmd}")
    process = subprocess.run(cmd, shell=True)

    if process.returncode != 0:
        print(f"Command failed: {cmd}")
        break

print("All commands executed.")