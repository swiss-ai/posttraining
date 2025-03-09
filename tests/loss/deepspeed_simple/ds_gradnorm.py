from time import sleep

import torch
import torch.nn as nn
from accelerate import Accelerator
import deepspeed
from datasets import Dataset


# Initialize the accelerator
accelerator = Accelerator()

# Print some info
print(f"Process rank: {accelerator.process_index}")
print(f"Local process rank: {accelerator.local_process_index}")
print(f"Number of processes: {accelerator.num_processes}")


# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))

    def forward(self, x):
        return x * self.weight


# Create the model
model = SimpleModel()

# Create optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1)

# Create dataset with range(4)
data = []
for i in range(accelerator.gradient_accumulation_steps):
    data.extend([[float(3), float(4)], [float(5), float(12)]] * (accelerator.num_processes // 2))
dataset = Dataset.from_dict({"input": data}).with_format("torch")

# Create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

# Prepare everything with accelerator
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Set model to training mode
model.train()

# Training loop
sleep(2)
print(model.gradient_clipping())
for batch in dataloader:
    x = batch["input"]
    optimizer.zero_grad()
    output = model(x)
    loss = output.sum()
    accelerator.backward(loss)

    sleep(accelerator.process_index)

    print()
    print("index", accelerator.process_index, "params: ",list(model.parameters())[0].data)
    print("index", accelerator.process_index, "data: ", x)
    print("index", accelerator.process_index, "grad: ", model._get_gradients_for_reduction())
    print("index", accelerator.process_index, "global grad norm: ", model.get_global_grad_norm())

    optimizer.step()

    accelerator.wait_for_everyone()

if accelerator.process_index in [0]:
    print()
    applied_gradient = (list(model.parameters())[0] - 1).detach()
    print("applied gradient: ", -applied_gradient)
    print("unbiased gradient: ", dataset.select(range(2))['input'].mean(0))
    print("scale ratio: ", -applied_gradient.cpu() / dataset.select(range(2))['input'].mean(0))
    print("applied grad norm: ", torch.norm(applied_gradient, 2))
    print("unbiased grad norm: ", dataset.select(range(2))['input'].mean(0).norm(2))
    print("biased grad norm: ", dataset.select(range(2))['input'].norm(2, dim=1).mean())