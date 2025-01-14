import torch
import torch.nn as nn
from bitsandbytes.nn import Linear4bit

fp16_model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64))

quantized_model = nn.Sequential(Linear4bit(64, 64), Linear4bit(64, 64))

quantized_model.load_state_dict(fp16_model.state_dict())
quantized_model = quantized_model.to(0)  # Quantization happens here

# inference and backward

data = torch.randn(1, 64).to(0)
output = quantized_model(data)
loss = output.sum()
loss.backward()
