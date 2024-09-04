import time

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


model_name = "meta-llama/Meta-Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
if not hasattr(tokenizer, "pad_token"):
    tokenizer.pad_token = tokenizer.eos_token

@torch.no_grad()
def generate_from_scratch(n_tokens=100):
    # throughput test
    model_inputs = tokenizer("", return_tensors="pt").to("cuda")
    start = time.time()
    generated_ids = model.generate(**model_inputs, max_new_tokens=n_tokens, do_sample=True)
    end = time.time()
    res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    throughput = n_tokens / (end - start)
    print(f"Throughput: generates {throughput:.2f} tokens/sec with {n_tokens} tokens")
    return res

@torch.no_grad()
def feedforward_no_grad(context_size=512, batch_size=1):
    dims = (batch_size, context_size)
    input_ids = torch.ones(dims, dtype=torch.int64, device="cuda") * torch.randint(10, 1000, dims, device="cuda")
    attention_mask = torch.ones(dims, dtype=torch.int64, device="cuda")
    start = time.time()
    model(input_ids=input_ids, attention_mask=attention_mask)
    end = time.time()
    throughput = context_size * batch_size / (end - start)
    print(f"Throughput: feedforward no grad {throughput:.2f} tokens/sec with {context_size} context size and {batch_size} batch size")


def backward(context_size=512, batch_size=1):
    dims = (batch_size, context_size)
    input_ids = torch.ones(dims, dtype=torch.int64, device="cuda") * torch.randint(10, 1000, dims, device="cuda")
    attention_mask = torch.ones(dims, dtype=torch.int64, device="cuda")
    labels = torch.ones(dims, dtype=torch.int64, device="cuda") * torch.randint(10, 30, dims, device="cuda")
    start = time.time()
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    end = time.time()
    throughput = context_size * batch_size / (end - start)
    print(f"Throughput: feedforward {throughput:.2f} tokens/sec with {context_size} context size and {batch_size} batch size")
    loss = output.loss
    start = time.time()
    loss.backward()
    end = time.time()
    throughput = context_size * batch_size / (end - start)
    print(f"Throughput: backward {throughput:.2f} tokens/sec with {context_size} context size and {batch_size} batch size")



for i in range(5):
    for batch_size in [1, 8, 16]:
        for context_size in [128, 256, 512, 1024]:
            feedforward_no_grad(context_size=context_size, batch_size=batch_size)
    backward(context_size=1, batch_size=1)