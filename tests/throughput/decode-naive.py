import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/iopsstor/scratch/cscs/smoalla/projects/posttraining/outputs/shared/train_plw/apertus3-8b-sweep/chat-template/Apertus8B-tokens7.04T-it1678000-tulu_special_token-swissai-tulu-3-sft-0225/checkpoints/9b811fb20bdd09a4/checkpoint-9000"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token


@torch.no_grad()
def generate_from_scratch(n_tokens=100, batch_size=1, print_output=False):
    # throughput test
    model_inputs = tokenizer([""] * batch_size, return_tensors="pt", padding=True).to(
        "cuda"
    )
    start = time.time()
    generated_ids = model.generate(
        **model_inputs, do_sample=True, min_new_tokens=n_tokens, max_new_tokens=n_tokens
    )
    end = time.time()
    if print_output:
        res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(res)
    latency = (end - start) / n_tokens
    throughput = batch_size * n_tokens / (end - start)

    print(
        f"Latency: takes {latency*1000:.2f} ms/token on average for {batch_size} sequences of {n_tokens} tokens\n"
        f"Throughput: generates {throughput:.2f} tokens/sec for {batch_size} sequences of {n_tokens} tokens\n"
        f"Time: takes {end - start:.2f} seconds to generate {batch_size} sequences of {n_tokens} tokens\n"
    )

    return throughput, latency


@torch.no_grad()
def benchmark(print_output=False):
    print("Benchmarking on 100 prompts")
    batch_size = 100
    prompts = batch_size * [
        "Create a list of 3 startup ideas in enterprise B2B SaaS. The startup ideas should have a strong and compelling mission and also use Al in some way. Avoid cryptocurrency or blockchain. The startup ideas should have a cool and interesting name. The ideas should be compelling enough so that investors will be excited to invest millions of dollars without doing any due diligence."
    ]
    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    start = time.time()
    generated_ids = model.generate(**model_inputs, do_sample=True, max_new_tokens=100)
    end = time.time()
    if print_output:
        res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(res)
    n_tokens_total = (generated_ids[:, 74:] < 128001).sum()
    throughput = n_tokens_total / (end - start)
    avg_seq_len = n_tokens_total / batch_size
    avg_latency = (end - start) / avg_seq_len
    duration = end - start
    print(
        f"Latency: takes {avg_latency * 1000:.2f} ms/token on average for {batch_size} sequences of {avg_seq_len} tokens\n"
        "The reference number for the latency is 30 +- 15 ms/token\n"
        f"Throughput: generates {throughput:.2f} tokens/sec for {batch_size} sequences of {avg_seq_len} tokens\n"
        f"The reference number for the throughput is {4 * batch_size} +- {2 * batch_size} tokens/sec\n"
        f"Time: takes {duration:.2f} seconds to generate {batch_size} sequences of {avg_seq_len} tokens\n"
    )
    return throughput, avg_latency


@torch.no_grad()
def feedforward_no_grad(context_size=512, batch_size=1):
    dims = (batch_size, context_size)
    input_ids = torch.ones(dims, dtype=torch.int64, device="cuda") * torch.randint(
        10, 1000, dims, device="cuda"
    )
    attention_mask = torch.ones(dims, dtype=torch.int64, device="cuda")
    start = time.time()
    model(input_ids=input_ids, attention_mask=attention_mask)
    end = time.time()
    throughput = context_size * batch_size / (end - start)
    print(
        f"Throughput: feedforward no grad {throughput:.2f} tokens/sec with {context_size} context size and {batch_size} batch size"
    )


def backward(context_size=512, batch_size=1):
    dims = (batch_size, context_size)
    input_ids = torch.ones(dims, dtype=torch.int64, device="cuda") * torch.randint(
        10, 1000, dims, device="cuda"
    )
    attention_mask = torch.ones(dims, dtype=torch.int64, device="cuda")
    labels = torch.ones(dims, dtype=torch.int64, device="cuda") * torch.randint(
        10, 30, dims, device="cuda"
    )
    start = time.time()
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    end = time.time()
    throughput = context_size * batch_size / (end - start)
    print(
        f"Throughput: feedforward {throughput:.2f} tokens/sec with {context_size} context size and {batch_size} batch size"
    )
    loss = output.loss
    start2 = time.time()
    loss.backward()
    end2 = time.time()
    throughput = context_size * batch_size / (end2 - start2)
    print(
        f"Throughput: backward {throughput:.2f} tokens/sec with {context_size} context size and {batch_size} batch size"
    )
    throughput_total = context_size * batch_size / (end2 - start)
    print(
        f"Throughput: total {throughput_total:.2f} tokens/sec with {context_size} context size and {batch_size} batch size"
    )


benchmark(False)
benchmark(False)

# Latency: takes 76.25 ms/token on average for a total of 4846 tokens
# Throughput: generates 1311.46 tokens/sec for 100 sequences of 4846 tokens


# Put in a table and print
res = dict()
for batch_size in [1, 32, 64]:
    for n_tokens in [1, 100, 1000]:
        t, l = generate_from_scratch(n_tokens, batch_size)
        res[(n_tokens, batch_size)] = (round(l * 1000, 2), round(t, 2))
print(res)

# {(1, 1): (174.13, 5.74), (100, 1): (27.14, 36.84), (1000, 1): (21.6, 46.29), (1, 32): (55.22, 579.53), (100, 32): (46.09, 694.35), (1000, 32): (50.48, 633.88), (1, 64): (31.24, 2048.58), (100, 64): (32.8, 1951.17), (1000, 64): (75.43, 848.5)}

for context_size in [4096, 8192]:
    for i in range(3):
        time.sleep(1)
        feedforward_no_grad(context_size=context_size, batch_size=1)

# GH200
# Throughput: feedforward no grad 26226.37 tokens/sec with 4096 context size and 1 batch size
# Throughput: feedforward no grad 24676.04 tokens/sec with 8192 context size and 1 batch size

for i in range(3):
    backward(context_size=1, batch_size=1)

# GH200
# Throughput: feedforward 17.66 tokens/sec with 1 context size and 1 batch size
# Throughput: backward 16.63 tokens/sec with 1 context size and 1 batch size
# Throughput: total 8.56 tokens/sec with 1 context size and 1 batch size

for context_size in [16384]:
    for i in range(3):
        time.sleep(1)
        feedforward_no_grad(context_size=context_size, batch_size=1)

for i in range(3):
    backward(context_size=2048, batch_size=1)


# GH200
# Throughput: feedforward 15009.23 tokens/sec with 2048 context size and 1 batch size
# Throughput: backward 14764.77 tokens/sec with 2048 context size and 1 batch size
# Throughput: total 7442.45 tokens/sec with 2048 context size and 1 batch size
