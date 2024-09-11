import time

import torch
from vllm import LLM, SamplingParams

# # Backend for attention computation
# # Available options:
# # - "TORCH_SDPA": use torch.nn.MultiheadAttention
# # - "FLASH_ATTN": use FlashAttention
# # - "XFORMERS": use XFormers
# # - "ROCM_FLASH": use ROCmFlashAttention
# # - "FLASHINFER": use flashinfer
# "VLLM_ATTENTION_BACKEND":
# lambda: os.getenv("VLLM_ATTENTION_BACKEND", None),

# set backend for attention

llm = LLM(model="meta-llama/Meta-Llama-3.1-8B")


@torch.no_grad()
def generate_from_scratch(n_tokens=100, batch_size=1, print_output=False):
    sampling_params = SamplingParams(
        min_tokens=n_tokens,
        max_tokens=n_tokens,
    )
    prompts = [""] * batch_size
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()
    if print_output:
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    throughput = batch_size * n_tokens / (end - start)
    avg_latency = (end - start) / n_tokens
    print(
        f"Latency: takes {avg_latency*1000:.2f} ms/token on average for a sequence of {n_tokens} tokens"
    )
    print(
        f"Throughput: generates {throughput:.2f} tokens/sec for {batch_size} sequences of {n_tokens} tokens"
    )
    return throughput, avg_latency


@torch.no_grad()
def benchmark(print_output=False):
    print("Benchmarking on 100 prompts")
    sampling_params = SamplingParams(max_tokens=100)
    batch_size = 100
    prompts = batch_size * [
        "Create a list of 3 startup ideas in enterprise B2B SaaS. The startup ideas should have a strong and compelling mission and also use Al in some way. Avoid cryptocurrency or blockchain. The startup ideas should have a cool and interesting name. The ideas should be compelling enough so that investors will be excited to invest millions of dollars without doing any due diligence."
    ]
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()
    for output in outputs:
        generated_text = output.outputs[0].text
        if print_output:
            print(f"Generated text: {generated_text!r}")
        # print(len(output.outputs[0].token_ids))
    # count number of generated tokens
    n_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    throughput = n_tokens / (end - start)
    avg_latency = batch_size * (end - start) / n_tokens
    print(
        f"Latency: takes {avg_latency*1000:.2f} ms/token on average for a total of {n_tokens} tokens"
    )
    print(
        f"Throughput: generates {throughput:.2f} tokens/sec for {batch_size} sequences of {n_tokens} tokens"
    )
    return throughput, avg_latency


# benchmark
benchmark()

# Latency: takes 23.50 ms/token on average for a total of 8334 tokens
# Throughput: generates 4254.42 tokens/sec for 100 sequences of 8334 tokens

# Put in a table and print
res = dict()
for batch_size in [1, 32, 64]:
    for n_tokens in [1, 100, 1000]:
        t, l = generate_from_scratch(n_tokens, batch_size)
        res[(n_tokens, batch_size)] = (round(l * 1000, 2), round(t, 2))
print(res)


# Results
# default on GH200
# {(1, 1): (25.48, 39.25), (100, 1): (7.2, 138.85), (1000, 1): (7.02, 142.51), (1, 32): (28.11, 1138.28), (100, 32): (7.96, 4018.86), (1000, 32): (8.43, 3796.11), (1, 64): (33.63, 1903.16), (100, 64): (9.28, 6896.99), (1000, 64): (10.23, 6258.44)}

# transformers with flashattention
# {(1, 1): (42.77, 23.38), (100, 1): (27.36, 36.55), (1000, 1): (24.98, 40.03), (1, 32): (27.43, 1166.48), (100, 32): (29.54, 1083.15), (1000, 32): (31.73, 1008.44), (1, 64): (32.52, 1967.96), (100, 64): (29.62, 2161.02), (1000, 64): (36.62, 1747.69)}
