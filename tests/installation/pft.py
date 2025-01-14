# Examples from https://github.com/huggingface/peft

import time

import torch
from peft import AutoPeftModelForCausalLM, LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def backward(model, context_size=128, batch_size=1):
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
    duration = end - start
    throughput = context_size * batch_size / (end - start)
    print(
        f"Time: feedforward {duration:.2f} s with {context_size} context size and {batch_size} batch size\n"
        f"Throughput: feedforward {throughput:.2f} tokens/sec with {context_size} context size and {batch_size} batch size"
    )
    loss = output.loss
    start2 = time.time()
    loss.backward()
    end2 = time.time()
    duration2 = end2 - start2
    throughput = context_size * batch_size / (end2 - start2)
    print(
        f"Time: backward {duration2:.2f} ms with {context_size} context size and {batch_size} batch size\n"
        f"Throughput: backward {throughput:.2f} tokens/sec with {context_size} context size and {batch_size} batch size"
    )
    throughput_total = context_size * batch_size / (end2 - start)
    print(
        f"Time: total {duration + duration2:.2f} ms with {context_size} context size and {batch_size} batch size\n"
        f"Throughput: total {throughput_total:.2f} tokens/sec with {context_size} context size and {batch_size} batch size"
    )


model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, device_map="auto")
backward(model)
print("No PEFT")
backward(model)


peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)
print("PEFT")
model.print_trainable_parameters()
backward(model)


model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

model.eval()
inputs = tokenizer(
    "Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt"
)

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
