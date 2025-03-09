from time import sleep
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD
from accelerate import Accelerator
from transformers.loss.loss_utils import ForCausalLMLoss


def compute_per_token_loss(logits, labels):
    logits = logits.float()
    labels = labels.to(logits.device)

    labels = torch.nn.functional.pad(labels, (0, 1), value=-100)

    shift_labels = labels[..., 1:].contiguous()

    vocab_size = logits.shape[-1]
    logits = logits.view(-1, vocab_size)

    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(logits.device)

    token_losses = torch.nn.functional.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="none")

    loss = token_losses.sum() / (input_ids.shape[-1] - 1)

    return token_losses, loss


# Initialize the Accelerator
accelerator = Accelerator()

# Load the model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"  # Use the correct model path if necessary

# Load the model and tokenizer automatically using Auto classes
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

# Define two sentences for training
sentences = ["Hello. Hello. Hello.", "Model. Model. Model.",
             "Hello. Hello. Hello.", "Model. Model. Model."] * (accelerator.num_processes // 2)

# Tokenize the sentences
tokenized_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Extract input IDs and attention masks
input_ids = tokenized_inputs["input_ids"]
attention_mask = tokenized_inputs["attention_mask"]

# Create a TensorDataset and DataLoader
dataset = TensorDataset(input_ids, attention_mask)
train_dataloader = DataLoader(dataset, batch_size=1)

# Define the SGD optimizer with a learning rate of 1.0
optimizer = SGD(model.parameters(), lr=1.0)

# Use the Accelerator to handle multi-GPU and mixed precision (if available)
device = accelerator.device
model.to(device)

# Prepare the dataloader and optimizer with accelerator
model, train_dataloader, optimizer = accelerator.prepare(model, train_dataloader, optimizer)

sleep(2)
print(model.gradient_clipping())

training_losses = []

for batch in train_dataloader:
    input_ids, attention_mask = [item.to(device) for item in batch]

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    LlamaForCausalLM
    ForCausalLMLoss
    loss = outputs.loss

    # Backward pass
    accelerator.backward(loss)

    training_losses.append(loss.item())

    # Optimizer step
    optimizer.step()

    sleep(accelerator.process_index)

    print()

    print("index", accelerator.process_index, "decice", device)
    print("index", accelerator.process_index, "loss", loss)
    print("index", accelerator.process_index, "per token loss", compute_per_token_loss(outputs.logits, input_ids))
    print("index", accelerator.process_index, "input", input_ids)
    print("index", accelerator.process_index, "global grad norm: ", model.get_global_grad_norm())

    accelerator.wait_for_everyone()

# After training

if accelerator.process_index in [0]:
    with open("losses.json", "w") as f:
        import json

        json.dump(training_losses, f)

accelerator.end_training()
