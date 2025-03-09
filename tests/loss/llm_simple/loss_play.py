from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.loss.loss_utils import ForCausalLMLoss, fixed_cross_entropy
import torch
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")


prompt = "Hello World. Here I am on my way to the store."
inputs = tokenizer(prompt, return_tensors="pt")


input_ids = inputs['input_ids']


tokenizer.pad_token = tokenizer.eos_token



outputs = model(input_ids=input_ids, attention_mask=inputs['attention_mask'], labels=input_ids)



logits = outputs.logits.float()
labels = input_ids.to(logits.device)

labels = torch.nn.functional.pad(labels, (0, 1), value=-100)

shift_labels = labels[..., 1:].contiguous()

vocab_size = logits.shape[-1]
logits = logits.view(-1, vocab_size)

shift_labels = shift_labels.view(-1)
shift_labels = shift_labels.to(logits.device)
# loss = fixed_cross_entropy(logits, shift_labels, input_ids.shape[-1] - 1, -100)

token_losses = torch.nn.functional.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="none")

loss = token_losses.sum() / (input_ids.shape[-1] - 1)

# reduction = "sum" if num_items_in_batch is not None else "mean"
# loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
# if reduction == "sum":
#     loss = loss / num_items_in_batch
# return loss

print("Per token Loss", token_losses)
print("Per token Loss sum", token_losses.sum())
print("Computed Loss", loss)
print("Output loss", outputs.loss)

func_loss = ForCausalLMLoss(outputs.logits, input_ids, vocab_size=outputs.logits.shape[-1])
print("Function loss", func_loss)
print("Ratio", loss / func_loss)
print("Size", input_ids.shape)

