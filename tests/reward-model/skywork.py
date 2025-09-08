import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
device = "cuda:0"
model_name = "/users/smoalla/projects/posttraining/dev/artifacts/shared/reward-models/skywork-qwen3-8b"
rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"
responses = [
    """1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.
2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.
3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples.""",
    """1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.
2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.
3. Jane splits the 9 apples equally among her 2 siblings (2 people in total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples.""",
    """Hello my name is foo""",
    """Fuck you!""",
    """This should have the highest score.""",
]


convs = [
    [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    for response in responses
]

# Format and tokenize the conversations
convs_formatted = [
    tokenizer.apply_chat_template(conv, tokenize=False) for conv in convs
]
# These two lines remove the potential duplicate bos token
for i, conv in enumerate(convs_formatted):
    if tokenizer.bos_token is not None and conv.startswith(tokenizer.bos_token):
        convs_formatted[i] = conv[len(tokenizer.bos_token) :]

convs_tokenized = [
    tokenizer(conv, return_tensors="pt").to(device) for conv in convs_formatted
]

# Get the reward scores
with torch.no_grad():
    scores = [
        rm(**conv_tokenized).logits[0][0].item() for conv_tokenized in convs_tokenized
    ]

print(scores)

# Expected output:
# Score for response 1: 23.0
# Score for response 2: 3.59375
