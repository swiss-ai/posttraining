from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained(
    "allenai/open-instruct-opt-6.7b-tulu", device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained("allenai/open-instruct-opt-6.7b-tulu")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
history = ""


def template(message):
    return f"<|user|>\n{message}\n<|assistant|>\n"


def chat(message, reset_history=False):
    global history
    if reset_history:
        history = ""
    if history != "":
        message = f"{history}\n{template(message)}"
    else:
        message = template(message)
    history = generator(message, max_length=300, num_return_sequences=1)[0][
        "generated_text"
    ]
    print(history)
