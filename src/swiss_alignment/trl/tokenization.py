import logging
from dataclasses import dataclass
from typing import Optional

from accelerate.logging import get_logger
from accelerate.state import PartialState
from transformers import (
    AutoTokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
)

from swiss_alignment import utils

utils.config.register_resolvers()
acc_state = PartialState()
acc_logger = get_logger(__name__)
hydra_logger = logging.getLogger(__name__)

# chat templates taken from: https://github.com/allenai/open-instruct/blob/main/open_instruct/dataset_transformation.py#L105
# E.g. for [{"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "The result is 4"}]
# with add_generation_prompt=False
CHAT_TEMPLATES = {
    # 'What is 2 + 2? The result is 4<|end_of_text|>'
    "simple_concat_with_space": (
        "{% for message in messages %}"
        "{{ ' ' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    # 'What is 2 + 2?\nThe result is 4<|end_of_text|>'
    "simple_concat_with_new_line": (
        "{% for message in messages %}"
        "{{ '\n' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    # 'User: What is 2 + 2?\n\nAssistant: The result is 4<|end_of_text|>'
    "simple_chat": (
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': ' + message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    # 'The result is 4'
    "assistant_message_only": (
        "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    # '<|user|>\nWhat is 2 + 2?<|end_of_text|>\n<|assistant|>\nThe result is 4<|end_of_text|>\n'
    "zephyr": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + eos_token + '\n' }}"
        "{% elif message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + eos_token + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    # '<|user|>\nWhat is 2 + 2?\n<|assistant|>\nThe result is 4<|end_of_text|>'
    "tulu": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% if not loop.last %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
        "{% else %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token }}"
        "{% endif %}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    # 'A conversation between User and Assistant. The user asks a question ... <answer> answer here </answer>.\n\nUser: What is 2 + 2?\n\n\nAssistant: The result is 4\n'
    "r1_simple_chat": (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in "
        "the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> "
        "<answer> answer here </answer>."
        "\n\n"
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': ' + message['content'] + '\n' }}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ 'Assistant:' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    #' A conversation between User and Assistant. The user asks a question ... <answer> answer here </answer>.\n\nUser: What is 2 + 2?\n\n\nAssistant: The result is 4\n'
    "r1_simple_chat_postpend_think": (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in "
        "the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> "
        "<answer> answer here </answer>."
        "\n\n"
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': ' + message['content'] + '\n' }}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ 'Assistant: <think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
}


@dataclass
class TokenizerConfig:
    model_name_or_path: str
    padding_side: str = "right"
    trust_remote_code: bool = True
    add_bos: bool = False
    chat_template_name: Optional[str] = None
    model_pad_token_id: Optional[int] = None
    model_eos_token_id: Optional[int] = None


# Adapted from: https://github.com/allenai/open-instruct/blob/main/open_instruct/dataset_transformation.py#L378
def get_tokenizer(tc: TokenizerConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        tc.model_name_or_path,
        padding_side=tc.padding_side,
        trust_remote_code=tc.trust_remote_code,
    )

    # Update special tokens.
    if tc.model_pad_token_id is not None:
        tokenizer.pad_token_id = tc.model_pad_token_id
        acc_logger.info(f"Overriding tokenizer pad token id to {tc.model_pad_token_id}")
    if tc.model_eos_token_id is not None:
        tokenizer.eos_token_id = tc.model_eos_token_id
        acc_logger.info(f"Overriding tokenizer eos token id to {tc.model_eos_token_id}")

    # Perform checks
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer must have a pad token.")
    if tokenizer.pad_token == tokenizer.eos_token:
        raise ValueError(
            "Tokenizer pad token is the same as the eos token. The eos will be masked as if it was a pad."
        )

    # set the tokenizer chat template to the training format
    # this will be used for encoding the training examples
    # and saved together with the tokenizer to be used later.
    if tc.chat_template_name in CHAT_TEMPLATES:
        tokenizer.chat_template = CHAT_TEMPLATES[tc.chat_template_name]
    else:
        try:
            tokenizer.chat_template = AutoTokenizer.from_pretrained(
                tc.model_name_or_path
            ).chat_template
        except Exception:
            raise ValueError(
                f"Could not find chat template for {tc.model_name_or_path}."
            )

    if tc.add_bos:
        if tokenizer.chat_template.startswith("{{ bos_token }}") or (
            tokenizer.bos_token is not None
            and tokenizer.chat_template.startswith(tokenizer.bos_token)
        ):
            raise ValueError(
                "You specified add_bos=True, but the chat template already has a bos_token at the beginning."
            )
        # also add bos in the chat template if not already there
        tokenizer.chat_template = "{{ bos_token }}" + tokenizer.chat_template

    return tokenizer
