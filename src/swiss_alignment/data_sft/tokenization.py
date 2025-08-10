import logging
from dataclasses import dataclass
from typing import Optional

from accelerate.logging import get_logger
from transformers import AutoTokenizer

acc_logger = get_logger(__name__)
hydra_logger = logging.getLogger(__name__)

# chat templates taken from: https://github.com/allenai/open-instruct/blob/main/open_instruct/dataset_transformation.py#L105
# E.g. for [{"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "The result is 4"}]
# with add_generation_prompt=False
CHAT_TEMPLATES = {
    # 'What is 2 + 2? The result is 4</s>'
    "simple_concat_with_space": (
        "{% for message in messages %}"
        "{{ ' ' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    # 'What is 2 + 2?\nThe result is 4</s>'
    "simple_concat_with_new_line": (
        "{% for message in messages %}"
        "{{ '\n' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    # 'User: What is 2 + 2?\n\nAssistant: The result is 4</s>'
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
    # '<|user|>\nWhat is 2 + 2?</s>\n<|assistant|>\nThe result is 4</s>\n'
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
    # '<|user|>\nWhat is 2 + 2?\n<|assistant|>\nThe result is 4</s>'
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
    # '[INST]\nWhat is 2 + 2?\n[/INST]\nThe result is 4</s>'
    "tulu_special_token": (
        "{%- if messages[0]['role'] == 'system' %}"
        "{%- set system_message = messages[0]['content'] %}"
        "{%- set loop_messages = messages[1:] %}"
        "{%- else %}"
        "{%- set loop_messages = messages %}"
        "{%- endif %}"
        "{% for message in messages %}"
        "{%- if message['role'] == 'user' %} "
        "{%- if system_message is defined and loop.first %} "
        "{{ '[INST]\n' + system_message + '\n\n' + message['content'] + '\n' }}"
        "{%- else %} "
        "{{ '[INST]\n' + message['content'] + '\n' }}"
        "{%- endif %} "
        "{% elif message['role'] == 'assistant' %}"
        "{% if not loop.last %}"
        "{{ '[/INST]\n'  + message['content'] + eos_token + '\n' }}"
        "{% else %}"
        "{{ '[/INST]\n'  + message['content'] + eos_token }}"
        "{% endif %}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '[/INST]\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    # '[INST] What is 2 + 2?[/INST] The result is 4</s>'
    "mistral": (
        "{%- if messages[0]['role'] == 'system' %}"
        "{%- set system_message = messages[0]['content'] %}"
        "{%- set loop_messages = messages[1:] %}"
        "{%- else %}"
        "{%- set loop_messages = messages %}"
        "{%- endif %}"
        "{%- for message in loop_messages %}"
        "{%- if message['role'] == 'user' %} "
        "{%- if system_message is defined and loop.first %} "
        "{{ '[INST] ' + system_message + '\n\n' + message['content'] + '[/INST]' }} "
        "{%- else %} "
        "{{ '[INST] ' + message['content'] + '[/INST]' }} "
        "{%- endif %} "
        "{%- elif message['role'] == 'assistant' %} "
        "{{ ' ' + message['content']|trim + eos_token }} "
        "{%- else %} "
        "{{ raise_exception('Only user and assistant roles are supported!') }} "
        "{%- endif %} "
        "{%- endfor %}"
    ),
    # '<|user|>What is 2 + 2?<|assistant|>The result is 4</s>'
    "apertus_chatml_no_special_tokens": (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'user' %}"
        "{{ '<|user|>' + message['content'] }}"
        "{%- elif message['role'] == 'assistant' %}"
        "{{ '<|assistant|>' + message['content'] + eos_token }}"
        "{%- endif %}"
        "{%- if loop.last and add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
        "{%- endfor %}"
    ),
    # '[INST]What is 2 + 2?[/INST]The result is 4</s>'
    "apertus_chatml_special_tokens": (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'user' %}"
        "{{ '[INST]' + message['content'] }}"
        "{%- elif message['role'] == 'assistant' %}"
        "{{ '[/INST]' + message['content'] + eos_token }}"
        "{%- endif %}"
        "{%- if loop.last and add_generation_prompt %}{{ '[/INST]' }}{% endif %}"
        "{%- endfor %}"
    ),
    # '[QUERY]What is 2 + 2?[/QUERY][RESPONSE]The result is 4[/RESPONSE]</s>'
    "apertus_mistral_no_special_tokens": (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'user' %}"
        "{{ '[QUERY]' + message['content'] + '[/QUERY]' }}"
        "{%- elif message['role'] == 'assistant' %}"
        "{{ '[RESPONSE]' + message['content'] + '[/RESPONSE]' + eos_token }}"
        "{%- endif %}"
        "{%- if loop.last and add_generation_prompt %}{{ '[RESPONSE]' }}{% endif %}"
        "{%- endfor %}"
    ),
    # '[INST]What is 2 + 2?[/INST]<SPECIAL_61>The result is 4<SPECIAL_62></s>'
    "apertus_mistral_special_tokens": (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'user' %}"
        "{{ '[INST]' + message['content'] + '[/INST]' }}"
        "{%- elif message['role'] == 'assistant' %}"
        "{{ '<SPECIAL_61>' + message['content'] + '<SPECIAL_62>' + eos_token }}"
        "{%- endif %}"
        "{%- if loop.last and add_generation_prompt %}{{ '<SPECIAL_61>' }}{% endif %}"
        "{%- endfor %}"
    ),
    # '[INST]What is 2 + 2?[/INST]<SPECIAL_61>The result is 4</s>'
    "apertus_mistral_special_tokens_eos": (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'user' %}"
        "{{ '[INST]' + message['content'] + '[/INST]' }}"
        "{%- elif message['role'] == 'assistant' %}"
        "{{ '<SPECIAL_61>' + message['content'] + eos_token }}"
        "{%- endif %}"
        "{%- if loop.last and add_generation_prompt %}{{ '<SPECIAL_61>' }}{% endif %}"
        "{%- endfor %}"
    ),
    # '<user>What is 2 + 2?</user><assistant>The result is 4</assistant></s>'
    "apertus_xml_no_special_tokens": (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'user' %}"
        "{{ '<user>' + message['content'] + '</user>' }}"
        "{%- elif message['role'] == 'assistant' %}"
        "{{ '<assistant>' + message['content'] + '</assistant>' + eos_token }}"
        "{%- endif %}"
        "{%- if loop.last and add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
        "{%- endfor %}"
    ),
    # '<s>[INST]What is 2 + 2?[/INST]\n<SPECIAL_61>The result is 4</s>\n'
    "apertus_mistral_only_between_roles": (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'user' %}"
        "{{ '[INST]' + message['content'] + '[/INST]\n' }}"
        "{%- elif message['role'] == 'assistant' %}"
        "{{ '<SPECIAL_61>' + message['content'] + eos_token + '\n' }}"
        "{%- endif %}"
        "{%- if loop.last and add_generation_prompt %}{{ '<SPECIAL_61>' }}{% endif %}"
        "{%- endfor %}"
    ),
    # '<s>[INST]\nWhat is 2 + 2?\n[/INST]\n<SPECIAL_61>\nThe result is 4\n</s>\n'
    "apertus_mistral_everywhere": (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'user' %}"
        "{{ '[INST]\n' + message['content'] + '\n[/INST]\n' }}"
        "{%- elif message['role'] == 'assistant' %}"
        "{{ '<SPECIAL_61>\n' + message['content'] + '\n' + eos_token + '\n' }}"
        "{%- endif %}"
        "{%- if loop.last and add_generation_prompt %}{{ '<SPECIAL_61>\n' }}{% endif %}"
        "{%- endfor %}"
    ),
    # '<s>[INST]\nWhat is 2 + 2?\n[/INST]\n<SPECIAL_61>\nThe result is 4</s>\n'
    "apertus_mistral_everywhere_except_eos": (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'user' %}"
        "{{ '[INST]\n' + message['content'] + '\n[/INST]\n' }}"
        "{%- elif message['role'] == 'assistant' %}"
        "{{ '<SPECIAL_61>\n' + message['content'] + eos_token + '\n' }}"
        "{%- endif %}"
        "{%- if loop.last and add_generation_prompt %}{{ '<SPECIAL_61>\n' }}{% endif %}"
        "{%- endfor %}"
    ),
    # '<s>[INST]\nWhat is 2 + 2?[/INST]\n<SPECIAL_61>\nThe result is 4</s>\n'
    "apertus_mistral_beginning_of_message": (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'user' %}"
        "{{ '[INST]\n' + message['content'] + '[/INST]\n' }}"
        "{%- elif message['role'] == 'assistant' %}"
        "{{ '<SPECIAL_61>\n' + message['content'] + eos_token + '\n' }}"
        "{%- endif %}"
        "{%- if loop.last and add_generation_prompt %}{{ '<SPECIAL_61>\n' }}{% endif %}"
        "{%- endfor %}"
    ),
    # '<s>[INST]What is 2 + 2?[/INST]\n<SPECIAL_61>\nThe result is 4\n</s>\n'
    "apertus_mistral_asymmetric_assitant": (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'user' %}"
        "{{ '[INST]' + message['content'] + '[/INST]\n' }}"
        "{%- elif message['role'] == 'assistant' %}"
        "{{ '<SPECIAL_61>\n' + message['content'] + '\n' + eos_token + '\n' }}"
        "{%- endif %}"
        "{%- if loop.last and add_generation_prompt %}{{ '<SPECIAL_61>\n' }}{% endif %}"
        "{%- endfor %}"
    ),
}


@dataclass
class TokenizerConfig:
    model_name_or_path: str
    padding_side: str = "right"
    trust_remote_code: bool = True
    add_bos_to_chat_template: bool = False
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

    if tc.add_bos_to_chat_template:
        if tokenizer.chat_template.startswith("{{ bos_token }}") or (
            tokenizer.bos_token is not None
            and tokenizer.chat_template.startswith(tokenizer.bos_token)
        ):
            raise ValueError(
                "You specified add_bos_to_chat_template=True, but the chat template already has a bos_token at the beginning."
            )
        # also add bos in the chat template if not already there
        tokenizer.chat_template = "{{ bos_token }}" + tokenizer.chat_template

    return tokenizer
