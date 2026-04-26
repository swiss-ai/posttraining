import pytest
import torch

from batch import keys as K
from batch.offline_tokenization import offline_candidates_to_dataproto
from data.schemas import OfflineTrajectoryCandidate


class FakeChatTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "right"

    system_header = 101
    user_header = 103
    assistant_header = 104
    tool_header = 105
    generation_prompt = 104

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        return_assistant_tokens_mask=False,
        tools=None,
    ):
        assert tokenize is True
        assert return_tensors == "pt"
        assert return_dict is True

        conversations = conversation
        if conversations and isinstance(conversations[0], dict):
            conversations = [conversations]

        rows = []
        assistant_rows = []

        for messages in conversations:
            ids = []
            assistant = []

            for message in messages:
                role = message["role"]
                if role == "system":
                    header, mask_value = self.system_header, 0
                elif role == "user":
                    header, mask_value = self.user_header, 0
                elif role == "assistant":
                    header, mask_value = self.assistant_header, 1
                elif role == "tool":
                    header, mask_value = self.tool_header, 0
                else:
                    header, mask_value = 199, 0

                ids.append(header)
                assistant.append(0)

                for ch in message.get("content", ""):
                    ids.append(1000 + ord(ch))
                    assistant.append(mask_value)

                for _ in message.get("tool_calls", []):
                    ids.append(1500)
                    assistant.append(mask_value)

            if add_generation_prompt:
                ids.append(self.generation_prompt)
                assistant.append(0)

            rows.append(ids)
            assistant_rows.append(assistant)

        max_len = max(len(row) for row in rows)

        input_ids = []
        attention_mask = []
        assistant_masks = []

        for ids, assistant in zip(rows, assistant_rows, strict=True):
            pad_len = max_len - len(ids)

            if self.padding_side == "right":
                input_ids.append(ids + [self.pad_token_id] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)
                assistant_masks.append(assistant + [0] * pad_len)
            else:
                input_ids.append([self.pad_token_id] * pad_len + ids)
                attention_mask.append([0] * pad_len + [1] * len(ids))
                assistant_masks.append([0] * pad_len + assistant)

        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        if return_assistant_tokens_mask:
            out["assistant_masks"] = torch.tensor(assistant_masks, dtype=torch.long)

        return out


class NoAssistantMaskTokenizer(FakeChatTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        kwargs.pop("return_assistant_tokens_mask", None)
        return super().apply_chat_template(*args, **kwargs)


class BadPrefixTokenizer(FakeChatTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        out = super().apply_chat_template(*args, **kwargs)
        if kwargs.get("add_generation_prompt", False):
            out["input_ids"][:, -1] = 9999
        return out


def prompt_messages():
    return (
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Q"},
    )


def assistant_trajectory(text: str):
    return ({"role": "assistant", "content": text},)


def tool_trajectory():
    return (
        {"role": "assistant", "content": "", "tool_calls": [{"name": "calculator"}]},
        {"role": "tool", "name": "calculator", "content": "4"},
        {"role": "assistant", "content": "A"},
    )


def candidate(trajectory_id: str, trajectory_messages, *, reward: float = 1.0, tools=None):
    return OfflineTrajectoryCandidate(
        prompt_id="p0",
        trajectory_id=trajectory_id,
        prompt_messages=prompt_messages(),
        trajectory_messages=trajectory_messages,
        reward=reward,
        ref_rewards=(0.1, 0.2, 0.3),
        tools=tools,
    )


def test_offline_candidates_to_dataproto_returns_verl_dataproto() -> None:
    data = offline_candidates_to_dataproto(
        candidates=[
            candidate("p0::offline::0", assistant_trajectory("A"), reward=1.0),
            candidate("p0::offline::1", assistant_trajectory("ABC"), reward=2.0),
        ],
        tokenizer=FakeChatTokenizer(),
    )

    assert len(data) == 2

    assert K.INPUT_IDS in data.batch
    assert K.ATTENTION_MASK in data.batch
    assert K.POSITION_IDS in data.batch
    assert K.LOSS_MASK in data.batch
    assert K.TRAJECTORY_REWARD in data.batch
    assert K.REF_REWARDS in data.batch

    assert data.batch[K.INPUT_IDS].dtype == torch.long
    assert data.batch[K.ATTENTION_MASK].dtype == torch.long
    assert data.batch[K.LOSS_MASK].dtype == torch.bool

    assert data.batch[K.INPUT_IDS].shape == data.batch[K.LOSS_MASK].shape
    assert data.batch[K.TRAJECTORY_REWARD].tolist() == [1.0, 2.0]
    assert data.batch[K.REF_REWARDS].shape == (2, 3)

    assert data.non_tensor_batch[K.PROMPT_ID].tolist() == ["p0", "p0"]
    assert data.non_tensor_batch[K.TRAJECTORY_ID].tolist() == [
        "p0::offline::0",
        "p0::offline::1",
    ]
    assert data.non_tensor_batch[K.SOURCE].tolist() == ["offline", "offline"]


def test_loss_mask_excludes_prompt_tokens() -> None:
    data = offline_candidates_to_dataproto(
        candidates=[candidate("p0::offline::0", assistant_trajectory("A"))],
        tokenizer=FakeChatTokenizer(),
    )

    prompt_len = 5

    assert not data.batch[K.LOSS_MASK][0, :prompt_len].any()
    assert data.batch[K.LOSS_MASK][0, prompt_len:].any()


def test_tool_outputs_are_not_trainable() -> None:
    tokenizer = FakeChatTokenizer()

    data = offline_candidates_to_dataproto(
        candidates=[candidate("p0::offline::tool", tool_trajectory())],
        tokenizer=tokenizer,
    )

    input_ids = data.batch[K.INPUT_IDS][0]
    loss_mask = data.batch[K.LOSS_MASK][0]
    attention_mask = data.batch[K.ATTENTION_MASK][0].bool()

    tool_pos = int((input_ids == tokenizer.tool_header).nonzero(as_tuple=True)[0].item())

    assert attention_mask[tool_pos].item() is True
    assert loss_mask[tool_pos].item() is False
    assert loss_mask.any().item() is True


def test_requires_assistant_mask_by_default() -> None:
    with pytest.raises(KeyError, match="assistant_masks"):
        offline_candidates_to_dataproto(
            candidates=[candidate("p0::offline::0", assistant_trajectory("A"))],
            tokenizer=NoAssistantMaskTokenizer(),
        )


def test_can_disable_assistant_mask_requirement() -> None:
    data = offline_candidates_to_dataproto(
        candidates=[candidate("p0::offline::0", assistant_trajectory("A"))],
        tokenizer=NoAssistantMaskTokenizer(),
        config={"require_assistant_mask": False},
    )

    prompt_len = 5

    assert not data.batch[K.LOSS_MASK][0, :prompt_len].any()
    assert data.batch[K.LOSS_MASK][0, prompt_len:].all()


def test_verifies_prompt_prefix() -> None:
    with pytest.raises(ValueError, match="not a prefix"):
        offline_candidates_to_dataproto(
            candidates=[candidate("p0::offline::0", assistant_trajectory("A"))],
            tokenizer=BadPrefixTokenizer(),
        )


def test_can_disable_prompt_prefix_verification() -> None:
    data = offline_candidates_to_dataproto(
        candidates=[candidate("p0::offline::0", assistant_trajectory("A"))],
        tokenizer=BadPrefixTokenizer(),
        config={"verify_prompt_prefix": False},
    )

    assert len(data) == 1


def test_temporarily_forces_right_padding_and_restores_padding_side() -> None:
    tokenizer = FakeChatTokenizer()
    tokenizer.padding_side = "left"

    data = offline_candidates_to_dataproto(
        candidates=[
            candidate("p0::offline::0", assistant_trajectory("A")),
            candidate("p0::offline::1", assistant_trajectory("ABC")),
        ],
        tokenizer=tokenizer,
    )

    assert tokenizer.padding_side == "left"
    assert data.batch[K.INPUT_IDS][0, 0].item() == tokenizer.system_header


def test_rejects_empty_candidate_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        offline_candidates_to_dataproto(candidates=[], tokenizer=FakeChatTokenizer())


def test_rejects_different_tools_when_required() -> None:
    tools_a = ({"type": "function", "function": {"name": "tool_a", "parameters": {}}},)
    tools_b = ({"type": "function", "function": {"name": "tool_b", "parameters": {}}},)

    with pytest.raises(ValueError, match="same tools"):
        offline_candidates_to_dataproto(
            candidates=[
                candidate("p0::offline::0", assistant_trajectory("A"), tools=tools_a),
                candidate("p0::offline::1", assistant_trajectory("B"), tools=tools_b),
            ],
            tokenizer=FakeChatTokenizer(),
        )
