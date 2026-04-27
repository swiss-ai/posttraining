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
            elif self.padding_side == "left":
                input_ids.append([self.pad_token_id] * pad_len + ids)
                attention_mask.append([0] * pad_len + [1] * len(ids))
                assistant_masks.append([0] * pad_len + assistant)
            else:
                raise ValueError(f"Unsupported padding_side={self.padding_side!r}")

        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        if return_assistant_tokens_mask:
            out["assistant_masks"] = torch.tensor(assistant_masks, dtype=torch.long)

        return out

    def pad(
        self,
        encoded_inputs,
        *,
        padding=True,
        return_tensors=None,
        return_attention_mask=True,
        **kwargs,
    ):
        assert padding is True
        assert return_tensors == "pt"

        input_rows = encoded_inputs["input_ids"]

        rows = []
        for row in input_rows:
            if isinstance(row, torch.Tensor):
                rows.append(row.to(dtype=torch.long).tolist())
            else:
                rows.append(list(row))

        max_len = max(len(row) for row in rows)

        padded_input_ids = []
        attention_masks = []

        provided_attention = encoded_inputs.get("attention_mask", None)
        provided_attention_rows = None

        if provided_attention is not None:
            provided_attention_rows = []
            for row in provided_attention:
                if isinstance(row, torch.Tensor):
                    provided_attention_rows.append(row.to(dtype=torch.long).tolist())
                else:
                    provided_attention_rows.append(list(row))

        for i, row in enumerate(rows):
            pad_len = max_len - len(row)

            if provided_attention_rows is None:
                mask = [1] * len(row)
            else:
                mask = provided_attention_rows[i]

            if self.padding_side == "right":
                padded_input_ids.append(row + [self.pad_token_id] * pad_len)
                attention_masks.append(mask + [0] * pad_len)
            elif self.padding_side == "left":
                padded_input_ids.append([self.pad_token_id] * pad_len + row)
                attention_masks.append([0] * pad_len + mask)
            else:
                raise ValueError(f"Unsupported padding_side={self.padding_side!r}")

        output = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        }

        if return_attention_mask:
            output["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)

        return output


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


def prompt_messages(system: str = "S", user: str = "Q"):
    return (
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    )


def assistant_trajectory(text: str):
    return ({"role": "assistant", "content": text},)


def tool_trajectory():
    return (
        {"role": "assistant", "content": "", "tool_calls": [{"name": "calculator"}]},
        {"role": "tool", "name": "calculator", "content": "4"},
        {"role": "assistant", "content": "A"},
    )


def candidate(
    trajectory_id: str,
    trajectory_messages,
    *,
    reward: float = 1.0,
    tools=None,
    prompt=None,
):
    return OfflineTrajectoryCandidate(
        prompt_id="p0",
        trajectory_id=trajectory_id,
        prompt_messages=prompt if prompt is not None else prompt_messages(),
        trajectory_messages=trajectory_messages,
        reward=reward,
        ref_rewards=(0.1, 0.2, 0.3),
        tools=tools,
    )


def test_offline_candidates_to_dataproto_returns_verl_prompt_response_batch() -> None:
    data = offline_candidates_to_dataproto(
        candidates=[
            candidate("p0::offline::0", assistant_trajectory("A"), reward=1.0),
            candidate("p0::offline::1", assistant_trajectory("ABC"), reward=2.0),
        ],
        tokenizer=FakeChatTokenizer(),
    )

    assert len(data) == 2

    assert K.PROMPTS in data.batch
    assert K.RESPONSES in data.batch
    assert K.RESPONSE_MASK in data.batch
    assert K.INPUT_IDS in data.batch
    assert K.ATTENTION_MASK in data.batch
    assert K.POSITION_IDS in data.batch
    assert K.TRAJECTORY_REWARD in data.batch
    assert K.REF_REWARDS in data.batch

    assert data.batch[K.PROMPTS].dtype == torch.long
    assert data.batch[K.RESPONSES].dtype == torch.long
    assert data.batch[K.INPUT_IDS].dtype == torch.long
    assert data.batch[K.ATTENTION_MASK].dtype == torch.long

    assert data.batch[K.PROMPTS].shape[0] == 2
    assert data.batch[K.RESPONSES].shape[0] == 2
    assert data.batch[K.RESPONSE_MASK].shape == data.batch[K.RESPONSES].shape

    expected_seq_len = data.batch[K.PROMPTS].shape[1] + data.batch[K.RESPONSES].shape[1]
    assert data.batch[K.INPUT_IDS].shape[1] == expected_seq_len
    assert data.batch[K.ATTENTION_MASK].shape == data.batch[K.INPUT_IDS].shape
    assert data.batch[K.POSITION_IDS].shape == data.batch[K.INPUT_IDS].shape

    assert torch.equal(
        data.batch[K.INPUT_IDS],
        torch.cat([data.batch[K.PROMPTS], data.batch[K.RESPONSES]], dim=-1),
    )

    assert data.batch[K.TRAJECTORY_REWARD].tolist() == [1.0, 2.0]
    assert data.batch[K.REF_REWARDS].shape == (2, 3)

    assert data.non_tensor_batch[K.PROMPT_ID].tolist() == ["p0", "p0"]
    assert data.non_tensor_batch[K.TRAJECTORY_ID].tolist() == [
        "p0::offline::0",
        "p0::offline::1",
    ]
    assert data.non_tensor_batch[K.SOURCE].tolist() == ["offline", "offline"]

    assert data.meta_info["qrpo_batch_format"] == "verl_prompt_response"
    assert data.meta_info["source"] == "offline"


def test_prompts_are_left_padded_and_responses_are_right_padded() -> None:
    tokenizer = FakeChatTokenizer()

    data = offline_candidates_to_dataproto(
        candidates=[
            candidate(
                "p0::offline::0",
                assistant_trajectory("A"),
                prompt=prompt_messages(system="S", user="Q"),
            ),
            candidate(
                "p0::offline::1",
                assistant_trajectory("ABC"),
                prompt=prompt_messages(system="SS", user="Q"),
            ),
        ],
        tokenizer=tokenizer,
    )

    prompts = data.batch[K.PROMPTS]
    responses = data.batch[K.RESPONSES]

    # First prompt is shorter and should be left-padded.
    assert prompts[0, 0].item() == tokenizer.pad_token_id
    assert prompts[0, -1].item() == tokenizer.generation_prompt

    # First response is shorter and should be right-padded.
    assert responses[0, 0].item() == 1000 + ord("A")
    assert responses[0, -1].item() == tokenizer.pad_token_id


def test_response_mask_excludes_prompt_tokens_by_construction() -> None:
    tokenizer = FakeChatTokenizer()

    data = offline_candidates_to_dataproto(
        candidates=[candidate("p0::offline::0", assistant_trajectory("A"))],
        tokenizer=tokenizer,
    )

    # Prompt tokens live in prompts, response_mask is response-only.
    assert data.batch[K.PROMPTS].shape[1] == 5
    assert data.batch[K.RESPONSE_MASK].shape == data.batch[K.RESPONSES].shape
    assert data.batch[K.RESPONSE_MASK].sum().item() == 1


def test_tool_outputs_are_not_trainable_response_tokens() -> None:
    tokenizer = FakeChatTokenizer()

    data = offline_candidates_to_dataproto(
        candidates=[candidate("p0::offline::tool", tool_trajectory())],
        tokenizer=tokenizer,
    )

    responses = data.batch[K.RESPONSES][0]
    response_mask = data.batch[K.RESPONSE_MASK][0].bool()

    tool_pos = int((responses == tokenizer.tool_header).nonzero(as_tuple=True)[0].item())

    assert response_mask[tool_pos].item() is False
    assert response_mask.any().item() is True

    # The assistant tool-call token and final assistant content should be trainable.
    assert (responses == 1500).any()
    tool_call_pos = int((responses == 1500).nonzero(as_tuple=True)[0].item())
    assert response_mask[tool_call_pos].item() is True


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

    assert data.batch[K.RESPONSE_MASK].sum().item() == data.batch[K.RESPONSES].numel()


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


def test_restores_tokenizer_padding_side() -> None:
    tokenizer = FakeChatTokenizer()
    tokenizer.padding_side = "left"

    offline_candidates_to_dataproto(
        candidates=[
            candidate("p0::offline::0", assistant_trajectory("A")),
            candidate("p0::offline::1", assistant_trajectory("ABC")),
        ],
        tokenizer=tokenizer,
    )

    assert tokenizer.padding_side == "left"


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
