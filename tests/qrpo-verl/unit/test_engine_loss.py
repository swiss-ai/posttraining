import torch
from tensordict import TensorDict

from batch import keys as K
from verl_adapters import engine_loss as engine_loss_module
from verl_adapters.engine_loss import qrpo_engine_loss


def make_data() -> TensorDict:
    batch_size = 2
    response_len = 3

    return TensorDict(
        {
            K.RESPONSES: torch.tensor(
                [
                    [101, 102, 0],
                    [201, 202, 203],
                ],
                dtype=torch.long,
            ),
            K.RESPONSE_MASK: torch.tensor(
                [
                    [1, 1, 0],
                    [1, 0, 1],
                ],
                dtype=torch.bool,
            ),
            K.REF_LOG_PROBS: torch.tensor(
                [
                    [-1.0, -2.0, 0.0],
                    [-1.0, 0.0, -3.0],
                ],
                dtype=torch.float32,
            ),
            K.TRANSFORMED_REWARD: torch.tensor([0.5, 1.0], dtype=torch.float32),
            K.EFFECTIVE_BETA: torch.tensor([2.0, 3.0], dtype=torch.float32),
            K.BETA_LOG_PARTITION: torch.tensor([0.1, 0.2], dtype=torch.float32),
        },
        batch_size=[batch_size],
    )


def test_qrpo_engine_loss_uses_verl_padding_adapter_and_computes_stable_loss(monkeypatch) -> None:
    data = make_data()

    current_log_probs = torch.tensor(
        [
            [-0.5, -1.5, 0.0],
            [-2.0, 0.0, -2.5],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    model_output = TensorDict(
        {
            "log_probs": current_log_probs,
        },
        batch_size=[2],
    )

    called = {"no_padding_2_padding": False}

    def fake_no_padding_2_padding(log_probs, data_arg):
        called["no_padding_2_padding"] = True
        assert log_probs is current_log_probs
        assert data_arg is data
        return log_probs

    monkeypatch.setattr(
        engine_loss_module,
        "no_padding_2_padding",
        fake_no_padding_2_padding,
    )

    loss, metrics = qrpo_engine_loss(
        config={},
        model_output=model_output,
        data=data,
        dp_group=None,
    )

    sequence_log_ratio = torch.tensor(
        [
            (-0.5 + 1.0) + (-1.5 + 2.0),
            (-2.0 + 1.0) + (-2.5 + 3.0),
        ],
        dtype=torch.float32,
    )
    expected_residual = (
        data[K.TRANSFORMED_REWARD]
        - data[K.BETA_LOG_PARTITION]
        - data[K.EFFECTIVE_BETA] * sequence_log_ratio
    )
    expected_loss = expected_residual.square().mean()

    assert called["no_padding_2_padding"] is True
    assert torch.allclose(loss, expected_loss)

    assert "actor/qrpo_loss" in metrics
    assert "actor/sequence_log_ratio_mean" in metrics
    assert "actor/qrpo_residual_mean" in metrics
    assert "actor/qrpo_per_sample_loss_mean" in metrics

    loss.backward()
    assert current_log_probs.grad is not None
    assert torch.isfinite(current_log_probs.grad).all()


def test_qrpo_engine_loss_masks_tool_and_padding_response_tokens(monkeypatch) -> None:
    data = make_data()

    current_log_probs = torch.tensor(
        [
            [-0.5, -1.5, -100.0],
            [-2.0, -100.0, -2.5],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    model_output = TensorDict(
        {
            "log_probs": current_log_probs,
        },
        batch_size=[2],
    )

    monkeypatch.setattr(
        engine_loss_module,
        "no_padding_2_padding",
        lambda log_probs, data_arg: log_probs,
    )

    loss, _ = qrpo_engine_loss(
        config={},
        model_output=model_output,
        data=data,
        dp_group=None,
    )

    loss.backward()

    # Masked positions should have zero gradient.
    assert current_log_probs.grad[0, 2].item() == 0.0
    assert current_log_probs.grad[1, 1].item() == 0.0

    # Unmasked positions should participate in the loss.
    assert current_log_probs.grad[0, 0].item() != 0.0
    assert current_log_probs.grad[0, 1].item() != 0.0
    assert current_log_probs.grad[1, 0].item() != 0.0
    assert current_log_probs.grad[1, 2].item() != 0.0


def test_qrpo_engine_loss_does_not_require_old_log_probs(monkeypatch) -> None:
    data = make_data()
    assert "old_log_probs" not in data.keys()

    current_log_probs = torch.zeros(2, 3, dtype=torch.float32, requires_grad=True)

    model_output = TensorDict(
        {
            "log_probs": current_log_probs,
        },
        batch_size=[2],
    )

    monkeypatch.setattr(
        engine_loss_module,
        "no_padding_2_padding",
        lambda log_probs, data_arg: log_probs,
    )

    loss, _ = qrpo_engine_loss(
        config={},
        model_output=model_output,
        data=data,
        dp_group=None,
    )

    assert loss.ndim == 0
    