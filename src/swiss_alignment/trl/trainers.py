import logging
from pathlib import Path

import torch
from accelerate.logging import get_logger
from accelerate.state import PartialState
from thunder.core.dtypes import dtype
from trl import (
    SFTTrainer,
)
from transformers import Trainer
from torch.nn import CrossEntropyLoss
import numpy as np


def preprocess_logits_for_plw_metrics(logits, labels):
    # get predictions
    token_preds = logits.argmax(-1)[..., :-1]

    # compute per-token losses
    loss_fct = CrossEntropyLoss(reduction="none")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    token_losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)

    # pass predictions and losses to compute_metrics function (above)
    predictions = (token_preds, token_losses)
    return predictions

# https://github.com/davidsvaughn/prompt-loss-weight/blob/main/run_plw.py#L544
class PLWTrainer(SFTTrainer):
    def __init__(self, *args, prompt_loss_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.plw = prompt_loss_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # get outputs without computing loss (by not passing in labels)
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.get("logits")
        labels = inputs.pop("labels")

        # compute per-token weights
        weights = torch.tensor(self.plw, dtype=logits.dtype, device=logits.device) * inputs["prompt_mask"] + inputs["completion_mask"]

        # Shift-by-1 so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        shift_weights = shift_weights.to(shift_logits.device)

        # Compute per-token losses
        loss_fct = CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Compute weighted average of losses
        loss = token_losses @ shift_weights.view(-1) / shift_weights.sum()
        return (loss, outputs) if return_outputs else loss

class CustomSFTTrainer(SFTTrainer):
    def evaluate(
            self,
            eval_dataset=None,
            ignore_keys=None,
            metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Saves eval metrics to files"""
        acc_state = PartialState()
        acc_logger = get_logger(__name__)
        acc_logger.info("\nEvaluating model\n")
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        self.log_metrics(f"eval_{self.state.global_step}", metrics)
        self.save_metrics(f"eval_{self.state.global_step}", metrics, combined=False)
        return metrics