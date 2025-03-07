import logging

import numpy as np
import torch
import torch.nn as nn
from accelerate.logging import get_logger
from accelerate.state import PartialState
from torch.nn import CrossEntropyLoss
from trl import SFTTrainer

from swiss_alignment import utils

utils.config.register_resolvers()
acc_state = PartialState()
acc_logger = get_logger(__name__)
hydra_logger = logging.getLogger(__name__)


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


# Adapted from: https://github.com/davidsvaughn/prompt-loss-weight/blob/main/run_plw.py
class PLWTrainer(CustomSFTTrainer):
    def __init__(self, *args, prompt_loss_weight=1.0, **kwargs):
        kwargs.update(
            {
                "preprocess_logits_for_metrics": self.preprocess_logits_for_metrics,
                "compute_metrics": self.compute_metrics,
            }
        )
        super().__init__(*args, **kwargs)
        self.plw = torch.tensor(
            prompt_loss_weight, dtype=self.model.dtype, device=self.args.device
        )

        # Store eval masks as tensors if eval_dataset is provided
        if self.eval_dataset is not None:
            prompt_masks = self.eval_dataset["prompt_mask"]
            completion_masks = self.eval_dataset["completion_mask"]

            max_length = min(
                self.args.max_seq_length,
                max([len(tokens) for tokens in self.eval_dataset["input_ids"]]),
            )
            dataset_size = len(self.eval_dataset)

            self.prompt_mask = np.zeros((dataset_size, max_length), dtype=np.int8)
            self.completion_mask = np.zeros((dataset_size, max_length), dtype=np.int8)

            for i in range(dataset_size):
                p_len = min(len(prompt_masks[i]), max_length)
                c_len = min(len(completion_masks[i]), max_length)

                self.prompt_mask[i, :p_len] = prompt_masks[i][:p_len]
                self.completion_mask[i, :c_len] = completion_masks[i][:c_len]
        else:
            self.prompt_mask = self.completion_mask = None

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # get outputs without computing loss (by not passing in labels)
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        logits = outputs.get("logits").float()
        labels = inputs.pop("labels")

        # compute per-token weights
        weights = self.plw * inputs["prompt_mask"] + inputs["completion_mask"]

        # Shift-by-1 so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()

        # Enable model parallelism
        shift_labels = shift_labels.to(logits.device)
        shift_weights = shift_weights.to(logits.device)

        # Compute weighted average of losses
        loss_fct = CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)

        if num_items_in_batch is not None:
            # TODO: clarify
            # loss = (token_losses * shift_weights).sum() / shift_weights.sum()
            loss = (token_losses * shift_weights).sum() / num_items_in_batch
            loss *= self.accelerator.num_processes
        else:
            # TODO: verify this, because here the "mean" is per mini-batch and does not consider num_items_in_batch
            loss = (token_losses * shift_weights).mean()

        return (loss, outputs) if return_outputs else loss

    # uses PyTorch tensors (on GPU)
    def preprocess_logits_for_metrics(self, logits, labels):
        token_preds = logits.argmax(-1)[..., :-1]
        loss_fct = CrossEntropyLoss(reduction="none")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        token_losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)
        return token_preds, token_losses

    # uses numpy arrays (on CPU)
    def compute_metrics(self, data):
        token_preds, token_losses = data.predictions
        shift_labels = data.label_ids[..., 1:]

        # Align masks with processed dataset (due to dataloader_drop_last)
        dataset_size = token_losses.shape[0]
        shift_prompt_mask = self.prompt_mask[:dataset_size, 1:]
        shift_comp_mask = self.completion_mask[:dataset_size, 1:]

        # TODO: clarify this
        # Compute total/prompt/completion average loss
        # do we want this? it depends on the compute_loss function when num_items_in_batch is set to None
        # total_loss = (token_losses * (self.plw.to(dtype=torch.float32).cpu().numpy() * shift_prompt_mask + shift_comp_mask)).mean()
        prompt_loss = (token_losses * shift_prompt_mask).mean()
        completion_loss = (token_losses * shift_comp_mask).mean()
        # or
        # prompt_loss = (token_losses * shift_prompt_mask).sum() / shift_prompt_mask.sum()
        # completion_loss = (token_losses * shift_comp_mask).sum() / shift_comp_mask.sum()

        # Compute total/prompt/completion accuracies
        def compute_accuracy(preds, labels, mask):
            correct_tokens = ((preds == labels) & mask).sum()
            total_tokens = mask.sum()
            return correct_tokens / total_tokens if total_tokens > 0 else 0.0

        pad_mask = shift_labels != -100
        total_accuracy = compute_accuracy(token_preds, shift_labels, pad_mask)
        prompt_accuracy = compute_accuracy(
            token_preds, shift_labels, pad_mask & shift_prompt_mask.astype(bool)
        )
        completion_accuracy = compute_accuracy(
            token_preds, shift_labels, pad_mask & shift_comp_mask.astype(bool)
        )

        return {
            "prompt_loss": prompt_loss,
            "completion_loss": completion_loss,
            "mean_token_accuracy": total_accuracy,
            "prompt_token_accuracy": prompt_accuracy,
            "completion_token_accuracy": completion_accuracy,
        }
