import numpy as np
import torch
from accelerate.logging import get_logger
from accelerate.state import PartialState
from torch.nn import CrossEntropyLoss
from transformers import DataCollatorForLanguageModeling
from trl import SFTTrainer

# ------------------------------------------------------------
# SFT Trainers


# In our current setup, this will train over prompt + completion sequence
# If we want to train on only the completion, then use PLWTrainer with plw=0,
# or modify the dataset tokenization (e.g. sft_tulu_tokenize_and_truncate) to
# assign -100 to prompt in labels
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


class PLWDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, pad_to_multiple_of=None):
        super(PLWDataCollator, self).__init__(
            tokenizer=tokenizer, mlm=mlm, pad_to_multiple_of=pad_to_multiple_of
        )

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # Delegate to parent __call__ for standard fields
        standard_fields = [
            {"input_ids": ex["input_ids"], "attention_mask": ex["attention_mask"]}
            for ex in features
        ]
        batch = super(PLWDataCollator, self).__call__(
            standard_fields, return_tensors=return_tensors
        )

        # Custom fields to pad separately
        max_length = batch["input_ids"].shape[1]  # Match length of padded input_ids

        # Pad and tensorize custom fields with framework-appropriate tensors
        for field_name in ["prompt_mask", "completion_mask"]:
            # First create as numpy array (neutral format)
            padded_field = np.zeros((len(features), max_length), dtype=np.int8)
            for i, ex in enumerate(features):
                length = min(len(ex[field_name]), max_length)
                padded_field[i, :length] = ex[field_name][:length]

            # Convert to appropriate tensor type
            if return_tensors == "pt":
                import torch

                batch[field_name] = torch.tensor(padded_field)
            elif return_tensors == "tf":
                import tensorflow as tf

                batch[field_name] = tf.convert_to_tensor(padded_field)
            elif return_tensors == "np":
                batch[field_name] = padded_field

        return batch


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
            padded_eval = self.data_collator(self.eval_dataset, return_tensors="np")
            self.prompt_mask = padded_eval["prompt_mask"]
            self.completion_mask = padded_eval["completion_mask"]
        else:
            self.prompt_mask = self.completion_mask = None

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batch_samples = []
        num_items_in_batch = None

        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break

        count_num_items_in_batch = (
            len(batch_samples) > 0
            and "prompt_mask" in batch_samples[0]
            and "completion_mask" in batch_samples[0]
            and (
                # num_items_in_batch is passed to model forward
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3757
                self.model_accepts_loss_kwargs
                # num_items_in_batch is passed to compute_loss_func
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3773
                or self.compute_loss_func is not None
                # num_items_in_batch is also verified if (self.model_accepts_loss_kwargs or self.compute_loss_func)
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3790
            )
        )
        if count_num_items_in_batch:
            try:
                num_items_in_batch = sum(
                    [
                        (
                            self.plw * batch["prompt_mask"] + batch["completion_mask"]
                        ).sum()
                        for batch in batch_samples
                    ]
                )
            except (TypeError, AttributeError):
                pass

        if num_items_in_batch is not None:
            if self.args.average_tokens_across_devices:
                num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()

            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(device)

                if self.args.n_gpu > 1 and num_items_in_batch.dim() == 0:
                    # In the DataParallel case, convert the scalar tensor into a 1-dim tensor
                    # Note this is not DistributedDataParallel case
                    num_items_in_batch = num_items_in_batch.unsqueeze(0)

        return batch_samples, num_items_in_batch

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

        # Compute weighted average of losses
        loss_fct = CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)

        if num_items_in_batch is not None:
            loss = (token_losses * shift_weights).sum() / num_items_in_batch
        else:
            loss = (token_losses * shift_weights).sum() / shift_weights.sum()

        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes

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

        # Computing prompt/completion loss
        prompt_loss = (token_losses * shift_prompt_mask).sum() / shift_prompt_mask.sum()
        completion_loss = (token_losses * shift_comp_mask).sum() / shift_comp_mask.sum()

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


# Normalizes the prompt and completion independently before performing plw
class LengthNormalizedPLWTrainer(PLWTrainer):
    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batch_samples = []
        num_prompt_in_batch = None
        num_completion_in_batch = None

        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break

        count_num_items_in_batch = (
            len(batch_samples) > 0
            and "prompt_mask" in batch_samples[0]
            and "completion_mask" in batch_samples[0]
            and (
                # num_items_in_batch is passed to model forward
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3757
                self.model_accepts_loss_kwargs
                # num_items_in_batch is passed to compute_loss_func
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3773
                or self.compute_loss_func is not None
                # num_items_in_batch is also verified if (self.model_accepts_loss_kwargs or self.compute_loss_func)
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3790
            )
        )
        if count_num_items_in_batch:
            try:
                num_prompt_in_batch = sum(
                    [(batch["prompt_mask"]).sum() for batch in batch_samples]
                )
                num_completion_in_batch = sum(
                    [(batch["completion_mask"]).sum() for batch in batch_samples]
                )
            except (TypeError, AttributeError):
                pass

        if num_prompt_in_batch is not None and num_completion_in_batch is not None:
            if self.args.average_tokens_across_devices:
                num_prompt_in_batch = self.accelerator.gather(num_prompt_in_batch).sum()
                num_completion_in_batch = self.accelerator.gather(
                    num_completion_in_batch
                ).sum()

            if torch.is_tensor(num_prompt_in_batch):
                num_prompt_in_batch = num_prompt_in_batch.to(device)
                if self.args.n_gpu > 1 and num_prompt_in_batch.dim() == 0:
                    # In the DataParallel case, convert the scalar tensor into a 1-dim tensor
                    num_prompt_in_batch = num_prompt_in_batch.unsqueeze(0)

            if torch.is_tensor(num_completion_in_batch):
                num_completion_in_batch = num_completion_in_batch.to(device)
                if self.args.n_gpu > 1 and num_completion_in_batch.dim() == 0:
                    # In the DataParallel case, convert the scalar tensor into a 1-dim tensor
                    num_completion_in_batch = num_completion_in_batch.unsqueeze(0)

        return batch_samples, (num_prompt_in_batch, num_completion_in_batch)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # get outputs without computing loss (by not passing in labels)
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        logits = outputs.get("logits").float()
        labels = inputs.pop("labels")

        # Shift-by-1 so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_prompt_mask = inputs["prompt_mask"][..., 1:].contiguous()
        shift_completion_mask = inputs["completion_mask"][..., 1:].contiguous()

        # Compute weighted average of losses
        loss_fct = CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)

        # only occurs for eval
        if num_items_in_batch is not None:
            num_prompt_in_batch, num_completion_in_batch = num_items_in_batch
            prompt_loss = (token_losses * shift_prompt_mask).sum() / num_prompt_in_batch
            completion_loss = (
                token_losses * shift_completion_mask
            ).sum() / num_completion_in_batch
            loss = prompt_loss * self.plw + completion_loss
        else:
            prompt_loss = (
                token_losses * shift_prompt_mask
            ).sum() / shift_prompt_mask.sum()
            completion_loss = (
                token_losses * shift_completion_mask
            ).sum() / shift_completion_mask.sum()
            loss = prompt_loss * self.plw + completion_loss

        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


# https://arxiv.org/pdf/2409.01369
# Adapted from: https://github.com/sankyde/IRL/blob/main/IRL.py
class IRLTrainer(LengthNormalizedPLWTrainer):
    def __init__(
        self,
        *args,
        prompt_loss_weight=1.0,
        lambda_td: float = 0.5,
        gamma: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, prompt_loss_weight=prompt_loss_weight, **kwargs)
        self.lambda_td = torch.tensor(
            lambda_td, dtype=self.model.dtype, device=self.args.device
        )
        self.gamma = torch.tensor(
            gamma, dtype=self.model.dtype, device=self.args.device
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # get outputs without computing loss (by not passing in labels)
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        Q = outputs.get("logits").float()  # Q(s,a)
        actions = inputs.pop("labels")

        # Shift-by-1 so that tokens < n predict n
        shift_Q = Q[..., :-1, :].contiguous()
        shift_actions = actions[..., 1:].contiguous()
        shift_prompt_mask = inputs["prompt_mask"][..., 1:].contiguous()
        shift_completion_mask = inputs["completion_mask"][..., 1:].contiguous()
        shift_pad_mask = shift_prompt_mask + shift_completion_mask

        # Compute values: v(s) = logsumexp(Q(s,:))
        v = torch.logsumexp(shift_Q, dim=-1)
        v_next = torch.zeros_like(v)  # next state
        v_next[:, :-1] = v[:, 1:]  # shift right by one

        chosen_Q = Q.gather(-1, (shift_actions * shift_pad_mask).unsqueeze(-1)).squeeze(
            -1
        )
        log_pi = chosen_Q - v
        nll_loss = -log_pi
        # or
        # loss_fct = CrossEntropyLoss(reduction="none")
        # nll_loss = loss_fct(shift_Q.transpose(1, 2), shift_actions)
        # log_pi = -nll_loss

        # temporal difference regularization term
        td_loss = (v + log_pi - self.gamma * v_next) ** 2

        token_losses = self.lambda_td * td_loss + nll_loss
        if num_items_in_batch is not None:
            num_prompt_in_batch, num_completion_in_batch = num_items_in_batch
            prompt_loss = (token_losses * shift_prompt_mask).sum() / num_prompt_in_batch
            completion_loss = (
                token_losses * shift_completion_mask
            ).sum() / num_completion_in_batch
            loss = prompt_loss * self.plw + completion_loss
        else:
            prompt_loss = (
                token_losses * shift_prompt_mask
            ).sum() / shift_prompt_mask.sum()
            completion_loss = (
                token_losses * shift_completion_mask
            ).sum() / shift_completion_mask.sum()
            loss = prompt_loss * self.plw + completion_loss

        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
