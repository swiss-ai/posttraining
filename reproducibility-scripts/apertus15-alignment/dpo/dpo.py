from dataclasses import dataclass, field
from typing import Literal, Union, Dict, List, Tuple

import torch
from trl import DPOConfig, DPOTrainer

@dataclass
class NormedDPOConfig(DPOConfig):
    normalize_logps: bool = field(
        default=False,
        metadata={
            "help": "If `True`, all logps are normalized by the respective number of tokens."
        },
    )

class NormedDPOTrainer(DPOTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Store the normalize flag from args for easy access
        self.normalize_logps = getattr(self.args, "normalize_logps", False)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        # 1. Forward pass
        model_output = self.concatenated_forward(model, batch)

        # 2. Get Reference Logps
        # If precomputed logps exist in batch, use them; otherwise compute them.
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        chosen_logps = model_output["chosen_logps"]
        rejected_logps = model_output["rejected_logps"]

        # --- NORMALIZATION LOGIC START ---
        if self.normalize_logps and self.loss_type == "sigmoid":
            # Count completion tokens from attention masks (1 = real token, 0 = padding)
            chosen_lengths = batch["chosen_attention_mask"].sum(dim=1).float()
            rejected_lengths = batch["rejected_attention_mask"].sum(dim=1).float()

            # Prevent division by zero
            chosen_lengths = chosen_lengths.clamp(min=1)
            rejected_lengths = rejected_lengths.clamp(min=1)

            # Normalize Policy Logps
            chosen_logps = chosen_logps / chosen_lengths
            rejected_logps = rejected_logps / rejected_lengths
            
            # Normalize Reference Logps
            ref_chosen_logps = ref_chosen_logps / chosen_lengths
            ref_rejected_logps = ref_rejected_logps / rejected_lengths
        # --- NORMALIZATION LOGIC END ---

        # 3. Calculate Loss
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps
        )
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        # 4. Handle Auxiliary Losses (RPO, NLL, etc.)
        if self.args.rpo_alpha is not None:
            # RPO loss from V3 of the paper
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        # 5. Calculate Metrics
        # KL Divergence (per token if normalized, per sequence if not)
        kl_chosen = chosen_logps - ref_chosen_logps
        kl_rejected = rejected_logps - ref_rejected_logps
        kl_all = torch.cat([kl_chosen, kl_rejected], dim=0)

        prefix = "eval_" if train_eval == "eval" else ""

        # Helper for cleaner logging
        def log_metric(name, value):
            # gather_for_metrics handles distributed training synchronization
            metrics[f"{prefix}{name}"] = self.accelerator.gather_for_metrics(value).mean().item()

        log_metric("rewards/chosen", chosen_rewards)
        log_metric("rewards/rejected", rejected_rewards)
        log_metric("rewards/accuracies", reward_accuracies)
        log_metric("rewards/margins", chosen_rewards - rejected_rewards)
        
        # Log raw (un-normalized) logps from model output for sanity checking
        log_metric("logps/chosen", model_output["chosen_logps"].detach())
        log_metric("logps/rejected", model_output["rejected_logps"].detach())
        
        # Log the actual logps used for loss (normalized or not)
        log_metric("logps/chosen_used", chosen_logps.detach())
        log_metric("logps/rejected_used", rejected_logps.detach())
        
        log_metric("logits/chosen", model_output["mean_chosen_logits"].detach())
        log_metric("logits/rejected", model_output["mean_rejected_logits"].detach())

        if self.args.rpo_alpha is not None:
             log_metric("nll_loss", model_output["nll_loss"].detach())
        
        if self.aux_loss_enabled:
             log_metric("aux_loss", model_output["aux_loss"].detach())

        log_metric("kl/chosen", kl_chosen)
        log_metric("kl/rejected", kl_rejected)
        log_metric("kl/overall", kl_all)

        return losses.mean(), metrics