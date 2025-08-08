# Code borrowed from https://github.com/huggingface/trl/
# License for original code from TRL

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import PartialState
from accelerate.logging import get_logger
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from packaging import version
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
    is_wandb_available,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_peft_available
from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.utils import (
    disable_dropout_in_model,
    pad,
    pad_to_length,
    peft_module_casting_to_bf16,
)

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


@dataclass
class PreferenceTrainerConfig(TrainingArguments):
    learning_rate: float = 1e-6
    beta: float = 0.1
    num_ref_rewards: Optional[int] = None
    loss_type: Literal["qrpo", "dpo"] = "qrpo"
    label_pad_token_id: int = -100
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None
    is_encoder_decoder: Optional[bool] = None
    disable_dropout: bool = True
    dataset_num_proc: Optional[int] = None
    model_init_kwargs: Optional[dict[str, Any]] = None
    load_ref_model: Optional[bool] = True
    ref_model_init_kwargs: Optional[dict[str, Any]] = None
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    reference_free: bool = False
    force_use_ref_model: bool = False
    use_num_logits_to_keep: bool = False
    precompute_ref_log_probs: bool = False
    sync_ref_model: bool = False

    def __post_init__(self):
        assert self.precompute_ref_log_probs == False, (
            "precompute_ref_log_probs is not supported in PreferenceTrainer. "
            "Please set precompute_ref_log_probs=False."
        )
        return super().__post_init__()


@dataclass
class PreferenceTrainerCollator(DataCollatorMixin):
    """
    Data collator that handles input_ids with right padding and a scalar column for prompt_length.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.
    """

    pad_token_id: int
    num_ref_rewards: int = 1
    return_tensors: str = "pt"

    def torch_call(
        self, examples: list[Union[list[int], Any, dict[str, Any]]]
    ) -> dict[str, Any]:
        # Convert to tensor

        # Text input.
        prompt_input_ids = [
            torch.tensor(example["prompt_input_ids"]) for example in examples
        ]
        prompt_attention_mask = [
            torch.ones_like(input_ids) for input_ids in prompt_input_ids
        ]
        chosen_input_ids = [
            torch.tensor(example["chosen_input_ids"]) for example in examples
        ]
        chosen_attention_mask = [
            torch.ones_like(input_ids) for input_ids in chosen_input_ids
        ]
        rejected_input_ids = [
            torch.tensor(example["rejected_input_ids"]) for example in examples
        ]
        rejected_attention_mask = [
            torch.ones_like(input_ids) for input_ids in rejected_input_ids
        ]

        # Pixel input.
        pixel_values = None
        pixel_attention_mask = None
        if "pixel_values" in examples[0]:
            pixel_values = [
                torch.tensor(example["pixel_values"]) for example in examples
            ]
        if "pixel_attention_mask" in examples[0]:
            pixel_attention_mask = [
                torch.tensor(example["pixel_attention_mask"]) for example in examples
            ]

        # Pad
        output = dict()
        output["prompt_input_ids"] = pad(
            prompt_input_ids, padding_value=self.pad_token_id, padding_side="left"
        )
        output["prompt_attention_mask"] = pad(
            prompt_attention_mask, padding_value=0, padding_side="left"
        )
        output["chosen_input_ids"] = pad(
            chosen_input_ids, padding_value=self.pad_token_id
        )
        output["chosen_attention_mask"] = pad(
            chosen_attention_mask, padding_value=0, padding_side="right"
        )
        output["rejected_input_ids"] = pad(
            rejected_input_ids, padding_value=self.pad_token_id, padding_side="right"
        )
        output["rejected_attention_mask"] = pad(
            rejected_attention_mask, padding_value=0, padding_side="right"
        )
        if "pixel_values" in examples[0]:
            output["pixel_values"] = pad(pixel_values, padding_value=0.0)
        if "pixel_attention_mask" in examples[0]:
            output["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)
        if "image_sizes" in examples[0]:
            output["image_sizes"] = torch.tensor(
                [example["image_sizes"] for example in examples]
            )

        # Ref log probs if already present
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            ref_chosen_logps = torch.tensor(
                [example["ref_chosen_logps"] for example in examples]
            )
            ref_rejected_logps = torch.tensor(
                [example["ref_rejected_logps"] for example in examples]
            )
            output["ref_chosen_logps"] = ref_chosen_logps
            output["ref_rejected_logps"] = ref_rejected_logps

        # Extract ref_rewards
        if "ref_rewards" in examples[0]:
            ref_rewards = torch.tensor(
                [example["ref_rewards"][: self.num_ref_rewards] for example in examples]
            )
            chosen_rewards = torch.tensor(
                [example["chosen_rewards"] for example in examples]
            )
            rejected_rewards = torch.tensor(
                [example["rejected_rewards"] for example in examples]
            )
            output["ref_rewards"] = ref_rewards
            output["chosen_rewards"] = chosen_rewards
            output["rejected_rewards"] = rejected_rewards

        return output


class PreferenceTrainer(Trainer):
    r"""
    Initialize PreferenceTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        args (`PreferenceTrainerConfig`):
            The PreferenceTrainer config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`PreferenceTrainerCollator`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
    """

    _tag_names = ["trl", "swiss-alignment", "preference-trainer"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[PreferenceTrainerConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[
                PreTrainedTokenizerBase,
                BaseImageProcessor,
                FeatureExtractionMixin,
                ProcessorMixin,
            ]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        peft_config: Optional[dict] = None,
    ):
        if model is None:
            raise ValueError("No model provided. Please provide a model to train.")

        if not isinstance(model, str) and ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must mass a copy of it, or `None` if you use peft."
            )

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError(
                "You passed model_init_kwargs to the PreferenceTrainer/PreferenceTrainerConfig, but your model is already instantiated."
            )
        else:
            model_init_kwargs = args.model_init_kwargs
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the PreferenceTrainerConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                model_init_kwargs["torch_dtype"] = torch_dtype

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_init_kwargs to the PreferenceTrainer/PreferenceTrainerConfig, but your ref_model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            torch_dtype = ref_model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                # Convert to `torch.dtype` if an str is passed
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the PreferenceTrainerConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
                ref_model_init_kwargs["torch_dtype"] = torch_dtype

        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            ref_model = AutoModelForCausalLM.from_pretrained(
                ref_model, **ref_model_init_kwargs
            )

        # Initialize this variable to False. This helps track the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if ref_model is not None and not args.force_use_ref_model:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with PreferenceTrainer there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in PreferenceTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(
                model, "is_loaded_in_4bit", False
            ):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {
                    "use_gradient_checkpointing": args.gradient_checkpointing
                }

                if _support_gc_kwargs:
                    prepare_model_kwargs[
                        "gradient_checkpointing_kwargs"
                    ] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(
                        make_inputs_require_grad
                    )

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.is_vision_model = (
            model.config.model_type in MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.keys()
        )
        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name
        self.reference_free = args.reference_free

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or args.precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if processing_class is None:
            raise ValueError(
                "processing_class must be specified to tokenize a PreferenceTrainer dataset."
            )

        if (
            hasattr(processing_class, "pad_token_id")
            and processing_class.pad_token_id is not None
        ):
            self.pad_token_id = processing_class.pad_token_id
        elif (
            hasattr(processing_class, "tokenizer")
            and processing_class.tokenizer.pad_token_id is not None
        ):
            self.pad_token_id = processing_class.tokenizer.pad_token_id
        else:
            raise ValueError(
                "Can't find `pad_token_id` in the `processing_class`. "
                "Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`) "
                "before instantiating the trainer."
            )

        if data_collator is None:
            data_collator = PreferenceTrainerCollator(
                pad_token_id=self.pad_token_id, num_ref_rewards=args.num_ref_rewards
            )

        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = args.max_length
        self.label_pad_token_id = args.label_pad_token_id
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.use_num_logits_to_keep = args.use_num_logits_to_keep

        self.beta = args.beta
        self.loss_type = args.loss_type
        self.num_ref_rewards = args.num_ref_rewards

        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)
        self.aux_loss_coef = getattr(model.config, "router_aux_loss_coef", 0.0)
        if self.aux_loss_enabled and self.aux_loss_coef == 0.0:
            warnings.warn(
                "You set `output_router_logits` to True in the model config, but `router_aux_loss_coef` is set to 0.0,"
                " meaning the auxiliary loss will not be used."
            )

        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.dataset_num_proc = args.dataset_num_proc

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in DPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys are "prompt_input_ids", "chosen_input_ids", and
        # "rejected_input_ids". As a result, the trainer issues the warning: "Could not estimate the number of tokens
        # of the input, floating-point operations will not be computed." To suppress this warning, we set the
        # "estimate_tokens" key in the model's "warnings_issued" dictionary to True. This acts as a flag to indicate
        # that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Dataset preparation
        train_dataset = self._prepare_dataset(
            train_dataset, processing_class, args, "train"
        )
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    key: self._prepare_dataset(dataset, processing_class, args, key)
                    for key, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._prepare_dataset(
                    eval_dataset, processing_class, args, "eval"
                )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        # https://github.com/huggingface/trl/pull/2615/files
        self.model_accepts_loss_kwargs = False

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if (
                self.accelerator.state.deepspeed_plugin.zero_stage == 3
                and self.precompute_ref_log_probs
            ):
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
            if args.sync_ref_model:
                raise ValueError(
                    "You currently cannot use `ref_model=None` with TR-DPO method. Please provide `ref_model`."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        if args.sync_ref_model:
            if self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with `sync_ref_model=True`. Please set `precompute_ref_log_probs=False`."
                )

            self.add_callback(
                SyncRefModelCallback(
                    ref_model=self.ref_model, accelerator=self.accelerator
                )
            )

    def _prepare_dataset(
        self,
        dataset: Dataset,
        processing_class: Union[
            PreTrainedTokenizerBase,
            BaseImageProcessor,
            FeatureExtractionMixin,
            ProcessorMixin,
        ],
        args: PreferenceTrainerConfig,
        dataset_name: str,
    ) -> Dataset:
        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # Extract the prompt if needed, and apply the chat template if needed
            dataset = dataset.map(
                maybe_extract_prompt,
                num_proc=args.dataset_num_proc,
                desc="Extracting prompt from dataset",
            )
            dataset = dataset.map(
                maybe_apply_chat_template,
                fn_kwargs={"tokenizer": processing_class},
                num_proc=args.dataset_num_proc,
                desc="Applying chat template to dataset",
            )

            # tokenize the dataset, lower writer batch size to avoid OOM (frequent in vision models)
            fn_kwargs = {
                "processing_class": processing_class,
                "max_prompt_length": args.max_prompt_length,
                "max_completion_length": args.max_completion_length,
                # for enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
                "add_special_tokens": self.is_encoder_decoder,
            }
            dataset = dataset.map(
                self.tokenize_row if not self.is_vision_model else self.process_row,
                fn_kwargs=fn_kwargs,
                num_proc=self.dataset_num_proc,
                desc="Tokenizing dataset",
            )

        return dataset

    @staticmethod
    def tokenize_row(
        features,
        processing_class,
        max_prompt_length,
        max_completion_length,
        add_special_tokens,
    ):
        """
        Tokenize a row of the dataset.

        Args:
            features (`dict[str, str]`):
                Row of the dataset, should contain the keys `"prompt"`, `"chosen"`, and `"rejected"`.
            processing_class (`PreTrainedTokenizerBase`):
                Processing class used to process the data.
            max_prompt_length (`int` or `None`):
                Maximum length of the prompt sequence. If `None`, the prompt sequence is not truncated.
            max_completion_length (`int` or `None`):
                Maximum length of the completion sequences. If `None`, the completion sequences are not truncated.
            add_special_tokens (`bool`):
                Whether to add special tokens to the sequences. Typically used for encoder-decoder models. If `True`,
                the prompt sequence will have a bos token prepended and an eos token appended. In any case, the
                completion sequences will have an eos token appended.

        Returns:
            `dict[str, list[int]]`:
                Tokenized sequences with the keys `"prompt_input_ids"`, `"chosen_input_ids"`, and
                `"rejected_input_ids".

        Example:
        ```python
        >>> from transformers import GPT2Tokenizer
        >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        >>> features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}
        >>> PreferenceTrainer.tokenize_row(features, tokenizer, max_prompt_length=3, max_completion_length=3, add_special_tokens=False)
        {'prompt_input_ids': [464, 6766, 318], 'chosen_input_ids': [4171, 50256], 'rejected_input_ids': [4077, 50256]}
        ```
        """
        tokenizer = processing_class  # the processing class is a tokenizer
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)[
            "input_ids"
        ]
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)[
            "input_ids"
        ]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)[
            "input_ids"
        ]

        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        # Patch: Remove extra eos token as it is added in the maybe_apply_chat_template before already.
        chosen_input_ids = chosen_input_ids  # + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids  # + [tokenizer.eos_token_id]

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

    @staticmethod
    def process_row(
        features,
        processing_class,
        max_prompt_length,
        max_completion_length,
        add_special_tokens,
    ):
        """
        Same as `tokenize_row` but for vision models. Please refer to `tokenize_row` for more information.
        """
        processor, tokenizer = (
            processing_class,
            processing_class.tokenizer,
        )  # the processing class is a processor
        processed_features = processor(
            images=features["images"], text=features["prompt"], add_special_tokens=False
        )

        prompt_input_ids = processed_features["input_ids"][0]
        pixel_values = processed_features["pixel_values"][0]
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)[
            "input_ids"
        ]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)[
            "input_ids"
        ]

        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        # Patch: Remove extra eos token as it is added in the maybe_apply_chat_template before already.
        chosen_input_ids = chosen_input_ids  # + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids  # + [tokenizer.eos_token_id]

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        output = {
            "prompt_input_ids": prompt_input_ids,
            "pixel_values": pixel_values,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

        if "pixel_attention_mask" in processed_features:
            output["pixel_attention_mask"] = processed_features["pixel_attention_mask"][
                0
            ]
        if "image_sizes" in processed_features:
            output["image_sizes"] = processed_features["image_sizes"][0]

        return output

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        """Only used for the ref model."""
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if (
                    hidden_size is not None
                    and config_kwargs["zero_optimization"]["stage"] == 3
                ):
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size
                            * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10
                            * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9
                            * hidden_size
                            * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In PreferenceTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by `PreferenceCollator`, hence the override.
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_input_ids",
                "prompt_attention_mask",
                "chosen_input_ids",
                "chosen_attention_mask",
                "rejected_input_ids",
                "rejected_attention_mask",
                "ref_rewards",
                "chosen_rewards",
                "rejected_rewards",
                "ref_chosen_logps",
                "ref_rejected_logps",
            ]

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(
            self.model
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_ref_log_probs(self, batch: dict[str, torch.LongTensor]) -> dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compute_ref_context_manager = (
            amp.autocast("cuda")
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        )
        with torch.no_grad(), compute_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(self.model, batch)
            else:
                ref_model_output = self.concatenated_forward(self.ref_model, batch)

        output = {}

        if "chosen_input_ids" in batch.keys():
            output["ref_chosen_logps"] = ref_model_output["chosen_logps"]

        if "rejected_input_ids" in batch.keys():
            output["ref_rejected_logps"] = ref_model_output["rejected_logps"]

        return output

    @staticmethod
    def concatenated_inputs(
        batch: dict[str, Union[list, torch.LongTensor, torch.Tensor]],
        padding_value: int,
    ) -> dict[str, torch.LongTensor]:
        """
        Concatenate the prompt and completions inputs from the batch into a single tensor for both the prompt
        and completion sequences.
        And concatenate those from the chosen, rejected, and online on the zeroth dimension.

        Args:
            batch (`dict[str, Union[list, torch.LongTensor]]`):
                A batch of input data. The batch must contain the following keys:

                - `"prompt_input_ids"`: Tensor of shape `(batch_size, prompt_length)` representing the prompt input IDs.
                - `"chosen_input_ids"`: Tensor of shape `(batch_size, chosen_length)` representing the chosen completion input IDs.
                - `"rejected_input_ids"`: Tensor of shape `(batch_size, rejected_length)` representing the rejected completion input IDs.
                - `"prompt_pixel_values"` (optional): Tensor for pixel values, if available.
                - `"prompt_pixel_attention_mask"` (optional): Tensor for pixel attention masks, if available.

            padding_value (`int`):
                The padding value to use for the concatenated completion sequences (`chosen_input_ids` and
                `rejected_input_ids`).

        Returns:
            `dict[str, torch.LongTensor]`: A dictionary containing:

                - `"prompt_input_ids"`: Concatenated prompt input IDs of shape `(2 * batch_size, prompt_length)`.
                - `"completion_input_ids"`: Concatenated chosen, rejected, completion input IDs of shape `(2 * batch_size, max_completion_length)`.
                - `"prompt_attention_mask"`: Concatenated prompt attention masks of shape `(2 * batch_size, prompt_length)`.
                - `"completion_attention_mask"`: Concatenated chosen, rejected, attention masks of shape `(2 * batch_size, max_completion_length)`.
                - `"pixel_values"` (optional): Concatenated pixel values if `"prompt_pixel_values"` are present.
                - `"pixel_attention_mask"` (optional): Concatenated pixel attention masks if `"prompt_pixel_attention_mask"` are present.

        Notes:
            The completion input IDs and attention masks are padded to the maximum completion length of the chosen and rejected sequences.
        """
        output = {}
        assert "chosen_input_ids" in batch and "rejected_input_ids" in batch
        # For the prompt, the input_ids are the same for both the chosen and rejected responses
        output["prompt_input_ids"] = torch.cat(
            [batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0
        )
        output["prompt_attention_mask"] = torch.cat(
            [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
        )
        if "pixel_values" in batch:
            output["pixel_values"] = torch.cat(
                [batch["pixel_values"], batch["pixel_values"]], dim=0
            )

        if "pixel_attention_mask" in batch:
            output["pixel_attention_mask"] = torch.cat(
                [batch["pixel_attention_mask"], batch["pixel_attention_mask"]],
                dim=0,
            )
        if "image_sizes" in batch:
            output["image_sizes"] = torch.cat(
                [batch["image_sizes"], batch["image_sizes"]], dim=0
            )

        # Concatenate the chosen and rejected completions
        max_completion_length = max(
            batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1]
        )
        output["completion_input_ids"] = torch.cat(
            (
                pad_to_length(
                    batch["chosen_input_ids"],
                    max_completion_length,
                    pad_value=padding_value,
                ),
                pad_to_length(
                    batch["rejected_input_ids"],
                    max_completion_length,
                    pad_value=padding_value,
                ),
            ),
        )
        output["completion_attention_mask"] = torch.cat(
            (
                pad_to_length(
                    batch["chosen_attention_mask"],
                    max_completion_length,
                    pad_value=0,
                ),
                pad_to_length(
                    batch["rejected_attention_mask"],
                    max_completion_length,
                    pad_value=0,
                ),
            ),
        )

        return output

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]
    ):
        """Run the given model on the given batch of inputs, concatenating the chosen, rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(
            batch, padding_value=self.pad_token_id
        )

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Add the pixel values and attention masks for vision models
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch[
                "pixel_attention_mask"
            ]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat(
                (prompt_attention_mask, completion_attention_mask), dim=1
            )
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush left to reduce the memory usage
            # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
            #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
            for i in range(attention_mask.size(0)):
                first_one_idx = torch.nonzero(attention_mask[i])[0].item()
                input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
                attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
                loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

            # Get the first column idx that is all zeros and remove every column after that
            empty_cols = torch.sum(attention_mask, dim=0) == 0
            first_empty_col = (
                torch.nonzero(empty_cols)[0].item()
                if empty_cols.any()
                else attention_mask.size(1)
            )
            input_ids = input_ids[:, :first_empty_col]
            attention_mask = attention_mask[:, :first_empty_col]
            loss_mask = loss_mask[:, :first_empty_col]

            # Truncate right
            if self.max_length is not None:
                input_ids = input_ids[:, : self.max_length]
                attention_mask = attention_mask[:, : self.max_length]
                loss_mask = loss_mask[:, : self.max_length]

            if self.use_num_logits_to_keep:
                # Compute num_logits_to_keep based on loss_mask pattern:
                # [[0, 0, 0, x, x, x, x],
                #  [0, 0, 0, x, x, x, 0]]
                #         ^ start computing logits from here ([:, -(7-3+1):])
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                num_logits_to_keep = loss_mask.shape[1] - first_compute_index
                model_kwargs["num_logits_to_keep"] = (
                    num_logits_to_keep.item() + 1
                )  # +1 for the first label

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, **model_kwargs
            )

            # Offset the logits by one to align with the labels
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:].clone()
            loss_mask = loss_mask[:, 1:].bool()

            if self.use_num_logits_to_keep:
                # Align labels with logits
                # logits:    -,  -, [x2, x3, x4, x5, x6]
                #                     ^ --------- ^       after logits[:, :-1, :]
                # labels:   [y0, y1, y2, y3, y4, y5, y6]
                #                         ^ --------- ^   with num_logits_to_keep=4, [:, -4:]
                # loss_mask: [0,  0,  0,  1,  1,  1,  1]
                labels = labels[:, -num_logits_to_keep:]
                loss_mask = loss_mask[:, -num_logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[
            ~loss_mask
        ] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)
        per_token_logps[~loss_mask] = 0

        all_logps = per_token_logps.sum(-1)
        all_lens = loss_mask.sum(-1)

        output = {}
        assert "chosen_input_ids" in batch and "rejected_input_ids" in batch
        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["chosen_lens"] = all_lens[:num_examples]
        output["rejected_lens"] = all_lens[num_examples:]
        output["mean_chosen_logits"] = logits[:num_examples][
            loss_mask[:num_examples]
        ].mean()
        output["mean_rejected_logits"] = logits[num_examples:][
            loss_mask[num_examples:]
        ].mean()
        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        compute_loss_context_manager = (
            amp.autocast("cuda")
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        )
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(
                model, inputs, train_eval="train"
            )

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics

        return loss

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the PreferenceTrainer loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        model_output = self.concatenated_forward(model, batch)

        # if ref_chosen_logps and ref_rejected_logps in batch use them, otherwise use the reference model
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            # Compute logprobs
            logps = self.compute_ref_log_probs(batch)
            ref_chosen_logps = logps["ref_chosen_logps"]
            ref_rejected_logps = logps["ref_rejected_logps"]

        if (
            "chosen_rewards" in batch
            and "rejected_rewards" in batch
            and "ref_rewards" in batch
        ):
            chosen_rewards = batch["chosen_rewards"]
            rejected_rewards = batch["rejected_rewards"]
            ref_rewards = batch["ref_rewards"]
        else:
            chosen_rewards, rejected_rewards, ref_rewards = None, None, None

        if self.loss_type == "qrpo":
            loss_fc = self.qrpo_loss
        elif self.loss_type == "dpo":
            loss_fc = self.dpo_loss
        else:
            raise ValueError('args.loss_type must be either "qrpo", "dpo" or "rebel"')

        extra_logs = {}
        (
            losses,
            chosen_rewards_prediction,
            rejected_rewards_prediction,
        ) = loss_fc(
            model_output,
            ref_chosen_logps,
            ref_rejected_logps,
            chosen_rewards,
            rejected_rewards,
            ref_rewards,
            extra_logs,
        )

        # Gather the essential metrics
        # g_ stands for gathered
        g_chosen_rewards_prediction = self.accelerator.gather_for_metrics(
            chosen_rewards_prediction.detach()
        )
        g_rejected_rewards_prediction = self.accelerator.gather_for_metrics(
            rejected_rewards_prediction.detach()
        )
        g_chosen_logps = self.accelerator.gather_for_metrics(
            model_output["chosen_logps"].detach()
        )
        g_rejected_logps = self.accelerator.gather_for_metrics(
            model_output["rejected_logps"].detach()
        )
        g_mean_chosen_logits = self.accelerator.gather_for_metrics(
            model_output["mean_chosen_logits"].detach()
        )
        g_mean_rejected_logits = self.accelerator.gather_for_metrics(
            model_output["mean_rejected_logits"].detach()
        )

        if self.loss_type == "qrpo":
            g_chosen_rewards = self.accelerator.gather_for_metrics(
                chosen_rewards.detach()
            )
            g_rejected_rewards = self.accelerator.gather_for_metrics(
                rejected_rewards.detach()
            )
            g_quantile_rewards_chosen = self.accelerator.gather_for_metrics(
                extra_logs["quantile_rewards/chosen"]
            )
            g_quantile_rewards_rejected = self.accelerator.gather_for_metrics(
                extra_logs["quantile_rewards/rejected"]
            )
            g_log_Z = self.accelerator.gather_for_metrics(extra_logs["log_Z"])
            g_calibrated_targets_chosen = (
                g_quantile_rewards_chosen - self.beta * g_log_Z
            )
            g_calibrated_targets_rejected = (
                g_quantile_rewards_rejected - self.beta * g_log_Z
            )

        g_rewards_prediction_accuracies = (
            g_chosen_rewards_prediction > g_rejected_rewards_prediction
        )
        g_sign_chosen_dpo_rewards = torch.sign(g_chosen_rewards_prediction)
        g_sign_rejected_dpo_rewards = torch.sign(g_rejected_rewards_prediction)

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval/" if train_eval == "eval" else ""

        metrics[
            f"{prefix}dpo-rewards/chosen"
        ] = g_chosen_rewards_prediction.mean().item()
        metrics[
            f"{prefix}dpo-rewards/rejected"
        ] = g_rejected_rewards_prediction.mean().item()
        metrics[f"{prefix}dpo-rewards/margins"] = (
            (g_chosen_rewards_prediction - g_rejected_rewards_prediction).mean().item()
        )
        metrics[f"{prefix}dpo-rewards/accuracy-chosen-rejected"] = (
            g_rewards_prediction_accuracies.float().mean().item()
        )

        metrics[f"{prefix}no-beta-dpo-rewards/chosen"] = (
            metrics[f"{prefix}dpo-rewards/chosen"] / self.beta
        )
        metrics[f"{prefix}no-beta-dpo-rewards/rejected"] = (
            metrics[f"{prefix}dpo-rewards/rejected"] / self.beta
        )
        metrics[f"{prefix}no-beta-dpo-rewards/margins"] = (
            metrics[f"{prefix}dpo-rewards/margins"] / self.beta
        )
        metrics[
            f"{prefix}sign-dpo-rewards/chosen"
        ] = g_sign_chosen_dpo_rewards.mean().item()
        metrics[
            f"{prefix}sign-dpo-rewards/rejected"
        ] = g_sign_rejected_dpo_rewards.mean().item()
        metrics[f"{prefix}sign-dpo-rewards/margins"] = (
            ((g_sign_chosen_dpo_rewards - g_sign_rejected_dpo_rewards) / 2)
            .mean()
            .item()
        )
        metrics[f"{prefix}logps/chosen"] = g_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = g_rejected_logps.mean().item()

        metrics[f"{prefix}mean-logits/chosen"] = g_mean_chosen_logits.mean().item()
        metrics[f"{prefix}mean-logits/rejected"] = g_mean_rejected_logits.mean().item()

        if self.loss_type == "qrpo":
            metrics[f"{prefix}rewards/chosen"] = g_chosen_rewards.mean().item()
            metrics[f"{prefix}rewards/rejected"] = g_rejected_rewards.mean().item()
            metrics[f"{prefix}rewards/margins"] = (
                (g_chosen_rewards - g_rejected_rewards).mean().item()
            )
            metrics[
                f"{prefix}quantile-rewards/chosen"
            ] = g_quantile_rewards_chosen.mean().item()
            metrics[
                f"{prefix}quantile-rewards/rejected"
            ] = g_quantile_rewards_rejected.mean().item()
            metrics[f"{prefix}quantile-rewards/margins"] = (
                (g_quantile_rewards_chosen - g_quantile_rewards_rejected).mean().item()
            )
            metrics[f"{prefix}quantile-rewards/sign-of-margins"] = (
                torch.sign(g_quantile_rewards_chosen - g_quantile_rewards_rejected)
                .float()
                .mean()
                .item()
            )
            metrics[f"{prefix}quantile-rewards/logZ"] = g_log_Z.mean().item()
            metrics[f"{prefix}quantile-rewards/beta-logZ"] = (
                self.beta * metrics[f"{prefix}quantile-rewards/logZ"]
            )
            # quantile reward margins conditioned on the accuracy.
            # I.e. mean margin when logratio chosen > logratio rejected and margin in the other case
            metrics[
                f"{prefix}quantile-rewards/sign-of-margin-when-chosen-predicted-better"
            ] = torch.sign(g_quantile_rewards_chosen - g_quantile_rewards_rejected)[
                g_rewards_prediction_accuracies
            ].mean()
            metrics[
                f"{prefix}quantile-rewards/sign-of-margin-when-rejected-predicted-better"
            ] = torch.sign(g_quantile_rewards_chosen - g_quantile_rewards_rejected)[
                ~g_rewards_prediction_accuracies
            ].mean()

            # quantile reward - beta * log Z (compare to beta * logratio)
            metrics[f"{prefix}calibrated-targets/chosen"] = (
                (g_calibrated_targets_chosen).mean().item()
            )
            metrics[f"{prefix}calibrated-targets/rejected"] = (
                (g_calibrated_targets_rejected).mean().item()
            )
            metrics[f"{prefix}calibrated-targets/margins"] = metrics[
                f"{prefix}quantile-rewards/margins"
            ]
            metrics[f"{prefix}calibrated-targets/accuracy-chosen"] = (
                (
                    torch.sign(g_calibrated_targets_chosen)
                    * torch.sign(g_chosen_rewards_prediction)
                )
                .float()
                .mean()
                .item()
            )
            metrics[f"{prefix}calibrated-targets/accuracy-rejected"] = (
                (
                    torch.sign(g_calibrated_targets_rejected)
                    * torch.sign(g_rejected_rewards_prediction)
                )
                .float()
                .mean()
                .item()
            )
            # calibrated_targets / beta - log Z  (compare to logratio)
            metrics[f"{prefix}no-beta-calibrated-targets/chosen"] = (
                metrics[f"{prefix}calibrated-targets/chosen"] / self.beta
            )
            metrics[f"{prefix}no-beta-calibrated-targets/rejected"] = (
                metrics[f"{prefix}calibrated-targets/rejected"] / self.beta
            )
            metrics[f"{prefix}no-beta-calibrated-targets/margins"] = (
                metrics[f"{prefix}calibrated-targets/margins"] / self.beta
            )
            metrics[f"{prefix}no-beta-calibrated-targets/accuracy-chosen"] = metrics[
                f"{prefix}calibrated-targets/accuracy-chosen"
            ]
            metrics[f"{prefix}no-beta-calibrated-targets/accuracy-rejected"] = metrics[
                f"{prefix}calibrated-targets/accuracy-rejected"
            ]

        if self.aux_loss_enabled:
            metrics[f"{prefix}aux-loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"].detach())
                .mean()
                .item()
            )

        return losses.mean(), metrics

    def qrpo_loss(
        self,
        model_outputs: dict[str, torch.Tensor],
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        ref_rewards: torch.Tensor,
        extra_logs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, float | Any, float | Any]:
        """
        Compute the QRPO loss for a batch of policy and reference model log probabilities.

        Args:
            model_outputs (`dict`): containing
                chosen_logps (`torch.Tensor`):
                    Log probabilities of the model for the chosen responses. Shape: `(batch_size,)`.
                rejected_logps (`torch.Tensor`):
                    Log probabilities of the model for the rejected responses. Shape: `(batch_size,)`.
            ref_chosen_logps (`torch.Tensor`):
                Log probabilities of the reference model for the chosen responses. Shape: `(batch_size,)`.
            ref_rejected_logps (`torch.Tensor`):
                Log probabilities of the reference model for the rejected responses. Shape: `(batch_size,)`.
            chosen_rewards (`torch.Tensor`):
                Rewards from the reward model for the chosen responses. Shape: `(batch_size,)`.
            rejected_rewards (`torch.Tensor`):
                Rewards from the reward model for the rejected responses. Shape: `(batch_size,)`.
            ref_rewards (`torch.Tensor`):
                Rewards from the reward model for the reference model generated responses. Shape: `(batch_size, num_ref_rewards)`.

        Returns:
            A tuple of four tensors: `(losses, chosen_rewards, rejected_rewards, online_rewards)`.
            The losses tensor contains the QRPO loss for each example in the batch.
        """
        device = self.accelerator.device
        chosen_logps = model_outputs["chosen_logps"].to(device)
        rejected_logps = model_outputs["rejected_logps"].to(device)
        ref_chosen_logps = ref_chosen_logps.to(device)
        ref_rejected_logps = ref_rejected_logps.to(device)
        ref_rewards = ref_rewards.to(device)

        # Get the log ratios for the chosen and rejected responses
        logratio_chosen = chosen_logps - ref_chosen_logps
        logratio_rejected = rejected_logps - ref_rejected_logps

        chosen_quantile_rewards = (
            (ref_rewards <= chosen_rewards.unsqueeze(dim=-1)).float().mean(dim=1)
        )
        rejected_quantile_rewards = (
            (ref_rewards <= rejected_rewards.unsqueeze(dim=-1)).float().mean(dim=1)
        )

        extra_logs[("quantile_" "rewards/chosen")] = chosen_quantile_rewards
        extra_logs["quantile_rewards/rejected"] = rejected_quantile_rewards

        beta = self.beta * torch.ones_like(chosen_quantile_rewards)
        # Z = beta * (torch.exp(1 / beta) - 1) -> simplified to beta * torch.exp(1 / beta) -> use log directly
        log_Z = torch.log(beta) + 1 / beta
        extra_logs["log_Z"] = log_Z

        calibrated_targets_chosen = chosen_quantile_rewards - self.beta * log_Z
        calibrated_targets_rejected = rejected_quantile_rewards - self.beta * log_Z

        extra_logs["calibrated_targets/chosen"] = calibrated_targets_chosen
        extra_logs["calibrated_targets/rejected"] = calibrated_targets_rejected

        loss_chosen = (calibrated_targets_chosen - self.beta * logratio_chosen) ** 2
        loss_rejected = (
            calibrated_targets_rejected - self.beta * logratio_rejected
        ) ** 2

        losses = (loss_chosen + loss_rejected) / 2

        chosen_rewards_prediction = self.beta * logratio_chosen.detach()
        rejected_rewards_prediction = self.beta * logratio_rejected.detach()

        return losses, chosen_rewards_prediction, rejected_rewards_prediction

    def dpo_loss(
        self,
        model_outputs: dict[str, torch.Tensor],
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        *args: Any,
    ) -> tuple[torch.Tensor, float | Any, float | Any]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            model_outputs (`dict`): containing
                chosen_logps (`torch.FloatTensor`):
                    Log probabilities of the model for the chosen responses. Shape: `(batch_size,)`.
                rejected_logps (`torch.FloatTensor`):
                    Log probabilities of the model for the rejected responses. Shape: `(batch_size,)`.
            ref_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: `(batch_size,)`.
            ref_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: `(batch_size,)`.

        Returns:
            A tuple of three tensors: `(losses, chosen_rewards, rejected_rewards)`.
            The losses tensor contains the DPO loss for each example in the batch.
            The `chosen_rewards` and `rejected_rewards` tensors contain the rewards for the chosen and rejected
            responses, respectively.
        """
        device = self.accelerator.device

        # using notation similar to DPOTrainer
        chosen_logps = model_outputs["chosen_logps"].to(device)
        rejected_logps = model_outputs["rejected_logps"].to(device)
        ref_chosen_logps = ref_chosen_logps.to(device)
        ref_rejected_logps = ref_rejected_logps.to(device)

        logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = logratios - ref_logratios

        losses = -F.logsigmoid(self.beta * logits)
        chosen_rewards_prediction = (
            self.beta * (chosen_logps - ref_chosen_logps).detach()
        )
        rejected_rewards_prediction = (
            self.beta * (rejected_logps - ref_rejected_logps).detach()
        )

        return losses, chosen_rewards_prediction, rejected_rewards_prediction

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = (
            amp.autocast("cuda")
            if self._peft_has_been_casted_to_bf16
            else nullcontext()
        )
        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(
                model, inputs, train_eval="eval"
            )

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return loss.detach(), None, None

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = [v for k, v in logits_dict.items() if k not in ignore_keys]
        logits = torch.tensor(logits, device=self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(
        self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        return initial_output

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Saves eval metrics to files"""
        acc_logger = get_logger(__name__)
        acc_logger.info("\nEvaluating model\n")
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        self.log_metrics(f"eval_{self.state.global_step}", metrics)
        self.save_metrics(f"eval_{self.state.global_step}", metrics, combined=False)
        return metrics

    def log(
        self,
        logs: dict[str, Union[float, wandb.Table]],
        start_time: Optional[float] = None,
    ) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`float` or `None`, *optional*, defaults to `None`):
                Start time of the training.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            return super().log(logs, start_time)
        else:  # transformers<=4.46
            return super().log(logs)
