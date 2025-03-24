import numpy as np
from transformers.data.data_collator import DataCollatorForLanguageModeling


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
