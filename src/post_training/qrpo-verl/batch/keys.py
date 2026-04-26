"""Shared internal key names for QRPO batches."""

# VERL/model tensor keys
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
POSITION_IDS = "position_ids"

# QRPO tensor keys
LOSS_MASK = "loss_mask"
TRAJECTORY_REWARD = "trajectory_reward"
REF_REWARDS = "ref_rewards"

# Metadata keys
PROMPT_ID = "prompt_id"
TRAJECTORY_ID = "trajectory_id"

PROMPT_MESSAGES = "prompt_messages"
TRAJECTORY_MESSAGES = "trajectory_messages"
TRAJECTORY_TEXT = "trajectory_text"

TRAJECTORY_METADATA = "trajectory_metadata"
SOURCE_MODEL = "source_model"

SOURCE = "source"
SOURCE_ONLINE = "online"
SOURCE_OFFLINE = "offline"

ACTOR_VERSION = "actor_version"
REF_VERSION = "ref_version"

GROUP_ID = "group_id"
