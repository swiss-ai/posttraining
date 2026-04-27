"""Shared internal key names for QRPO batches."""

# VERL/model tensor keys
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
POSITION_IDS = "position_ids"

LOG_PROBS = "log_probs"
REF_LOG_PROBS = "ref_log_probs"

# VERL rollout tensor keys
PROMPTS = "prompts"
RESPONSES = "responses"
RESPONSE_MASK = "response_mask"

# QRPO tensor keys
TRAJECTORY_REWARD = "trajectory_reward"
REF_REWARDS = "ref_rewards"

REF_QUANTILE = "ref_quantile"
TRANSFORMED_REWARD = "transformed_reward"
TRAJECTORY_LENGTH = "trajectory_length"
EFFECTIVE_BETA = "effective_beta"
LOG_PARTITION = "log_partition"
BETA_LOG_PARTITION = "beta_log_partition"

SEQUENCE_LOG_RATIO = "sequence_log_ratio"
QRPO_RESIDUAL = "qrpo_residual"
QRPO_LOSS = "qrpo_loss"

# VERL reward-output keys
REWARD_TENSOR = "reward_tensor"
REWARD_EXTRA_INFO = "reward_extra_info"

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

# VERL AgentLoop input keys
RAW_PROMPT = "raw_prompt"
AGENT_NAME = "agent_name"
DATA_SOURCE = "data_source"
REWARD_MODEL = "reward_model"