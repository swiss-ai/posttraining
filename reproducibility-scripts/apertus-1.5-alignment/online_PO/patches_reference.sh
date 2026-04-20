#!/bin/bash
# Reference: patches that were previously applied via sed at runtime.
# These should now be applied directly to the davmels/verl-recipe fork.
# -----------------------------------------------------------------------

# 1. Fix Hydra searchpath in config
# sed -i 's|file://verl/trainer/config|pkg://verl.trainer.config|' spin/config/spin_trainer.yaml

# 2. Remove max_samples kwarg not present in verl 0.6's create_rl_dataset
# sed -i '/max_samples=self\.config\.data\.get/d' spin/spin_trainer.py

# 3. Fix need_reference_policy calls for verl 0.7 (API changed from dict to config)
# SPIN always uses a reference policy, so hardcode True
# sed -i 's|need_reference_policy(role_worker_mapping)|True|g' spin/main_spin.py
# sed -i 's|need_reference_policy(role_worker_mapping)|True|g' spin/spin_trainer.py

# 4. SPIN doesn't use a reward model, so hardcode False
# sed -i 's|need_reward_model(role_worker_mapping)|False|g' spin/spin_trainer.py

# 5. Patch fsdp_workers.py: add _qat_enabled attribute for verl 0.7 compat
# sed -i 's|def init_model(self):|def init_model(self):\n        self._qat_enabled = False|' spin/fsdp_workers.py
