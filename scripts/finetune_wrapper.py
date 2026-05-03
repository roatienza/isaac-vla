#!/usr/bin/env python3
"""
Wrapper to run OpenVLA-OFT finetune.py with wandb disabled.
Patches wandb.init before any imports to avoid API key requirement.
"""
import os
import sys
from pathlib import Path

# Disable wandb before any imports
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DISABLED"] = "true"

# Patch wandb module to be a no-op
class FakeWandbRun:
    def log(self, *args, **kwargs):
        pass
    def finish(self, *args, **kwargs):
        pass

class FakeWandb:
    @staticmethod
    def init(*args, **kwargs):
        print("[wandb] Disabled - running offline")
        return FakeWandbRun()
    @staticmethod
    def log(*args, **kwargs):
        pass

sys.modules["wandb"] = FakeWandb()

# Paths
OPENVLA_ROOT = Path("/home/rowel/sandbox/openvla-oft")
FINETUNE_SCRIPT = OPENVLA_ROOT / "vla-scripts/finetune.py"

# Replace sys.argv with finetune script args
sys.argv = [str(FINETUNE_SCRIPT)] + sys.argv[1:]

# Execute the finetune script
exec(compile(FINETUNE_SCRIPT.read_text(), str(FINETUNE_SCRIPT), 'exec'),
     {"__name__": "__main__", "__file__": str(FINETUNE_SCRIPT)})
