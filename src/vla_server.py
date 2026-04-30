"""
VLA Server: OpenVLA-OFT Inference Server
==========================================

FastAPI server that loads an OpenVLA-OFT model and exposes a REST API
for action prediction. Runs on the GPU machine (RTX 5090).

Endpoints:
    POST /act          - Predict action chunk from image + instruction
    GET  /health       - Health check
    GET  /model_info   - Model configuration info
    POST /warmup       - Trigger model warmup

Prerequisites:
    The openvla-oft package must be installed. Run:
        git clone https://github.com/moojink/openvla-oft.git
        cd openvla-oft && pip install -e .

Usage:
    python scripts/run_vla_server.py --config config/default.yaml
    # Or programmatically:
    from src.vla_server import VLAServer
    server = VLAServer(config)
    server.run()
"""

import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

logger = logging.getLogger(__name__)


# ── OpenVLA-OFT Import Helpers ──────────────────────────────────────────────

def _find_openvla_oft_root() -> Optional[str]:
    """Find the openvla-oft installation root directory.

    Checks in order:
    1. OPENVLA_OFT_ROOT environment variable
    2. openvla-oft as an installed Python package (pip install -e .)
    3. Sibling directory to this project (../openvla-oft)
    4. Home directory clone (~/openvla-oft)
    """
    # 1. Explicit env var
    env_root = os.environ.get("OPENVLA_OFT_ROOT")
    if env_root and Path(env_root).exists():
        return env_root

    # 2. Check if installed as package — find the package location
    try:
        import experiments.robot.openvla_utils  # noqa: F401
        # Package is importable, find its location
        pkg_dir = Path(experiments.robot.openvla_utils.__file__).parent.parent.parent
        return str(pkg_dir)
    except ImportError:
        pass

    # 3. Sibling directory
    project_root = Path(__file__).parent.parent
    sibling = project_root.parent / "openvla-oft"
    if sibling.exists() and (sibling / "experiments").exists():
        return str(sibling)

    # 4. Home directory
    home_clone = Path.home() / "openvla-oft"
    if home_clone.exists() and (home_clone / "experiments").exists():
        return str(home_clone)

    return None


def _ensure_openvla_oft_importable() -> None:
    """Ensure the openvla-oft modules are importable by adding to sys.path.

    Raises ImportError with installation instructions if not found.
    """
    # First try: already importable (pip install -e . was done)
    try:
        from experiments.robot.openvla_utils import get_vla  # noqa: F401
        return  # Already importable
    except ImportError:
        pass

    # Second try: find the repo and add to sys.path
    oft_root = _find_openvla_oft_root()
    if oft_root:
        oft_root = str(oft_root)
        if oft_root not in sys.path:
            sys.path.insert(0, oft_root)
            logger.info(f"Added openvla-oft to sys.path: {oft_root}")
        # Verify it works now
        try:
            from experiments.robot.openvla_utils import get_vla  # noqa: F401
            return
        except ImportError as e:
            logger.error(f"Found openvla-oft at {oft_root} but import still fails: {e}")

    # Failed — raise with instructions
    raise ImportError(
        "Cannot import openvla-oft modules. Please install openvla-oft:\n\n"
        "  # Option 1: Clone and install as editable package (recommended)\n"
        "  git clone https://github.com/moojink/openvla-oft.git\n"
        "  cd openvla-oft && pip install -e .\n\n"
        "  # Option 2: Set environment variable to openvla-oft root\n"
        "  export OPENVLA_OFT_ROOT=/path/to/openvla-oft\n\n"
        "  # Option 3: Clone as a sibling directory\n"
        "  git clone https://github.com/moojink/openvla-oft.git ../openvla-oft\n"
    )


# ── Configuration ────────────────────────────────────────────────────────────

class VLAConfig:
    """Configuration for the VLA server."""

    def __init__(
        self,
        pretrained_checkpoint: str = "moojink/openvla-7b-oft-finetuned-libero-spatial",
        use_l1_regression: bool = True,
        use_diffusion: bool = False,
        use_film: bool = False,
        num_images_in_input: int = 2,
        use_proprio: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        center_crop: bool = True,
        num_open_loop_steps: int = 8,
        unnorm_key: str = "libero_spatial_no_noops",
        device: str = "cuda:0",
        host: str = "0.0.0.0",
        port: int = 8777,
        warmup_on_start: bool = True,
        openvla_oft_root: Optional[str] = None,
        **kwargs,
    ):
        self.pretrained_checkpoint = pretrained_checkpoint
        self.use_l1_regression = use_l1_regression
        self.use_diffusion = use_diffusion
        self.use_film = use_film
        self.num_images_in_input = num_images_in_input
        self.use_proprio = use_proprio
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.center_crop = center_crop
        self.num_open_loop_steps = num_open_loop_steps
        self.unnorm_key = unnorm_key
        self.device = device
        self.host = host
        self.port = port
        self.warmup_on_start = warmup_on_start
        self.openvla_oft_root = openvla_oft_root
        # Store any extra keys from config YAML for forward compatibility
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


# ── VLA Server ───────────────────────────────────────────────────────────────

class VLAServer:
    """
    OpenVLA-OFT inference server.

    Loads the model on startup and exposes a REST API for action prediction.
    Supports both single-image and dual-image (third-person + wrist) inputs.
    """

    def __init__(self, config: Optional[VLAConfig] = None, config_dict: Optional[Dict] = None):
        if config is not None:
            self.config = config
        elif config_dict is not None:
            self.config = VLAConfig(**config_dict)
        else:
            self.config = VLAConfig()

        # If openvla_oft_root is set in config, add to sys.path and env
        if self.config.openvla_oft_root:
            oft_root = str(self.config.openvla_oft_root)
            os.environ["OPENVLA_OFT_ROOT"] = oft_root
            if oft_root not in sys.path:
                sys.path.insert(0, oft_root)
                logger.info(f"Added openvla-oft to sys.path from config: {oft_root}")

        self.vla = None
        self.processor = None
        self.action_head = None
        self.proprio_projector = None
        self._warm = False
        self._app = None
        self._oft_imports = None  # Cache for imported modules

    def _get_oft_imports(self):
        """Lazy-load and cache OpenVLA-OFT imports."""
        if self._oft_imports is not None:
            return self._oft_imports

        _ensure_openvla_oft_importable()

        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
            get_vla,
            get_vla_action,
        )
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

        self._oft_imports = {
            "get_action_head": get_action_head,
            "get_processor": get_processor,
            "get_proprio_projector": get_proprio_projector,
            "get_vla": get_vla,
            "get_vla_action": get_vla_action,
            "NUM_ACTIONS_CHUNK": NUM_ACTIONS_CHUNK,
            "PROPRIO_DIM": PROPRIO_DIM,
        }
        return self._oft_imports

    def load_model(self):
        """Load the OpenVLA-OFT model and components."""
        logger.info(f"Loading OpenVLA-OFT model: {self.config.pretrained_checkpoint}")
        start_time = time.time()

        # Import OpenVLA-OFT utilities (with auto-discovery)
        oft = self._get_oft_imports()
        get_vla = oft["get_vla"]
        get_processor = oft["get_processor"]
        get_action_head = oft["get_action_head"]
        get_proprio_projector = oft["get_proprio_projector"]
        PROPRIO_DIM = oft["PROPRIO_DIM"]

        # Create config object compatible with OpenVLA-OFT
        from experiments.robot.libero.run_libero_eval import GenerateConfig

        cfg = GenerateConfig(
            pretrained_checkpoint=self.config.pretrained_checkpoint,
            use_l1_regression=self.config.use_l1_regression,
            use_diffusion=self.config.use_diffusion,
            use_film=self.config.use_film,
            num_images_in_input=self.config.num_images_in_input,
            use_proprio=self.config.use_proprio,
            load_in_8bit=self.config.load_in_8bit,
            load_in_4bit=self.config.load_in_4bit,
            center_crop=self.config.center_crop,
            num_open_loop_steps=self.config.num_open_loop_steps,
            unnorm_key=self.config.unnorm_key,
        )

        # Load model components
        logger.info("Loading VLA backbone...")
        self.vla = get_vla(cfg)
        logger.info("Loading processor...")
        self.processor = get_processor(cfg)
        logger.info("Loading action head...")
        self.action_head = get_action_head(cfg, llm_dim=self.vla.llm_dim)
        logger.info("Loading proprio projector...")
        self.proprio_projector = get_proprio_projector(
            cfg, llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM
        )

        # Move to device
        self.vla = self.vla.to(self.config.device)
        self.action_head = self.action_head.to(self.config.device)
        self.proprio_projector = self.proprio_projector.to(self.config.device)

        elapsed = time.time() - start_time
        logger.info(f"Model loaded in {elapsed:.1f}s on {self.config.device}")

        return self

    def warmup(self, num_steps: int = 3):
        """Run dummy inference to warm up the model."""
        if self._warm:
            return

        logger.info("Warming up model...")

        # Create dummy observation matching OpenVLA-OFT expected format
        dummy_obs = {
            "full_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "wrist_image": np.zeros((128, 128, 3), dtype=np.uint8),
            "state": np.zeros(8, dtype=np.float32),  # 7 joints + gripper
            "task_description": "pick up the block",
        }

        for i in range(num_steps):
            self.predict_action(dummy_obs, "pick up the block")

        self._warm = True
        logger.info("Model warmup complete")

    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: str,
        unnorm_key: Optional[str] = None,
    ) -> List[np.ndarray]:
        """
        Predict an action chunk given observation and instruction.

        Args:
            observation: Dict with keys:
                - "full_image": np.ndarray (H, W, 3) uint8 third-person image
                - "wrist_image": np.ndarray (H, W, 3) uint8 wrist camera image
                - "state": np.ndarray (8,) float32 proprioceptive state
            instruction: Natural language task description
            unnorm_key: Dataset statistics key for action denormalization

        Returns:
            List of action arrays (each is 7-DOF: dx, dy, dz, droll, dpitch, dyaw, gripper)
        """
        oft = self._get_oft_imports()
        get_vla_action = oft["get_vla_action"]

        if self.vla is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if unnorm_key is None:
            unnorm_key = self.config.unnorm_key

        actions = get_vla_action(
            cfg=self._get_config(),
            vla=self.vla,
            processor=self.processor,
            obs=observation,
            task_label=instruction,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
        )

        return actions

    def _get_config(self):
        """Get the GenerateConfig object for OpenVLA-OFT utilities."""
        from experiments.robot.libero.run_libero_eval import GenerateConfig

        return GenerateConfig(
            pretrained_checkpoint=self.config.pretrained_checkpoint,
            use_l1_regression=self.config.use_l1_regression,
            use_diffusion=self.config.use_diffusion,
            use_film=self.config.use_film,
            num_images_in_input=self.config.num_images_in_input,
            use_proprio=self.config.use_proprio,
            load_in_8bit=self.config.load_in_8bit,
            load_in_4bit=self.config.load_in_4bit,
            center_crop=self.config.center_crop,
            num_open_loop_steps=self.config.num_open_loop_steps,
            unnorm_key=self.config.unnorm_key,
        )

    def create_app(self) -> FastAPI:
        """Create the FastAPI application."""
        self._app = FastAPI(
            title="Isaac-VLA Server",
            description="OpenVLA-OFT inference server for Franka manipulation",
            version="0.1.0",
        )

        @self._app.post("/act")
        async def predict_action(payload: Dict[str, Any]):
            """Predict action chunk from observation and instruction."""
            try:
                # Handle double-encoded payloads (json_numpy)
                if "encoded" in payload:
                    payload = json.loads(payload["encoded"])

                # Decode images from base64 if needed
                image = payload.get("image")
                wrist_image = payload.get("wrist_image")
                instruction = payload["instruction"]
                state = payload.get("state")
                unnorm_key = payload.get("unnorm_key", self.config.unnorm_key)

                # Convert image data
                if isinstance(image, list):
                    image = np.array(image, dtype=np.uint8)

                observation = {
                    "full_image": image,
                    "task_description": instruction,
                }

                if wrist_image is not None:
                    if isinstance(wrist_image, list):
                        wrist_image = np.array(wrist_image, dtype=np.uint8)
                    observation["wrist_image"] = wrist_image

                if state is not None:
                    if isinstance(state, list):
                        state = np.array(state, dtype=np.float32)
                    observation["state"] = state

                # Run inference
                start_time = time.time()
                actions = self.predict_action(observation, instruction, unnorm_key)
                inference_time = time.time() - start_time

                # Serialize actions
                result_actions = []
                for act in actions:
                    if isinstance(act, np.ndarray):
                        result_actions.append(act.tolist())
                    else:
                        result_actions.append(float(act))

                return JSONResponse({
                    "actions": result_actions,
                    "inference_time_s": inference_time,
                    "chunk_size": len(result_actions),
                    "model": self.config.pretrained_checkpoint,
                })

            except Exception as e:
                logger.error(f"Error in /act: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self._app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "ok" if self.vla is not None else "model_not_loaded",
                "model": self.config.pretrained_checkpoint,
                "device": self.config.device,
                "warm": self._warm,
            }

        @self._app.get("/model_info")
        async def model_info():
            """Get model configuration info."""
            return {
                "checkpoint": self.config.pretrained_checkpoint,
                "l1_regression": self.config.use_l1_regression,
                "diffusion": self.config.use_diffusion,
                "film": self.config.use_film,
                "num_images": self.config.num_images_in_input,
                "use_proprio": self.config.use_proprio,
                "chunk_size": self.config.num_open_loop_steps,
                "unnorm_key": self.config.unnorm_key,
                "device": self.config.device,
                "model_loaded": self.vla is not None,
            }

        @self._app.post("/warmup")
        async def warmup():
            """Trigger model warmup."""
            self.warmup()
            return {"status": "warmup_complete"}

        return self._app

    def run(self, host: Optional[str] = None, port: Optional[int] = None):
        """Start the VLA server."""
        if self.vla is None:
            self.load_model()

        app = self.create_app()

        host = host or self.config.host
        port = port or self.config.port

        logger.info(f"Starting VLA server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)


# ── Standalone Server Script ─────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenVLA-OFT Inference Server")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--openvla-oft-root", type=str, default=None,
                        help="Path to openvla-oft repo root (if not pip-installed)")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    # Load config
    import yaml

    config_dict = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            full_config = yaml.safe_load(f)
            config_dict = full_config.get("vla_server", {})

    # Override with CLI args
    if args.host:
        config_dict["host"] = args.host
    if args.port:
        config_dict["port"] = args.port
    if args.checkpoint:
        config_dict["pretrained_checkpoint"] = args.checkpoint
    if args.openvla_oft_root:
        config_dict["openvla_oft_root"] = args.openvla_oft_root

    # Create and start server
    server = VLAServer(config_dict=config_dict)
    server.load_model()

    if not args.no_warmup:
        server.warmup()

    server.run()