#!/usr/bin/env python3
"""
Rich TUI Client for Isaac-VLA
==============================

A terminal user interface built with Textual for interactive control
of the Franka robot via OpenVLA-OFT. Features:

- Live camera feed display (third-person + wrist)
- Joint state and gripper status panel
- Natural language task input
- Real-time action log
- Episode management (start/stop/reset)
- Keyboard teleoperation fallback

Usage:
    python scripts/run_tui_client.py
    python scripts/run_tui_client.py --bridge-url http://localhost:8889
    python scripts/run_tui_client.py --embedded

Requirements:
    pip install textual rich
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Resolve project root from this script's absolute location
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Add project root to path (must be absolute since python.sh may change cwd)
sys.path.insert(0, str(_PROJECT_ROOT))

# Also set working directory to project root so config files resolve correctly
os.chdir(str(_PROJECT_ROOT))

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Log,
    Static,
    TabbedContent,
    TabPane,
)
from textual import work

from src.utils import load_config


class CameraPanel(Static):
    """Panel showing camera feed (placeholder for actual image display)."""

    camera_label = reactive("No image")

    def compose(self) -> ComposeResult:
        yield Label("📷 Camera Feed", id="camera-title")
        yield Static(self.camera_label, id="camera-display")

    def update_image(self, label: str):
        self.camera_label = label


class RobotStatePanel(Static):
    """Panel showing robot state information."""

    joint_positions = reactive("—")
    gripper_state = reactive("—")
    ee_position = reactive("—")
    step_count = reactive(0)

    def compose(self) -> ComposeResult:
        yield Label("🤖 Robot State", id="state-title")
        yield Label(f"Joints: {self.joint_positions}", id="joints-label")
        yield Label(f"Gripper: {self.gripper_state}", id="gripper-label")
        yield Label(f"EE Pos: {self.ee_position}", id="ee-label")
        yield Label(f"Step: {self.step_count}", id="step-label")

    def update_state(self, state: dict):
        if "joint_positions" in state:
            jp = state["joint_positions"]
            self.joint_positions = f"[{', '.join(f'{v:.2f}' for v in jp[:3])}, ...]"
        if "gripper" in state:
            self.gripper_state = f"{state['gripper']:.3f}m"
        if "ee_position" in state:
            ep = state["ee_position"]
            self.ee_position = f"({ep[0]:.3f}, {ep[1]:.3f}, {ep[2]:.3f})"
        if "step" in state:
            self.step_count = state["step"]


class ActionLogPanel(Static):
    """Panel showing real-time action log."""

    def compose(self) -> ComposeResult:
        yield Label("📋 Action Log", id="log-title")
        yield Log(id="action-log", max_lines=100)

    def log_action(self, message: str):
        try:
            log_widget = self.query_one("#action-log", Log)
            log_widget.write_line(message)
        except Exception:
            pass


class TaskInputPanel(Static):
    """Panel for natural language task input."""

    def compose(self) -> ComposeResult:
        yield Label("💬 Task Instruction", id="task-title")
        yield Input(placeholder="Type a task instruction...", id="task-input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle task instruction submission."""
        instruction = event.value.strip()
        if instruction:
            self.app.run_task(instruction)
            event.input.value = ""


class ControlPanel(Static):
    """Panel with control buttons."""

    def compose(self) -> ComposeResult:
        yield Label("🎮 Controls", id="controls-title")
        with Horizontal():
            yield Button("▶ Run", id="run-btn", variant="success")
            yield Button("⏸ Pause", id="pause-btn", variant="warning")
            yield Button("🔄 Reset", id="reset-btn", variant="primary")
            yield Button("⏹ Stop", id="stop-btn", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "run-btn":
            self.app.resume_task()
        elif btn_id == "pause-btn":
            self.app.pause_task()
        elif btn_id == "reset-btn":
            self.app.reset_scene()
        elif btn_id == "stop-btn":
            self.app.stop_task()


class IsaacVLAApp(App):
    """Rich TUI for Isaac-VLA robot control."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 3 1;
        grid-columns: 1fr 2fr 1fr;
    }

    #left-panel {
        width: 100%;
        height: 100%;
    }

    #center-panel {
        width: 100%;
        height: 100%;
    }

    #right-panel {
        width: 100%;
        height: 100%;
    }

    .panel-title {
        text-style: bold;
        color: $text;
        background: $primary;
        padding: 1;
    }

    Log {
        height: 20;
        border: solid $primary;
    }

    Input {
        margin: 1 0;
    }

    Button {
        margin: 0 1;
    }

    #status-bar {
        dock: bottom;
        height: 3;
        background: $primary;
        color: $text;
        padding: 0 1;
    }
    """

    TITLE = "Isaac-VLA Control"
    SUB_TITLE = "OpenVLA-OFT on Franka Emika in Isaac Sim"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "reset", "Reset"),
        ("d", "toggle_dark", "Toggle dark mode"),
    ]

    # Reactive state
    status = reactive("Ready")
    current_task = reactive("None")
    episode_count = reactive(0)

    def __init__(self, bridge_url: str = "http://localhost:8889",
                 embedded: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.bridge_url = bridge_url
        self.embedded = embedded
        self._client = None
        self._running = False
        self._paused = False

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal():
            # Left panel: Robot state + Controls
            with Vertical(id="left-panel"):
                yield RobotStatePanel(id="robot-state")
                yield ControlPanel(id="controls")
                yield TaskInputPanel(id="task-input")

            # Center panel: Camera + Action log
            with Vertical(id="center-panel"):
                yield CameraPanel(id="camera-panel")
                yield ActionLogPanel(id="action-log")

            # Right panel: Task info + VLA status
            with Vertical(id="right-panel"):
                yield Static("📊 VLA Status", classes="panel-title")
                yield Label(f"Status: {self.status}", id="vla-status")
                yield Label(f"Task: {self.current_task}", id="task-label")
                yield Label(f"Episode: {self.episode_count}", id="episode-label")
                yield Static("", id="vla-info")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app after mounting."""
        self._initialize_client()

    @work(exclusive=True)
    async def _initialize_client(self):
        """Initialize the VLA client connection."""
        try:
            if self.embedded:
                from src.api import IsaacVLAClient
                self._client = IsaacVLAClient(mode="embedded")
                self._client.initialize()
                self.status = "Connected (embedded)"
            else:
                from src.api import RemoteVLAClient
                self._client = RemoteVLAClient(self.bridge_url)
                # Try to connect
                try:
                    status = self._client.get_status()
                    self.status = f"Connected to {self.bridge_url}"
                except Exception:
                    self.status = f"Connecting to {self.bridge_url}..."

            self._log_action("Client initialized")
        except Exception as e:
            self.status = f"Error: {e}"
            self._log_action(f"Initialization error: {e}")

    def run_task(self, instruction: str):
        """Run a task with the given instruction."""
        self.current_task = instruction
        self._log_action(f"Starting task: '{instruction}'")
        self.status = "Running"

        if self._client:
            try:
                if self.embedded:
                    result = self._client.run_task(instruction)
                    self._log_action(f"Task result: {result}")
                else:
                    result = self._client.run_task(instruction)
                    self._log_action(f"Task result: {result}")
            except Exception as e:
                self._log_action(f"Task error: {e}")
                self.status = f"Error: {e}"

    def pause_task(self):
        """Pause the current task."""
        self._paused = True
        self.status = "Paused"
        self._log_action("Task paused")

    def resume_task(self):
        """Resume the paused task."""
        self._paused = False
        self.status = "Running"
        self._log_action("Task resumed")

    def reset_scene(self):
        """Reset the simulation scene."""
        if self._client:
            try:
                self._client.reset()
                self._log_action("Scene reset")
                self.status = "Reset"
            except Exception as e:
                self._log_action(f"Reset error: {e}")

    def stop_task(self):
        """Stop the current task."""
        self._running = False
        self.status = "Stopped"
        self._log_action("Task stopped")

    def _log_action(self, message: str):
        """Log a message to the action log panel."""
        try:
            log_panel = self.query_one("#action-log", ActionLogPanel)
            log_panel.log_action(f"[{time.strftime('%H:%M:%S')}] {message}")
        except Exception:
            pass

    def action_quit(self) -> None:
        """Quit the application."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        self.exit()


def main():
    parser = argparse.ArgumentParser(description="Isaac-VLA Rich TUI Client")
    parser.add_argument("--bridge-url", type=str, default="http://localhost:8889",
                        help="Sim bridge server URL")
    parser.add_argument("--embedded", action="store_true",
                        help="Use embedded mode (no HTTP)")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Config file path")
    args = parser.parse_args()

    app = IsaacVLAApp(bridge_url=args.bridge_url, embedded=args.embedded)
    app.run()


if __name__ == "__main__":
    main()