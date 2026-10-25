"""Persistent enable/disable controls for the six Wrist and Palm model axes."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

AXIS_LABELS = {
    "wrist_world_roll": "Wrist → World roll",
    "wrist_world_pitch": "Wrist → World pitch",
    "wrist_world_yaw": "Wrist → World yaw",
    "palm_wrist_roll": "Palm → Wrist roll",
    "palm_wrist_pitch": "Palm → Wrist pitch",
    "palm_wrist_yaw": "Palm → Wrist yaw"
}
AXIS_DEFAULTS = {name: True for name in AXIS_LABELS}
PALM_QPOS = {"palm_wrist_roll": 0, "palm_wrist_pitch": 1, "palm_wrist_yaw": 2}
WRIST_EULER = {"wrist_world_roll": 0, "wrist_world_pitch": 1, "wrist_world_yaw": 2}


class AxisControls:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.values = AXIS_DEFAULTS.copy()
        self.last_read = 0.0
        self.refresh(force=True)

    def refresh(self, force: bool = False) -> dict[str, bool]:
        now = time.monotonic()
        if not force and now - self.last_read < 0.1: return self.values
        self.last_read = now
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            for name in AXIS_DEFAULTS:
                if name in saved: self.values[name] = bool(saved[name])
        except (FileNotFoundError, OSError, ValueError, TypeError): pass
        return self.values

    def apply_palm_wrist_qpos(self, qpos: np.ndarray) -> np.ndarray:
        values = self.refresh()
        output = np.asarray(qpos, dtype=float).copy()
        for name, index in PALM_QPOS.items():
            if not values[name]: output[index] = 0.0
        return output

    def apply_wrist_world_qpos(self, model_qpos: np.ndarray) -> np.ndarray:
        values = self.refresh()
        output = np.asarray(model_qpos, dtype=float).copy()
        for name, index in WRIST_EULER.items():
            if not values[name]: output[index] = 0.0
        return output
