"""Runtime display-mode controls shared by the dashboard, calibration preview and live tracking."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_DISPLAY_CONTROLS = {"raw_mode": False, "disable_abduction": False, "disable_middle_abduction": False}
PALM_WRIST_QPOS = {"palm_wrist_roll": 0, "palm_wrist_pitch": 1, "palm_wrist_yaw": 2}
ABDUCTION_QPOS = (4, 6, 9, 13, 17, 21)

def _truth(value) -> bool:
    return bool(value.get()) if hasattr(value, "get") else bool(value)

def locked_qpos_indices(axis_values=None, disable_abduction: bool = False) -> list[int]:
    locked = []
    if axis_values:
        for name, index in PALM_WRIST_QPOS.items():
            if name in axis_values and not _truth(axis_values[name]): locked.append(index)
    if disable_abduction: locked.extend(ABDUCTION_QPOS)
    return sorted(set(locked))

class DisplayControls:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.raw_mode = DEFAULT_DISPLAY_CONTROLS["raw_mode"]
        self.disable_abduction = DEFAULT_DISPLAY_CONTROLS["disable_abduction"]
        self.disable_middle_abduction = DEFAULT_DISPLAY_CONTROLS["disable_middle_abduction"]
        self._mtime = None
        self.refresh(force=True)

    def refresh(self, force: bool = False) -> None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self.raw_mode = DEFAULT_DISPLAY_CONTROLS["raw_mode"]
            self.disable_abduction = DEFAULT_DISPLAY_CONTROLS["disable_abduction"]
            self.disable_middle_abduction = DEFAULT_DISPLAY_CONTROLS["disable_middle_abduction"]
            self._mtime = None
            return
        if not force and self._mtime == stat.st_mtime:
            return
        self._mtime = stat.st_mtime
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        self.raw_mode = bool(data.get("raw_mode", DEFAULT_DISPLAY_CONTROLS["raw_mode"]))
        self.disable_abduction = bool(data.get("disable_abduction", DEFAULT_DISPLAY_CONTROLS["disable_abduction"]))
        self.disable_middle_abduction = bool(data.get("disable_middle_abduction", DEFAULT_DISPLAY_CONTROLS["disable_middle_abduction"]))

def read_display_controls(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        data = {}
    return {name: bool(data.get(name, default)) for name, default in DEFAULT_DISPLAY_CONTROLS.items()}

def save_display_controls(path: Path, raw_mode: bool, disable_abduction: bool, disable_middle_abduction: bool = False) -> None:
    Path(path).write_text(json.dumps({"raw_mode": bool(raw_mode), "disable_abduction": bool(disable_abduction), "disable_middle_abduction": bool(disable_middle_abduction)}, indent=2))
