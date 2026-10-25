"""Shared final qpos display path for calibration preview and live tracking."""
from __future__ import annotations

import numpy as np
from live_tracking.live_common.Live_IMU_Display_Controls import ABDUCTION_QPOS


def apply_display_qpos(qpos: np.ndarray, joint_ranges: np.ndarray, axis_controls, display_controls=None) -> np.ndarray:
    clipped = np.asarray(qpos, dtype=float).copy()
    for index in range(3, min(len(clipped), len(joint_ranges))):
        clipped[index] = np.clip(clipped[index], joint_ranges[index, 0], joint_ranges[index, 1])
    if display_controls is not None:
        display_controls.refresh()
        if display_controls.disable_abduction:
            clipped[list(ABDUCTION_QPOS)] = 0.0
        elif getattr(display_controls, "disable_middle_abduction", False):
            clipped[13] = 0.0
    return axis_controls.apply_palm_wrist_qpos(clipped)
