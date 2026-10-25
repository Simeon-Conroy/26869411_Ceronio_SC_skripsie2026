"""Persistent stationary calibration for the live glove."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from common.Const_INFO import IMU_ORDER, NUM_IMUS, Site_Initial_quat
from common.Quat_Operation import inv_conj_quat, quat_Mult, quat_Normalise

CALIBRATION_VERSION = 2
GYRO_STATIONARY_LIMIT_RAD_S = np.deg2rad(4.0)
ACCEL_CHANGE_LIMIT_M_S2 = 0.75
FILTER_SETTLE_FRAMES = 200

@dataclass
class CalibrationData:
    gyro_bias: np.ndarray
    gyro_std: np.ndarray
    mean_accel: np.ndarray
    gravity_magnitude: np.ndarray
    rejected_samples: np.ndarray
    enabled_imus: list[str]
    sample_count: int
    created_at: str
    guided_imu: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        payload = {
            "version": CALIBRATION_VERSION, "created_at": self.created_at, "sample_count": self.sample_count, "imu_order": IMU_ORDER, "enabled_imus": self.enabled_imus,
            "gyro_bias_rad_s": self.gyro_bias.tolist(), "gyro_std_rad_s": self.gyro_std.tolist(), "mean_accel_m_s2": self.mean_accel.tolist(),
            "gravity_magnitude_m_s2": self.gravity_magnitude.tolist(), "rejected_samples": self.rejected_samples.tolist(), "guided_imu": self.guided_imu
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CalibrationData":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") not in (1, CALIBRATION_VERSION): raise ValueError("unsupported calibration version")
        if payload.get("imu_order") != IMU_ORDER: raise ValueError("calibration IMU order does not match this project")
        return cls(np.asarray(payload["gyro_bias_rad_s"], dtype=float), np.asarray(payload["gyro_std_rad_s"], dtype=float), np.asarray(payload["mean_accel_m_s2"], dtype=float), np.asarray(payload["gravity_magnitude_m_s2"], dtype=float), np.asarray(payload["rejected_samples"], dtype=int), list(payload["enabled_imus"]), int(payload["sample_count"]), str(payload["created_at"]), dict(payload.get("guided_imu", {})))


class StationaryCalibration:
    def __init__(self, required_frames: int, enabled: np.ndarray) -> None:
        self.required_frames = required_frames
        self.enabled = enabled
        self.enabled_names = [site for index, site in enumerate(IMU_ORDER) if enabled[index]]
        self.gyro_samples: list[list[np.ndarray]] = [[] for _ in range(NUM_IMUS)]
        self.accel_samples: list[list[np.ndarray]] = [[] for _ in range(NUM_IMUS)]
        self.previous_accel: list[np.ndarray | None] = [None] * NUM_IMUS
        self.motion_rejections = np.zeros(NUM_IMUS, dtype=int)
        self.last_moving_imu = ""
        self.last_motion_detail = ""
        self.last_action = "accepted"
        self.gyro_limit_rad_s = GYRO_STATIONARY_LIMIT_RAD_S
        self.accel_change_limit_m_s2 = ACCEL_CHANGE_LIMIT_M_S2

    @property
    def count(self) -> int:
        counts = [len(self.gyro_samples[index]) for index in range(NUM_IMUS) if self.enabled[index]]
        return min(counts) if counts else 0

    def add(self, accel: np.ndarray, gyro: np.ndarray) -> bool:
        if not np.any(self.enabled): raise ValueError("no IMUs are enabled")
        accepted_any, rejected_details = False, []
        for index, site in enumerate(IMU_ORDER):
            if not self.enabled[index] or len(self.gyro_samples[index]) >= self.required_frames: continue
            history = self.gyro_samples[index]
            baseline = np.median(np.asarray(history[-100:]), axis=0) if history else gyro[index]
            gyro_change = float(np.linalg.norm(gyro[index] - baseline)) if history else float(np.linalg.norm(gyro[index]))
            accel_change = 0.0 if self.previous_accel[index] is None else float(np.linalg.norm(accel[index] - self.previous_accel[index]))
            self.previous_accel[index] = accel[index].copy()
            if gyro_change > self.gyro_limit_rad_s or accel_change > self.accel_change_limit_m_s2:
                self.motion_rejections[index] += 1
                detail = f"gyro {np.rad2deg(gyro_change):.1f} dps" if gyro_change > self.gyro_limit_rad_s else f"acceleration {accel_change:.2f} m/s^2"
                rejected_details.append((max(gyro_change / max(self.gyro_limit_rad_s, 1e-9), accel_change / max(self.accel_change_limit_m_s2, 1e-9)), site, detail))
                continue
            self.gyro_samples[index].append(gyro[index].copy())
            self.accel_samples[index].append(accel[index].copy())
            accepted_any = True
        if rejected_details:
            _, self.last_moving_imu, self.last_motion_detail = max(rejected_details)
            self.last_action = "partial" if accepted_any else "paused"
        else: self.last_action = "accepted"
        return accepted_any

    def complete(self) -> bool: return self.count >= self.required_frames

    def enabled_counts(self) -> np.ndarray:
        return np.asarray([len(self.gyro_samples[index]) if self.enabled[index] else self.required_frames for index in range(NUM_IMUS)], dtype=int)

    def calculate(self) -> CalibrationData:
        gyro_bias = np.zeros((NUM_IMUS, 3))
        gyro_std = np.zeros((NUM_IMUS, 3))
        mean_accel = np.zeros((NUM_IMUS, 3))
        gravity_magnitude = np.zeros(NUM_IMUS)
        rejected = np.zeros(NUM_IMUS, dtype=int)
        for index in range(NUM_IMUS):
            if not self.enabled[index]: continue
            gyro = np.asarray(self.gyro_samples[index])
            accel = np.asarray(self.accel_samples[index])
            median_gyro = np.median(gyro, axis=0)
            gyro_residual = np.linalg.norm(gyro - median_gyro, axis=1)
            gyro_residual_median = float(np.median(gyro_residual))
            gyro_mad = float(np.median(np.abs(gyro_residual - gyro_residual_median)))
            gyro_limit = max(np.deg2rad(0.10), gyro_residual_median + 6.0 * 1.4826 * gyro_mad)
            gyro_accepted = gyro_residual <= gyro_limit
            if not np.any(gyro_accepted): gyro_accepted[:] = True
            gyro_bias[index] = np.median(gyro[gyro_accepted], axis=0)
            gyro_std[index] = np.std(gyro[gyro_accepted], axis=0)
            median_accel = np.median(accel, axis=0)
            accel_residual = np.linalg.norm(accel - median_accel, axis=1)
            accel_residual_median = float(np.median(accel_residual))
            accel_mad = float(np.median(np.abs(accel_residual - accel_residual_median)))
            accel_limit = max(0.05, accel_residual_median + 6.0 * 1.4826 * accel_mad)
            accel_accepted = accel_residual <= accel_limit
            if not np.any(accel_accepted): accel_accepted[:] = True
            mean_accel[index] = np.mean(accel[accel_accepted], axis=0)
            gravity_magnitude[index] = np.linalg.norm(mean_accel[index])
            rejected[index] = self.motion_rejections[index] + len(gyro_accepted) - int(np.count_nonzero(gyro_accepted)) + len(accel_accepted) - int(np.count_nonzero(accel_accepted))
        return CalibrationData(gyro_bias, gyro_std, mean_accel, gravity_magnitude, rejected, self.enabled_names, self.count, datetime.now().isoformat(timespec="seconds"), {"stationary_mode": "strict_no_fallback", "gyro_stationary_limit_dps": float(np.rad2deg(self.gyro_limit_rad_s)), "accel_change_limit_m_s2": float(self.accel_change_limit_m_s2)})


class OnlineGyroBias:
    """Slowly follows temperature drift only while each IMU is independently stationary."""
    def __init__(self, calibration: CalibrationData, sample_rate_hz: float = 100.0) -> None:
        self.bias = calibration.gyro_bias.copy()
        self.gravity = calibration.gravity_magnitude.copy()
        self.enabled = np.asarray([site in calibration.enabled_imus for site in IMU_ORDER], dtype=bool)
        self.stable = np.zeros(NUM_IMUS, dtype=int)
        self.required = max(1, int(round(2.0 * sample_rate_hz)))
        self.alpha = 0.002

    def correct(self, accel: np.ndarray, raw_gyro: np.ndarray) -> np.ndarray:
        corrected = raw_gyro - self.bias
        for index in range(NUM_IMUS):
            if not self.enabled[index]: continue
            gyro_still = np.linalg.norm(corrected[index]) < np.deg2rad(1.5)
            accel_still = abs(np.linalg.norm(accel[index]) - self.gravity[index]) < 0.35
            self.stable[index] = self.stable[index] + 1 if gyro_still and accel_still else 0
            if self.stable[index] >= self.required:
                self.bias[index] += self.alpha * (raw_gyro[index] - self.bias[index])
                corrected[index] = raw_gyro[index] - self.bias[index]
        return corrected


def reference_corrections(filter_quaternions: np.ndarray) -> np.ndarray:
    corrections = np.zeros((NUM_IMUS, 4))
    for index, site in enumerate(IMU_ORDER): corrections[index] = quat_Normalise(quat_Mult(Site_Initial_quat[site], inv_conj_quat(filter_quaternions[index])))
    return corrections


def apply_reference(filter_quaternions: np.ndarray, corrections: np.ndarray) -> np.ndarray:
    output = np.zeros_like(filter_quaternions)
    for index in range(NUM_IMUS): output[index] = quat_Normalise(quat_Mult(corrections[index], filter_quaternions[index]))
    return output


def quaternion_mean(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=float)
    values = values / np.linalg.norm(values, axis=1, keepdims=True)
    matrix = values.T @ values
    mean = np.linalg.eigh(matrix)[1][:, -1]
    if mean[0] < 0.0: mean = -mean
    return quat_Normalise(mean)


def quaternion_angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(2.0 * np.arccos(np.clip(abs(np.dot(quat_Normalise(a), quat_Normalise(b))), 0.0, 1.0)))


def apply_sensor_alignment(filter_quaternions: np.ndarray, alignment: dict) -> np.ndarray:
    corrections = np.asarray(alignment["quaternions"], dtype=float)
    sides = alignment.get("sides", ["left"] * NUM_IMUS)
    output = np.zeros_like(filter_quaternions, dtype=float)
    for index in range(NUM_IMUS):
        output[index] = quat_Normalise(quat_Mult(filter_quaternions[index], corrections[index]) if sides[index] == "right" else quat_Mult(corrections[index], filter_quaternions[index]))
    return output


def fit_sensor_alignment(target_quaternions: np.ndarray, live_quaternions: np.ndarray, enabled_imus: list[str]) -> dict:
    targets, live = np.asarray(target_quaternions, dtype=float), np.asarray(live_quaternions, dtype=float)
    if targets.shape != live.shape or targets.ndim != 3 or targets.shape[1:] != (NUM_IMUS, 4): raise ValueError("Pose quaternion arrays must both have shape (poses, IMUs, 4).")
    enabled = set(enabled_imus)
    corrections, sides, details = np.zeros((NUM_IMUS, 4)), [], {}
    corrections[:, 0] = 1.0
    for index, site in enumerate(IMU_ORDER):
        if site not in enabled:
            sides.append("left")
            details[site] = {"used_poses": 0, "rejected_poses": len(targets), "rms_error_deg": 0.0, "side": "left"}
            continue
        best = None
        for side in ("left", "right"):
            candidates = []
            for target, measured in zip(targets[:, index], live[:, index]):
                candidate = quat_Mult(target, inv_conj_quat(measured)) if side == "left" else quat_Mult(inv_conj_quat(measured), target)
                candidates.append(quat_Normalise(candidate))
            candidates = np.asarray(candidates)
            accepted = np.ones(len(candidates), dtype=bool)
            initial = quaternion_mean(candidates)
            deviations = np.asarray([quaternion_angle(value, initial) for value in candidates])
            scale = np.deg2rad(12.0)
            weights = 0.10 + 0.90 / (1.0 + (deviations / max(scale, 1e-9)) ** 2)
            signed = np.asarray([value if np.dot(value, initial) >= 0.0 else -value for value in candidates])
            correction = quat_Normalise(np.sum(signed * weights[:, None], axis=0))
            errors = []
            for target, measured in zip(targets[:, index], live[:, index]):
                predicted = quat_Mult(measured, correction) if side == "right" else quat_Mult(correction, measured)
                errors.append(quaternion_angle(target, predicted))
            rms = float(np.sqrt(np.mean(np.square(errors))))
            result = (rms, correction, accepted)
            if best is None or result[0] < best[0]: best = result[0], result[1], result[2], side
        rms, correction, accepted, side = best
        corrections[index], sides = correction, [*sides, side]
        details[site] = {"used_poses": int(np.count_nonzero(accepted)), "rejected_poses": int(len(accepted) - np.count_nonzero(accepted)), "rms_error_deg": float(np.rad2deg(rms)), "side": side}
    return {"quaternions": corrections.tolist(), "sides": sides, "details": details, "method": "robust_multi_pose_sensor_to_segment"}




