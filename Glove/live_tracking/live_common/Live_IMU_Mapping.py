"""Stable MainBoard channel mapping and measured physical-to-model axes."""

from __future__ import annotations

import numpy as np

from common.Const_INFO import IMU_ORDER, NUM_IMUS, Site_Index

CHANNEL_TO_SITE = {
    0: "imu_middle_dist", 1: "imu_middle_mid", 2: "imu_ring_dist", 3: "imu_ring_mid", 4: "imu_ring_prox", 5: "imu_pinky_dist", 6: "imu_pinky_mid", 7: "imu_pinky_prox",
    8: "imu_thumb_dist", 9: "imu_thumb_prox", 10: "imu_index_dist", 11: "imu_index_mid", 12: "imu_index_prox", 13: "imu_middle_prox", 14: "imu_palm", 15: "imu_wrist", 16: "imu_thumb_cmc"
}

PHYSICAL_TO_MODEL_AXES = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def frame_to_arrays(frame, disabled_imus: set[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    disabled = disabled_imus or set()
    accel = np.zeros((NUM_IMUS, 3))
    gyro = np.zeros((NUM_IMUS, 3))
    for channel_id, record in frame.records.items():
        site = CHANNEL_TO_SITE.get(channel_id)
        if site is None or site in disabled: continue
        index = Site_Index[site]
        accel[index] = PHYSICAL_TO_MODEL_AXES @ np.asarray(record.acc_m_s2)
        gyro[index] = PHYSICAL_TO_MODEL_AXES @ np.asarray(record.gyro_rad_s)
    return accel, gyro


def enabled_mask(disabled_imus: set[str]) -> np.ndarray: return np.asarray([site not in disabled_imus for site in IMU_ORDER], dtype=bool)


def invalid_enabled_sites(frame, disabled_imus: set[str]) -> list[str]:
    invalid = []
    for channel_id, record in frame.records.items():
        site = CHANNEL_TO_SITE.get(channel_id)
        if site is None: continue
        if site not in disabled_imus and not record.available: invalid.append(site)
    return invalid
