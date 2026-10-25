"""Append incoming serial IMU frames to an HDF5 recording."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from common.Const_INFO import IMU_ORDER

class LiveIMURecorder:
    def __init__(self, project_folder: Path, sample_rate: float = 100.0, recording_folder: Path | None = None) -> None:
        recording_folder = recording_folder or (project_folder / "recordings")
        recording_folder.mkdir(parents=True, exist_ok=True)
        self.path = recording_folder / f"Live_IMU_{datetime.now():%Y%m%d_%H%M%S}.h5"
        self.file = h5py.File(self.path, "w")
        meta = self.file.create_group("meta")
        meta.attrs["sample_rate_hz"] = sample_rate
        meta.attrs["timestep"] = 1.0 / sample_rate
        meta.create_dataset("imu_order", data=np.asarray(IMU_ORDER, dtype="S"))
        trajectory = self.file.create_group("trajectory")
        self.datasets = {
            "sequence": trajectory.create_dataset("sequence", (0,), maxshape=(None,), dtype="u4", chunks=True),
            "timestamp_us": trajectory.create_dataset("timestamp_us", (0,), maxshape=(None,), dtype="u8", chunks=True),
            "group_scans": trajectory.create_dataset("group_scans", (0, 2), maxshape=(None, 2), dtype="u4", chunks=True),
            "status": trajectory.create_dataset("status", (0, len(IMU_ORDER), 3), maxshape=(None, len(IMU_ORDER), 3), dtype="?", chunks=True),
            "accel_raw": trajectory.create_dataset("accel_raw", (0, len(IMU_ORDER), 3), maxshape=(None, len(IMU_ORDER), 3), dtype="i2", chunks=True),
            "gyro_raw": trajectory.create_dataset("gyro_raw", (0, len(IMU_ORDER), 3), maxshape=(None, len(IMU_ORDER), 3), dtype="i2", chunks=True),
            "accel": trajectory.create_dataset("accel", (0, len(IMU_ORDER), 3), maxshape=(None, len(IMU_ORDER), 3), dtype="f4", chunks=True),
            "gyro": trajectory.create_dataset("gyro", (0, len(IMU_ORDER), 3), maxshape=(None, len(IMU_ORDER), 3), dtype="f4", chunks=True),
            "diagnostic_code": trajectory.create_dataset("diagnostic_code", (0, len(IMU_ORDER)), maxshape=(None, len(IMU_ORDER)), dtype="u1", chunks=True),
            "diagnostic_registers": trajectory.create_dataset("diagnostic_registers", (0, len(IMU_ORDER), 4), maxshape=(None, len(IMU_ORDER), 4), dtype="u1", chunks=True),
            "stale_scans": trajectory.create_dataset("stale_scans", (0, len(IMU_ORDER)), maxshape=(None, len(IMU_ORDER)), dtype="u2", chunks=True),
        }
        self.datasets["diagnostic_code"].attrs["legend"] = "0=good, 1=no_data_ready, 2=who_am_i_failed, 3=configuration_changed, 4=spi_read_failed, 5=reinitialisation_failed"
        self.datasets["diagnostic_registers"].attrs["columns"] = np.asarray(["WHO_AM_I", "CTRL1_XL", "CTRL2_G", "STATUS_REG"], dtype="S")
        self.datasets["stale_scans"].attrs["description"] = "Consecutive 100 Hz firmware scans without both fresh accelerometer and gyroscope data"
        self.count = 0

    def append(self, frame, channel_to_site: dict[int, str]) -> None:
        index_by_site = {site: index for index, site in enumerate(IMU_ORDER)}
        status = np.zeros((len(IMU_ORDER), 3), dtype=bool)
        accel_raw = np.zeros((len(IMU_ORDER), 3), dtype=np.int16)
        gyro_raw = np.zeros((len(IMU_ORDER), 3), dtype=np.int16)
        accel = np.zeros((len(IMU_ORDER), 3), dtype=np.float32)
        gyro = np.zeros((len(IMU_ORDER), 3), dtype=np.float32)
        diagnostic_code = np.zeros(len(IMU_ORDER), dtype=np.uint8)
        diagnostic_registers = np.zeros((len(IMU_ORDER), 4), dtype=np.uint8)
        stale_scans = np.zeros(len(IMU_ORDER), dtype=np.uint16)
        for channel_id, record in frame.records.items():
            index = index_by_site[channel_to_site[channel_id]]
            status[index] = (record.available, record.acc_fresh, record.gyro_fresh)
            accel_raw[index] = record.acc_raw
            gyro_raw[index] = record.gyro_raw
            accel[index] = record.acc_m_s2
            gyro[index] = record.gyro_rad_s
            diagnostic_code[index] = record.diagnostic_code
            diagnostic_registers[index] = (record.who_am_i, record.ctrl1_xl, record.ctrl2_g, record.status_register)
            stale_scans[index] = record.stale_scans
        row = self.count
        for dataset in self.datasets.values(): dataset.resize(row + 1, axis=0)
        self.datasets["sequence"][row] = frame.sequence
        self.datasets["timestamp_us"][row] = frame.timestamp_us
        self.datasets["group_scans"][row] = (frame.group_a_scan, frame.group_b_scan)
        self.datasets["status"][row] = status
        self.datasets["accel_raw"][row] = accel_raw
        self.datasets["gyro_raw"][row] = gyro_raw
        self.datasets["accel"][row] = accel
        self.datasets["gyro"][row] = gyro
        self.datasets["diagnostic_code"][row] = diagnostic_code
        self.datasets["diagnostic_registers"][row] = diagnostic_registers
        self.datasets["stale_scans"][row] = stale_scans
        self.count += 1
        if self.count % 100 == 0: self.file.flush()

    def close(self) -> None:
        self.file.flush()
        self.file.close()



