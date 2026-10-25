"""Shared definitions and decoder for the MainBoard serial packet protocol."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

MAGIC = b"IMU1"
VERSION = 3
RECORD_COUNT = 17
PACKET_SIZE = 372
HEADER = struct.Struct(">4sBBHIQII")
RECORD = struct.Struct(">BBhhhhhhBBBBH")
CRC = struct.Struct(">I")
ACC_G_PER_LSB = 0.000122
GYRO_DPS_PER_LSB = 0.070


class PacketError(ValueError): pass


@dataclass(frozen=True)
class LiveIMURecord:
    channel_id: int
    available: bool
    acc_fresh: bool
    gyro_fresh: bool
    diagnostic_code: int
    who_am_i: int
    ctrl1_xl: int
    ctrl2_g: int
    status_register: int
    stale_scans: int
    acc_raw: tuple[int, int, int]
    gyro_raw: tuple[int, int, int]

    @property
    def acc_m_s2(self) -> tuple[float, float, float]: return tuple(value * ACC_G_PER_LSB * 9.80665 for value in self.acc_raw)

    @property
    def gyro_rad_s(self) -> tuple[float, float, float]: return tuple(value * GYRO_DPS_PER_LSB * 0.017453292519943295 for value in self.gyro_raw)


@dataclass(frozen=True)
class LiveIMUFrame:
    sequence: int
    timestamp_us: int
    group_a_scan: int
    group_b_scan: int
    records: dict[int, LiveIMURecord]



def decode_packet(packet: bytes) -> LiveIMUFrame:
    if len(packet) < HEADER.size + CRC.size: raise PacketError("packet too short")
    magic, version, count, declared_size, sequence, timestamp_us, scan_a, scan_b = HEADER.unpack_from(packet)
    if magic != MAGIC: raise PacketError("invalid packet magic")
    if version != VERSION: raise PacketError(f"unsupported protocol version {version}")
    if count != RECORD_COUNT or declared_size != PACKET_SIZE or len(packet) != PACKET_SIZE: raise PacketError("invalid record count or packet size")
    received_crc = CRC.unpack_from(packet, declared_size - CRC.size)[0]
    calculated_crc = zlib.crc32(packet[:-CRC.size]) & 0xFFFFFFFF
    if received_crc != calculated_crc: raise PacketError("CRC-32 mismatch")
    records = {}
    for index in range(count):
        channel_id, status, ax, ay, az, gx, gy, gz, who_am_i, ctrl1_xl, ctrl2_g, status_register, stale_scans = RECORD.unpack_from(packet, HEADER.size + index * RECORD.size)
        if channel_id not in range(RECORD_COUNT): raise PacketError(f"unknown channel ID {channel_id}")
        if channel_id in records: raise PacketError(f"duplicate channel ID {channel_id}")
        records[channel_id] = LiveIMURecord(channel_id, bool(status & 0x01), bool(status & 0x02), bool(status & 0x04), (status >> 3) & 0x07, who_am_i, ctrl1_xl, ctrl2_g, status_register, stale_scans, (ax, ay, az), (gx, gy, gz))
    if len(records) != RECORD_COUNT: raise PacketError(f"packet does not contain all {RECORD_COUNT} channels")
    return LiveIMUFrame(sequence, timestamp_us, scan_a, scan_b, records)
