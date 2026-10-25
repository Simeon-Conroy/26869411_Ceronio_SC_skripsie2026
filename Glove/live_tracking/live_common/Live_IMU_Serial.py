"""Read the newest valid MainBoard IMU frame from a USB serial port."""

from __future__ import annotations

import os
import queue
import threading
import time
import csv
from pathlib import Path

import serial
from serial.tools import list_ports

from live_tracking.live_common.IMU_Serial_Protocol import MAGIC, HEADER, PACKET_SIZE, VERSION, LiveIMUFrame, PacketError, decode_packet
from common.Const_INFO import IMU_ORDER


MAINBOARD_SERIAL_NUMBERS = ("90:E5:B1:D1:C9:38", "90E5B1D1C938")

class LiveIMUSerial:
    def __init__(self, baud_rate: int = 921600) -> None:
        self.baud_rate = baud_rate
        self.frames: queue.Queue[LiveIMUFrame] = queue.Queue(maxsize=128)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.port_name: str | None = None
        self.valid_packets = 0
        self.invalid_packets = 0
        self.sequence_gaps = 0
        self.consumer_overruns = 0
        self.last_sequence: int | None = None
        self.error: str | None = None

        project_root = Path(__file__).resolve().parents[2]
        self._raw_debug_file = os.environ.get("IMU_RAW_DEBUG_FILE", str(project_root / "recordings" / "raw_packet_debug.csv"))
        self._raw_debug_ready = False

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive(): return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="IMU-serial-listener", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None: self.thread.join(timeout=2.0)

    def get_nowait(self) -> LiveIMUFrame | None:
        try: return self.frames.get_nowait()
        except queue.Empty: return None

    def _configured_port(self) -> str | None:
        return os.environ.get("MAINBOARD_SERIAL_PORT") or os.environ.get("IMU_SERIAL_PORT")

    def _candidate_ports(self) -> list[str]:
        ports = list(list_ports.comports())
        descriptions = {port.device: f"{port.description} {port.manufacturer} {port.hwid}".lower() for port in ports}
        mainboard_ports = [port.device for port in ports if any(serial_number.lower() in descriptions[port.device] for serial_number in MAINBOARD_SERIAL_NUMBERS)]
        uart_bridges = [port.device for port in ports if any(word in descriptions[port.device] for word in ("ch343", "ch340", "cp210", "ftdi", "usb-enhanced-serial"))]
        likely = [port.device for port in ports if any(word in descriptions[port.device] for word in ("esp", "jtag", "usb serial"))]
        ordered: list[str] = []
        for device in mainboard_ports + uart_bridges + likely + [port.device for port in ports]:
            if device not in ordered: ordered.append(device)
        return ordered

    def _port_has_imu_packets(self, port: str, seconds: float = 1.5) -> bool:
        deadline = time.monotonic() + seconds
        buffer = bytearray()
        try:
            connection = serial.Serial(port=None, baudrate=self.baud_rate, timeout=0.05)
            connection.port = port
            connection.dtr = False
            connection.rts = False
            connection.open()
            with connection:
                while time.monotonic() < deadline and not self.stop_event.is_set():
                    chunk = connection.read(max(connection.in_waiting, 1))
                    if not chunk: continue
                    buffer.extend(chunk)
                    while True:
                        magic_index = buffer.find(MAGIC)
                        if magic_index < 0:
                            if len(buffer) > len(MAGIC) - 1: del buffer[:-(len(MAGIC) - 1)]
                            break
                        if magic_index > 0: del buffer[:magic_index]
                        if len(buffer) >= HEADER.size:
                            received_version = buffer[4]
                            declared_size = int.from_bytes(buffer[6:8], "big")
                            if received_version != VERSION or declared_size != PACKET_SIZE:
                                del buffer[:1]
                                continue
                            if len(buffer) < declared_size: break
                            try:
                                decode_packet(bytes(buffer[:declared_size]))
                                return True
                            except PacketError:
                                del buffer[:1]
                        else:
                            break
        except serial.SerialException:
            return False
        return False

    def _find_port(self) -> str | None:
        configured = self._configured_port()
        if configured: return configured
        candidates = self._candidate_ports()
        for port in candidates:
            if self._port_has_imu_packets(port): return port
        if len(candidates) == 1: return candidates[0]
        return None

    def _run(self) -> None:
        while not self.stop_event.is_set():
            port = self._find_port()
            if port is None:
                self.error = "No MainBoard COM port with IMU1 packets found. Set MAINBOARD_SERIAL_PORT=COMx if needed."
                time.sleep(1.0)
                continue
            try:
                connection = serial.Serial(port=None, baudrate=self.baud_rate, timeout=0.1)
                connection.port = port
                connection.dtr = False
                connection.rts = False
                connection.open()
                with connection:
                    self.port_name = port
                    self.error = None
                    print(f"MainBoard serial connected on {port} at {self.baud_rate} baud", flush=True)
                    self._receive(connection)
            except serial.SerialException as exc:
                self.error = str(exc)
                time.sleep(1.0)
            finally: self.port_name = None

    def _receive(self, connection: serial.Serial) -> None:
        buffer = bytearray()
        while not self.stop_event.is_set():
            chunk = connection.read(max(connection.in_waiting, 1))
            if not chunk: continue
            buffer.extend(chunk)
            while True:
                magic_index = buffer.find(MAGIC)
                if magic_index < 0:
                    if len(buffer) > len(MAGIC) - 1: del buffer[:-(len(MAGIC) - 1)]
                    break
                if magic_index > 0: del buffer[:magic_index]
                if len(buffer) >= HEADER.size:
                    received_version = buffer[4]
                    declared_size = int.from_bytes(buffer[6:8], "big")
                    if received_version != VERSION or declared_size != PACKET_SIZE:
                        self.error = f"Ignoring false IMU1 header/protocol mismatch on serial stream: version {received_version}, {declared_size}-byte candidate; waiting for valid {PACKET_SIZE}-byte packets."
                        self.invalid_packets += 1
                        del buffer[:1]
                        continue
                else:
                    break
                if len(buffer) < declared_size: break
                packet = bytes(buffer[:declared_size])
                try: frame = decode_packet(packet)
                except PacketError:
                    self.invalid_packets += 1
                    del buffer[:1]
                    continue
                del buffer[:declared_size]
                self._track_sequence(frame.sequence)
                self._write_raw_debug(frame)
                try: self.frames.put_nowait(frame)
                except queue.Full:
                    self.consumer_overruns += 1
                    try: self.frames.get_nowait()
                    except queue.Empty: pass
                    self.frames.put_nowait(frame)
                self.valid_packets += 1

    def _write_raw_debug(self, frame: LiveIMUFrame) -> None:
        """Optionally save decoded packet values before filtering or fusion."""
        if not self._raw_debug_file:
            return
        try:
            Path(self._raw_debug_file).parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if self._raw_debug_ready else "w"
            with open(self._raw_debug_file, mode, newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if not self._raw_debug_ready:
                    writer.writerow(["sequence", "timestamp_us", "imu", "channel_id", "ax_raw", "ay_raw", "az_raw", "gx_raw", "gy_raw", "gz_raw", "available", "acc_fresh", "gyro_fresh", "diagnostic_code", "stale_scans"])
                    self._raw_debug_ready = True
                for channel_id, record in sorted(frame.records.items()):
                    writer.writerow([frame.sequence, frame.timestamp_us, IMU_ORDER[channel_id], channel_id, *record.acc_raw, *record.gyro_raw, int(record.available), int(record.acc_fresh), int(record.gyro_fresh), record.diagnostic_code, record.stale_scans])
        except OSError as exc:
            self.error = f"Raw IMU debug logging failed: {exc}"

    def _track_sequence(self, sequence: int) -> None:
        if self.last_sequence is not None:
            step = (sequence - self.last_sequence) & 0xFFFFFFFF
            if 1 < step < 0x80000000: self.sequence_gaps += step - 1
        self.last_sequence = sequence
