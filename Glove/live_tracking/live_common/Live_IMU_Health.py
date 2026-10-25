"""Shared IMU health reporting and invalid-sample holding."""
from __future__ import annotations
import time
from common.Const_INFO import IMU_ORDER, Site_Index
from live_tracking.live_common.Live_IMU_Mapping import CHANNEL_TO_SITE

def empty_health() -> dict:
    return {site: {"available": False, "fresh": False, "stale": 0, "code": 0, "who": 0, "ctrl1": 0, "ctrl2": 0, "status": 0, "held": 0} for site in IMU_ORDER}


def update_health(health: dict, frame) -> None:
    for channel_id, record in frame.records.items():
        site = CHANNEL_TO_SITE.get(channel_id)
        if site is None: continue
        previous_held = health.get(site, {}).get("held", 0)
        health[site] = {"available": record.available, "fresh": record.acc_fresh and record.gyro_fresh, "stale": record.stale_scans, "code": record.diagnostic_code, "who": record.who_am_i, "ctrl1": record.ctrl1_xl, "ctrl2": record.ctrl2_g, "status": record.status_register, "held": previous_held}


def emit_health(health: dict) -> None:
    for site in IMU_ORDER:
        item = health[site]
        state = "GOOD" if item["available"] and item["fresh"] and item["code"] == 0 else "BAD"
        print(f"IMU_HEALTH {site} {state} {item['stale']} {item.get('held', 0)} {item['code']} {item['who']:02X} {item['ctrl1']:02X} {item['ctrl2']:02X} {item['status']:02X}", flush=True)


def record_good(record) -> bool:
    return bool(record.available and record.acc_fresh and record.gyro_fresh and record.diagnostic_code == 0)


def hold_bad_records(accel, gyro, frame, previous_accel, previous_gyro, exclusions: set[str], health: dict | None = None) -> None:
    """Hold the previous valid sample when firmware marks the current record invalid."""
    if previous_accel is None or previous_gyro is None:
        return
    for channel_id, record in frame.records.items():
        site = CHANNEL_TO_SITE.get(channel_id)
        if site is None or site in exclusions or record_good(record):
            continue
        index = Site_Index.get(site)
        if index is None or index >= len(accel) or index >= len(gyro):
            continue
        accel[index] = previous_accel[index]
        gyro[index] = previous_gyro[index]
        if health is not None and site in health:
            health[site]["held"] = int(health[site].get("held", 0)) + 1


class HealthReporter:
    def __init__(self, interval_s: float = 1.0):
        self.health = empty_health()
        self.interval_s = interval_s
        self.last_emit = 0.0
        self.has_frame = False
        self.last_signature = None
        self.has_frame = False

    def update(self, frame) -> None:
        update_health(self.health, frame)
        self.has_frame = True

    def maybe_emit(self) -> None:
        if not self.has_frame: return
        now = time.monotonic()
        signature = tuple((site, self.health[site]["available"], self.health[site]["fresh"], self.health[site]["code"]) for site in IMU_ORDER)
        # Emit once, then only when a health value changes.
        if self.last_signature is None or signature != self.last_signature:
            emit_health(self.health)
            self.last_emit = now
            self.last_signature = signature



