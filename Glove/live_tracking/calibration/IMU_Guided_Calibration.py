"""Stage 1: guided stationary and motion calibration using only the live IMUs."""
from __future__ import annotations
import argparse, ctypes, json, os, queue, sys, threading, time
from pathlib import Path
PROJECT_FOLDER = Path(__file__).resolve().parents[2]
LIVE_FOLDER = Path(__file__).resolve().parents[1]
if str(PROJECT_FOLDER) not in sys.path: sys.path.insert(0, str(PROJECT_FOLDER))
import mujoco, mujoco.viewer
import numpy as np
from common.Const_INFO import IMU_ORDER, Joint_ORDER, MODEL_PATH, NUM_IMUS, NUM_QPOS, WRIST_WORLD_DOF, Site_Index, Site_Initial_quat
from common.Sweeps_Errors import rel_quat_compute
from live_tracking.live_common.Live_IMU_Axis_Controls import AxisControls
from live_tracking.live_common.Live_IMU_Calibration import FILTER_SETTLE_FRAMES, CalibrationData, OnlineGyroBias, StationaryCalibration, apply_reference, apply_sensor_alignment, reference_corrections
from live_tracking.live_common.Live_IMU_Display import apply_display_qpos
from live_tracking.live_common.Live_IMU_Display_Controls import DisplayControls, locked_qpos_indices
from live_tracking.Live_Tracking import initial_live_quaternions, update_madgwick_quaternions
from live_tracking.live_common.Live_IMU_Joint_Calibration import apply_calibration_locks, fit_stage1_endpoint_mapping
from live_tracking.live_common.Live_IMU_Mapping import CHANNEL_TO_SITE, enabled_mask, frame_to_arrays, invalid_enabled_sites
from live_tracking.live_common.Live_IMU_Health import HealthReporter, hold_bad_records
from live_tracking.live_common.Live_IMU_Recorder import LiveIMURecorder
from live_tracking.live_common.Live_IMU_Serial import LiveIMUSerial
from common.Quat_Operation import rel_quat_to_qpos2

DISABLED_FILE = LIVE_FOLDER / "disabled_imus.json"
CALIBRATION_FILE = LIVE_FOLDER / "live_imu_calibration_candidate.json"
AXIS_CONTROL_FILE = LIVE_FOLDER / "joint_axis_controls.json"
DISPLAY_CONTROL_FILE = LIVE_FOLDER / "display_controls.json"
SAMPLE_RATE_HZ = 100
FINGER_SITES = {
    "thumb": ["imu_thumb_prox", "imu_thumb_dist"],
    "index": ["imu_index_prox", "imu_index_mid", "imu_index_dist"],
    "middle": ["imu_middle_prox", "imu_middle_mid", "imu_middle_dist"],
    "ring": ["imu_ring_prox", "imu_ring_mid", "imu_ring_dist"],
    "pinky": ["imu_pinky_prox", "imu_pinky_mid", "imu_pinky_dist"]
}
GUIDED_STEP_SEQUENCE = ["stationary", "neutral", *[step for finger in FINGER_SITES for step in (f"straight_{finger}", f"flexed_{finger}")], "fingers_together", "all_spread", *[step for finger in FINGER_SITES for step in (f"abduction_left_{finger}", f"abduction_neutral_{finger}", f"abduction_right_{finger}")], "return_reference"]


class GoBackRequested(Exception): pass



def current_calibration_locks() -> list[int]:
    axis_controls = AxisControls(AXIS_CONTROL_FILE)
    display_controls = DisplayControls(DISPLAY_CONTROL_FILE)
    return locked_qpos_indices(axis_controls.values, display_controls.disable_abduction)

def read_disabled() -> set[str]:
    try: return set(json.loads(DISABLED_FILE.read_text(encoding="utf-8")))
    except (FileNotFoundError, OSError, ValueError): return set()


def command_reader(commands: queue.Queue[str]) -> None:
    for line in sys.stdin: commands.put(line.strip().upper())



def snap_viewer_right() -> int | None:
    if os.name != "nt": return None
    user32 = ctypes.windll.user32
    work_area = (ctypes.c_long * 4)()
    user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work_area), 0)
    left, top, right, bottom = work_area
    middle, current_pid = left + (right - left) // 2, os.getpid()

    windows = []
    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def move_process_window(window, parameter):
        process_id = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(window, ctypes.byref(process_id))
        if process_id.value == current_pid and user32.IsWindowVisible(window): windows.append(int(window)); user32.MoveWindow(window, middle, top, right - middle, bottom - top, True)
        return True
    user32.EnumWindows(move_process_window, 0)
    return windows[0] if windows else None


class Stage1Preview:
    def __init__(self, viewer, model, data, exclusions: set[str], gain: float, use_gyro_bias: bool) -> None:
        self.viewer, self.model, self.data, self.exclusions = viewer, model, data, exclusions
        self.gain, self.use_gyro_bias = gain, use_gyro_bias
        self.calibration, self.quaternions, self.corrections, self.online_bias, self.sensor_alignment = None, None, None, None, None
        self.health_reporter = HealthReporter()
        self.recorder = LiveIMURecorder(PROJECT_FOLDER, SAMPLE_RATE_HZ)
        print(f"CALIBRATION_PREVIEW_RECORDING {self.recorder.path}", flush=True)
        self.previous_accel = None
        self.previous_gyro = None
        self.mapping, self.capture_mapping = None, None
        self.axis_controls = AxisControls(AXIS_CONTROL_FILE)
        self.display_controls = DisplayControls(DISPLAY_CONTROL_FILE)
        self.latest_qpos = np.zeros(NUM_QPOS)
        self.state = {"frame": 0, "accel": np.zeros((1, NUM_IMUS, 3)), "gyro": np.zeros((1, NUM_IMUS, 3)), "site_quats": np.asarray([[Site_Initial_quat[site] for site in IMU_ORDER]], dtype=float)}

    def configure(self, calibration: CalibrationData) -> None:
        self.calibration = calibration
        self.online_bias = OnlineGyroBias(calibration, SAMPLE_RATE_HZ)
        self.quaternions = initial_live_quaternions()
        for _ in range(FILTER_SETTLE_FRAMES): update_madgwick_quaternions(self.quaternions, calibration.mean_accel, np.zeros((NUM_IMUS, 3)), self.gain, 1.0 / SAMPLE_RATE_HZ)
        self.corrections = reference_corrections(self.quaternions.copy())
        self.sensor_alignment = calibration.guided_imu.get("sensor_to_segment_alignment") if calibration.guided_imu else None

    def reanchor(self) -> None:
        if self.quaternions is not None: self.corrections = reference_corrections(self.quaternions.copy())

    def set_mapping(self, mapping: dict) -> None: self.mapping = mapping

    def set_sensor_alignment(self, alignment: dict | None) -> None: self.sensor_alignment = alignment

    def set_capture_mapping(self, mapping: dict | None) -> None: self.capture_mapping = mapping

    def update(self, frame=None, sync: bool = True) -> None:
        if frame is not None and self.quaternions is not None:
            self.recorder.append(frame, CHANNEL_TO_SITE)
            self.health_reporter.update(frame)
            self.health_reporter.maybe_emit()
            accel, gyro = frame_to_arrays(frame, self.exclusions)
            hold_bad_records(accel, gyro, frame, self.previous_accel, self.previous_gyro, self.exclusions, self.health_reporter.health)
            self.previous_accel, self.previous_gyro = accel.copy(), gyro.copy()
            if self.use_gyro_bias: gyro = self.online_bias.correct(accel, gyro)
            self.state["accel"][0], self.state["gyro"][0] = accel, gyro
            raw_quaternions = update_madgwick_quaternions(self.quaternions, accel, gyro, self.gain, 1.0 / SAMPLE_RATE_HZ)
            quaternions = apply_sensor_alignment(raw_quaternions, self.sensor_alignment) if self.sensor_alignment else apply_reference(raw_quaternions, self.corrections)
            self.state["site_quats"][0] = quaternions
            if sync:
                model_qpos = self.axis_controls.apply_wrist_world_qpos(rel_quat_to_qpos2(rel_quat_compute(self.state)))
                qpos = model_qpos[WRIST_WORLD_DOF:WRIST_WORLD_DOF + NUM_QPOS]
                self.latest_qpos = (np.asarray(self.capture_mapping["matrix"]) @ qpos + np.asarray(self.capture_mapping["offset"])) if self.capture_mapping else qpos.copy()
                if self.mapping:
                    display_qpos = np.asarray(self.mapping["matrix"]) @ qpos + np.asarray(self.mapping["offset"])
                    for joint_index, joint_name in enumerate(Joint_ORDER):
                        curve = self.mapping.get("piecewise_mapping", {}).get(joint_name)
                        if curve and len(curve.get("measured", [])) >= 2:
                            display_qpos[joint_index] = float(np.interp(float(qpos[joint_index]), curve["measured"], curve["target"]))
                else: display_qpos = qpos
                final_qpos = apply_display_qpos(display_qpos, self.model.jnt_range[WRIST_WORLD_DOF:WRIST_WORLD_DOF + NUM_QPOS], self.axis_controls, self.display_controls)
                model_qpos[WRIST_WORLD_DOF:WRIST_WORLD_DOF + NUM_QPOS] = final_qpos
                self.data.qpos[:] = model_qpos
                mujoco.mj_forward(self.model, self.data)
        if sync: self.viewer.sync()


def wait_for_command(receiver: LiveIMUSerial, preview: Stage1Preview, commands: queue.Queue[str], allowed: set[str]) -> str:
    while preview.viewer.is_running():
        try:
            command = commands.get_nowait()
            if command in allowed: return command
        except queue.Empty: pass
        frame = receiver.get_nowait()
        preview.update(frame)
        time.sleep(0.002)
    raise RuntimeError("Stage 1 MuJoCo viewer was closed.")


def wait_for_final_decision(receiver: LiveIMUSerial, preview: Stage1Preview, commands: queue.Queue[str]) -> str | None:
    while preview.viewer.is_running():
        try:
            command = commands.get_nowait()
            if command == "ACCEPT_STAGE1": return None
        except queue.Empty: pass
        preview.update(receiver.get_nowait())
        time.sleep(0.002)
    raise RuntimeError("Stage 1 MuJoCo viewer was closed.")


def wait_for_start(receiver: LiveIMUSerial, preview: Stage1Preview, commands: queue.Queue[str], code: str, instruction: str, seconds: float, skippable: bool = False, allow_back: bool = False) -> bool:
    print(f"CALIBRATION_STEP_READY {code} {seconds:g} {instruction}", flush=True)
    if allow_back: print("CALIBRATION_CAN_GO_BACK", flush=True)
    allowed = {"START"}
    if skippable: allowed.add("SKIP")
    if allow_back: allowed.add("BACK")
    command = wait_for_command(receiver, preview, commands, allowed)
    if command == "BACK": raise GoBackRequested
    if command == "SKIP": print(f"CALIBRATION_STEP_SKIPPED {code}", flush=True); return False
    return True


def next_valid_frame(receiver: LiveIMUSerial, exclusions: set[str], preview: Stage1Preview):
    last_report, last_report_time = None, 0.0
    while True:
        frame = receiver.get_nowait()
        if frame is None: preview.update(); time.sleep(0.002); continue
        preview.update(frame)
        invalid = invalid_enabled_sites(frame, exclusions)
        if invalid:
            now = time.monotonic()
            if invalid != last_report or now - last_report_time >= 1.0:
                print(f"CALIBRATION_HEALTH_HELD Using latest held sample for: {', '.join(invalid)}", flush=True)
                last_report, last_report_time = invalid.copy(), now
        return frame


def collect_pose_frames(receiver: LiveIMUSerial, exclusions: set[str], preview: Stage1Preview, code: str, seconds: float) -> np.ndarray:
    while receiver.get_nowait() is not None: pass
    required = max(50, int(round(seconds * SAMPLE_RATE_HZ)))
    poses, last_progress = [], -1
    while len(poses) < required:
        next_valid_frame(receiver, exclusions, preview)
        poses.append(preview.latest_qpos.copy())
        progress = int(100 * len(poses) / required)
        if progress != last_progress:
            print(f"CALIBRATION_STEP_PROGRESS {code} {progress} {len(poses)} {required}", flush=True)
            last_progress = progress
    return np.mean(np.unwrap(np.asarray(poses), axis=0), axis=0)


def capture_pose_step(receiver: LiveIMUSerial, exclusions: set[str], preview: Stage1Preview, commands: queue.Queue[str], code: str, instruction: str, seconds: float, skippable: bool = True, allow_back: bool = False) -> np.ndarray | None:
    while True:
        if not wait_for_start(receiver, preview, commands, code, instruction, seconds, skippable, allow_back): return None
        pose = collect_pose_frames(receiver, exclusions, preview, code, seconds)
        if review_step(receiver, preview, commands, code, "Stable endpoint captured. Check the MuJoCo pose, then accept or redo only this hold."): return pose
        print(f"CALIBRATION_STEP_REDO {code}", flush=True)


def capture_reanchored_baseline(receiver: LiveIMUSerial, exclusions: set[str], preview: Stage1Preview) -> np.ndarray:
    preview.reanchor()
    poses = []
    while len(poses) < 50:
        next_valid_frame(receiver, exclusions, preview)
        poses.append(preview.latest_qpos.copy())
    return np.mean(np.unwrap(np.asarray(poses), axis=0), axis=0)

def review_step(receiver: LiveIMUSerial, preview: Stage1Preview, commands: queue.Queue[str], code: str, summary: str) -> bool:
    print(f"CALIBRATION_STEP_REVIEW {code} {summary}", flush=True)
    return wait_for_command(receiver, preview, commands, {"ACCEPT", "REDO"}) == "ACCEPT"


def collect_stationary(receiver: LiveIMUSerial, exclusions: set[str], preview: Stage1Preview, commands: queue.Queue[str], duration_seconds: float, allow_back: bool = False) -> CalibrationData:
    print(f"CALIBRATION_STEP_READY stationary {duration_seconds:g} Hold the straight reference pose naturally still. Collection starts automatically; movement only pauses collection.", flush=True)
    enabled = enabled_mask(exclusions)
    required = max(SAMPLE_RATE_HZ, int(round(duration_seconds * SAMPLE_RATE_HZ)))
    calibration = StationaryCalibration(required, enabled)
    last_progress, movement_reported, attempts = -1, False, 0
    completed_sensors: set[str] = set()
    while not calibration.complete():
        frame = next_valid_frame(receiver, exclusions, preview)
        attempts += 1
        accel, gyro = frame_to_arrays(frame, exclusions)
        accepted = calibration.add(accel, gyro)
        for sensor_index, site in enumerate(IMU_ORDER):
            if enabled[sensor_index] and site not in completed_sensors and len(calibration.gyro_samples[sensor_index]) >= required:
                completed_sensors.add(site)
                print(f"CALIBRATION_SENSOR_COMPLETE stationary {site} {required}/{required}", flush=True)
        if calibration.last_action in {"paused", "partial"} and not movement_reported:
            continuation = "that IMU paused while the others continue" if calibration.last_action == "partial" else "all pending IMUs paused"
            print(f"CALIBRATION_MOTION Significant movement at {calibration.last_moving_imu}: {calibration.last_motion_detail}; {continuation}.", flush=True)
            movement_reported = True
        if calibration.last_action == "accepted" and movement_reported:
            print(f"CALIBRATION_STABLE Hand is stable again; collection resumed from {calibration.count}/{required} accumulated frames.", flush=True)
            movement_reported = False
        progress = int(100 * calibration.count / required)
        if progress != last_progress:
            print(f"CALIBRATION_STEP_PROGRESS stationary {progress} {calibration.count} {required}", flush=True)
            last_progress = progress
        if attempts >= required and attempts % required == 0 and not calibration.complete():
            counts = calibration.enabled_counts()
            lagging = [site for index, site in enumerate(IMU_ORDER) if enabled[index] and counts[index] < required]
            print(f"CALIBRATION_STATIONARY_WAIT Still waiting for clean stationary data from: {', '.join(lagging)}.", flush=True)
    result = calibration.calculate()
    failed, warnings = [], []
    for site in result.enabled_imus:
        index = Site_Index[site]
        maximum_noise_dps = float(np.max(np.rad2deg(result.gyro_std[index])))
        preferred = 7.5 <= result.gravity_magnitude[index] <= 12.0 and maximum_noise_dps <= 2.0
        usable = 6.0 <= result.gravity_magnitude[index] <= 14.0 and maximum_noise_dps <= 10.0
        status = "PASS" if preferred else ("WARN" if usable else "FAIL")
        print(f"CALIBRATION_IMU {site} gravity={result.gravity_magnitude[index]:.3f} max_gyro_std_dps={maximum_noise_dps:.3f}", flush=True)
        print(f"CALIBRATION_VALIDITY {site} {status}", flush=True)
        if status == "WARN": warnings.append(site)
        elif status == "FAIL": failed.append(site)
    if warnings: print(f"CALIBRATION_STATIONARY_WARNING Usable but noisier than preferred: {', '.join(warnings)}. Review the viewer and accept or redo the hold.", flush=True)
    if failed: print(f"CALIBRATION_STATIONARY_WARNING Quality failed for: {', '.join(failed)}. Accept only if the viewer looks usable, otherwise redo this hold.", flush=True)
    return result


def run(duration_seconds: float, gain: float, use_gyro_bias: bool) -> None:
    exclusions = read_disabled()
    enabled_now = [site for site in IMU_ORDER if site not in exclusions]
    receiver = LiveIMUSerial()
    receiver.start()
    commands: queue.Queue[str] = queue.Queue()
    threading.Thread(target=command_reader, args=(commands,), daemon=True).start()
    model = mujoco.MjModel.from_xml_path(str(PROJECT_FOLDER / MODEL_PATH))
    data = mujoco.MjData(model)
    print("CALIBRATION_STAGE1_BEGIN Guided IMU calibration started.", flush=True)
    try:
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "render_camera")
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer_hwnd = snap_viewer_right()
            if viewer_hwnd is not None: print(f"MUJOCO_HWND {viewer_hwnd}", flush=True)
            preview = Stage1Preview(viewer, model, data, exclusions, gain, use_gyro_bias)
            calibration = None
            neutral_pose = together = all_spread = return_reference = None
            straight = {finger: None for finger in FINGER_SITES}
            flexed = {finger: None for finger in FINGER_SITES}
            abduction_left = {finger: None for finger in FINGER_SITES}
            abduction_neutral = {finger: None for finger in FINGER_SITES}
            abduction_right = {finger: None for finger in FINGER_SITES}
            step_index = 0
            while step_index < len(GUIDED_STEP_SEQUENCE):
                code = GUIDED_STEP_SEQUENCE[step_index]
                try:
                    if code == "stationary":
                        while True:
                            calibration = collect_stationary(receiver, exclusions, preview, commands, duration_seconds, allow_back=step_index > 0)
                            preview.configure(calibration)
                            if review_step(receiver, preview, commands, "stationary", "Bias and gravity captured. Move slightly to inspect the response, then accept or redo this step."): break
                    elif code == "neutral": neutral_pose = capture_pose_step(receiver, exclusions, preview, commands, code, "Hold the same fully straight reference pose for the short orientation capture.", 3.0, skippable=False, allow_back=True)
                    elif code.startswith("straight_"):
                        finger = code.removeprefix("straight_")
                        captured = capture_pose_step(receiver, exclusions, preview, commands, code, f"Hold the {finger} finger fully straight. The other fingers may rest comfortably; just avoid large hand movement.", 2.5, allow_back=True)
                        straight[finger] = capture_reanchored_baseline(receiver, exclusions, preview) if captured is not None else None
                    elif code.startswith("flexed_"):
                        finger = code.removeprefix("flexed_")
                        flexed[finger] = capture_pose_step(receiver, exclusions, preview, commands, code, f"Slowly position the ENTIRE {finger} finger at comfortable full flexion, then hold it. Small movement of the other fingers is ignored.", 2.5, allow_back=True)
                    elif code == "fingers_together":
                        together = capture_pose_step(receiver, exclusions, preview, commands, code, "Hold all fingers straight and together at their neutral abduction positions.", 2.5, allow_back=True)
                        if together is not None: capture_reanchored_baseline(receiver, exclusions, preview)
                    elif code == "all_spread": all_spread = capture_pose_step(receiver, exclusions, preview, commands, code, "Hold all fingers straight and comfortably spread apart.", 2.5, allow_back=True)
                    elif code.startswith("abduction_left_"):
                        finger = code.removeprefix("abduction_left_")
                        abduction_left[finger] = capture_pose_step(receiver, exclusions, preview, commands, code, f"Hold the {finger} finger straight at its LEFTMOST comfortable abduction endpoint.", 2.5, allow_back=True)
                    elif code.startswith("abduction_neutral_"):
                        finger = code.removeprefix("abduction_neutral_")
                        abduction_neutral[finger] = capture_pose_step(receiver, exclusions, preview, commands, code, f"Hold the {finger} finger straight at its NEUTRAL abduction position.", 2.5, allow_back=True)
                    elif code.startswith("abduction_right_"):
                        finger = code.removeprefix("abduction_right_")
                        abduction_right[finger] = capture_pose_step(receiver, exclusions, preview, commands, code, f"Hold the {finger} finger straight at its RIGHTMOST comfortable abduction endpoint.", 2.5, allow_back=True)
                    elif code == "return_reference": return_reference = capture_pose_step(receiver, exclusions, preview, commands, code, "Return to the original straight reference pose for the final drift measurement and re-anchor.", 3.0, skippable=False, allow_back=True)
                    step_index += 1
                except GoBackRequested:
                    print(f"CALIBRATION_WENT_BACK {code}", flush=True)
                    step_index = max(0, step_index - 1)
            enabled_sites = set(calibration.enabled_imus)
            guided = {}
            guided.update({"sample_rate_hz": SAMPLE_RATE_HZ, "neutral_mean_accel_m_s2": calibration.mean_accel.tolist(), "neutral_mean_gyro_rad_s": calibration.gyro_bias.tolist(), "neutral_qpos": neutral_pose.tolist(), "flexion": {}, "abduction": {}, "ranges": {}})
            drift_difference = (return_reference[3:] - neutral_pose[3:] + np.pi) % (2.0 * np.pi) - np.pi
            drift_deg = float(np.rad2deg(np.max(np.abs(drift_difference))))
            print(f"CALIBRATION_REFERENCE_DRIFT {drift_deg:.2f}", flush=True)
            preview.reanchor()
            while True:
                calibration_locked_qpos = current_calibration_locks()
                refinement_mapping = apply_calibration_locks(fit_stage1_endpoint_mapping(straight, flexed, abduction_left, abduction_neutral, abduction_right, model.jnt_range[WRIST_WORLD_DOF:WRIST_WORLD_DOF + NUM_QPOS]), calibration_locked_qpos, "guided endpoint calibration")
                mapping = apply_calibration_locks(refinement_mapping, calibration_locked_qpos, "guided endpoint calibration")
                preview.set_mapping(mapping)
                for warning in mapping["warnings"]: print(f"CALIBRATION_MAPPING_WARNING {warning}", flush=True)
                print(f"CALIBRATION_STAGE1_FINAL_REVIEW {len(mapping['warnings'])} mapping warning(s). Test the live viewer, accept Stage 1, or select one calibration block to redo.", flush=True)
                redo = wait_for_final_decision(receiver, preview, commands)
                if redo is None: break
            pose_json = lambda pose: None if pose is None else pose.tolist()
            guided["endpoint_holds"] = {"straight": {finger: pose_json(pose) for finger, pose in straight.items()}, "flexed": {finger: pose_json(pose) for finger, pose in flexed.items()}, "fingers_together": pose_json(together), "all_spread": pose_json(all_spread), "abduction_left": {finger: pose_json(pose) for finger, pose in abduction_left.items()}, "abduction_neutral": {finger: pose_json(pose) for finger, pose in abduction_neutral.items()}, "abduction_right": {finger: pose_json(pose) for finger, pose in abduction_right.items()}, "return_reference": return_reference.tolist(), "reference_drift_deg": drift_deg}
            guided["stage1_endpoint_mapping"] = mapping
            guided["flexion"] = {finger: {"straight_qpos": pose_json(straight[finger]), "flexed_qpos": pose_json(flexed[finger])} for finger in FINGER_SITES}
            guided["abduction"] = {finger: {"left_qpos": pose_json(abduction_left[finger]), "neutral_qpos": pose_json(abduction_neutral[finger]), "right_qpos": pose_json(abduction_right[finger])} for finger in FINGER_SITES}
            guided["ranges"] = mapping["details"]
            calibration.guided_imu = guided
            calibration.save(CALIBRATION_FILE)
            print(f"CALIBRATION_COMPLETE {CALIBRATION_FILE}", flush=True)
    finally:
        try: preview.recorder.close()
        except Exception: pass
        receiver.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--filter", default="Madgwick", help=argparse.SUPPRESS)
    parser.add_argument("--gain", type=float, default=0.1)
    parser.add_argument("--use-gyro-bias", action=argparse.BooleanOptionalAction, default=True)
    arguments = parser.parse_args()
    try:
        run(arguments.duration, arguments.gain, arguments.use_gyro_bias)
    except KeyboardInterrupt: print("CALIBRATION_CANCELLED", flush=True)










