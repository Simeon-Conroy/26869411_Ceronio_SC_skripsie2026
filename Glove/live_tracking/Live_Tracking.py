"""Drive the MuJoCo hand from calibrated live MainBoard measurements."""
from __future__ import annotations
import argparse, ctypes, json, os, sys, time
from datetime import datetime
from pathlib import Path
PROJECT_FOLDER = Path(__file__).resolve().parents[1]
LIVE_FOLDER = Path(__file__).resolve().parent
if str(PROJECT_FOLDER) not in sys.path: sys.path.insert(0, str(PROJECT_FOLDER))
import mujoco, mujoco.viewer
import numpy as np
from common.Const_INFO import IMU_ORDER, MODEL_PATH, NUM_IMUS, NUM_QPOS, WRIST_WORLD_DOF, Site_Initial_quat
from common.Sweeps_Errors import rel_quat_compute
from live_tracking.live_common.Live_IMU_Axis_Controls import AxisControls
from live_tracking.live_common.Live_IMU_Calibration import FILTER_SETTLE_FRAMES, CalibrationData, OnlineGyroBias, apply_reference, apply_sensor_alignment, reference_corrections
from live_tracking.live_common.Live_IMU_Display import apply_display_qpos
from live_tracking.live_common.Live_IMU_Display_Controls import DisplayControls
from common.madgwick import madgwick_filter
from live_tracking.live_common.Live_IMU_Joint_Calibration import apply_joint_mapping
from live_tracking.live_common.Live_IMU_Mapping import CHANNEL_TO_SITE, frame_to_arrays
from live_tracking.live_common.Live_IMU_Health import HealthReporter, hold_bad_records
from live_tracking.live_common.Live_IMU_Recorder import LiveIMURecorder
from live_tracking.live_common.Live_IMU_Serial import LiveIMUSerial
from common.Quat_Operation import rel_quat_to_qpos2

SAMPLE_PERIOD_S = 0.01
RAW_GRAVITY_M_S2 = 9.80665
MADGWICK_FILTER = "Madgwick"
DEFAULT_MADGWICK_GAIN = 0.1


def initial_live_quaternions() -> np.ndarray:
    return np.asarray([Site_Initial_quat[site] for site in IMU_ORDER], dtype=float).copy()

def update_madgwick_quaternions(quaternions: np.ndarray, accel: np.ndarray, gyro: np.ndarray, gain: float, sample_period: float) -> np.ndarray:
    for index in range(len(IMU_ORDER)):
        quaternions[index] = madgwick_filter(gyro[index].copy(), quaternions[index], accel[index].copy(), sample_period, gain)
    return quaternions.copy()

DISABLED_FILE = LIVE_FOLDER / "disabled_imus.json"
CALIBRATION_FILE = LIVE_FOLDER / "live_imu_calibration.json"
AXIS_CONTROL_FILE = LIVE_FOLDER / "joint_axis_controls.json"
DISPLAY_CONTROL_FILE = LIVE_FOLDER / "display_controls.json"



def snap_viewer_right() -> int | None:
    if os.name != "nt": return None
    user32 = ctypes.windll.user32
    work_area = (ctypes.c_long * 4)()
    user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work_area), 0)
    left, top, right, bottom = work_area
    middle = left + (right - left) // 2
    current_pid = os.getpid()

    windows = []
    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def move_process_window(window, parameter):
        process_id = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(window, ctypes.byref(process_id))
        if process_id.value == current_pid and user32.IsWindowVisible(window): windows.append(int(window)); user32.MoveWindow(window, middle, top, right - middle, bottom - top, True)
        return True
    user32.EnumWindows(move_process_window, 0)
    return windows[0] if windows else None


def read_disabled() -> set[str]:
    try: return set(json.loads(DISABLED_FILE.read_text(encoding="utf-8")))
    except (FileNotFoundError, OSError, ValueError): return set()

def run(gain: float, use_gyro_bias: bool, recording_folder: str | None = None) -> None:
    disabled = read_disabled()
    enabled_now = [site for site in IMU_ORDER if site not in disabled]

    display_controls = DisplayControls(DISPLAY_CONTROL_FILE)

    if display_controls.raw_mode:
        calibration = CalibrationData(gyro_bias=np.zeros((NUM_IMUS, 3), dtype=float), gyro_std=np.zeros((NUM_IMUS, 3), dtype=float), mean_accel=np.tile(np.array([0.0, 0.0, RAW_GRAVITY_M_S2], dtype=float), (NUM_IMUS, 1)), gravity_magnitude=np.full(NUM_IMUS, RAW_GRAVITY_M_S2, dtype=float), rejected_samples=np.zeros(NUM_IMUS, dtype=int), enabled_imus=enabled_now, sample_count=0, created_at=datetime.now().isoformat(timespec="seconds"), guided_imu={"mode": "raw_runtime"})
    else:
        calibration = CalibrationData.load(CALIBRATION_FILE)
        if calibration.enabled_imus != enabled_now: raise RuntimeError("Enabled IMUs do not match the saved calibration; recalibrate with the current selection.")

    model = mujoco.MjModel.from_xml_path(str(PROJECT_FOLDER / MODEL_PATH))
    data = mujoco.MjData(model)

    axis_controls = AxisControls(AXIS_CONTROL_FILE)
    if model.nq < WRIST_WORLD_DOF + NUM_QPOS: raise RuntimeError(f"Expected at least {WRIST_WORLD_DOF + NUM_QPOS} qpos values, model has {model.nq}")

    receiver = LiveIMUSerial()
    quaternions = initial_live_quaternions()

    recorder_folder = Path(recording_folder) if recording_folder else None
    recorder = LiveIMURecorder(PROJECT_FOLDER, 1.0 / SAMPLE_PERIOD_S, recorder_folder)

    state = {"frame": 0, "accel": np.zeros((1, NUM_IMUS, 3)), "gyro": np.zeros((1, NUM_IMUS, 3)), "site_quats": np.asarray([[Site_Initial_quat[site] for site in IMU_ORDER]], dtype=float)}

    for _ in range(FILTER_SETTLE_FRAMES): update_madgwick_quaternions(quaternions, calibration.mean_accel, np.zeros((NUM_IMUS, 3)), gain, SAMPLE_PERIOD_S)

    corrections = reference_corrections(quaternions.copy())

    sensor_alignment = calibration.guided_imu.get("sensor_to_segment_alignment") if calibration.guided_imu else None

    receiver.start()
    last_processed_sequence, last_status_time, last_disabled_refresh = None, 0.0, 0.0
    health_reporter = HealthReporter()
    previous_accel = previous_gyro = None
    viewer_holder = {}

    def key_callback(keycode: int) -> None:
        if (chr(keycode).upper() if 0 <= keycode < 256 else "") == "Q": viewer_holder["viewer"].close()

    online_bias = None if display_controls.raw_mode else OnlineGyroBias(calibration, 1.0 / SAMPLE_PERIOD_S)
    bias_state = "enabled" if use_gyro_bias else "disabled"
    parameters = f"gain {gain}"
    mode_text = "raw/no-calibration" if display_controls.raw_mode else "calibrated"
    print(f"CALIBRATION_LOADED {CALIBRATION_FILE} Gyro bias correction {bias_state}. Mode {mode_text}.")
    print(f"Live {MADGWICK_FILTER} tracking started with {parameters}. Wrist-to-world tracking is active. Recording to {recorder.path}")
    
    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback, show_left_ui=True, show_right_ui=False) as viewer:
            viewer_holder["viewer"] = viewer
            viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "render_camera")
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer_hwnd = snap_viewer_right()
            
            if viewer_hwnd is not None: print(f"MUJOCO_HWND {viewer_hwnd}", flush=True)
            while viewer.is_running():
                processed = 0
                while True:
                    frame = receiver.get_nowait()
                    if frame is None: break
                    health_reporter.update(frame)
                    if time.monotonic() - last_disabled_refresh >= 0.1:
                        disabled = read_disabled()
                        last_disabled_refresh = time.monotonic()
                    accel, gyro = frame_to_arrays(frame, disabled)
                    hold_bad_records(accel, gyro, frame, previous_accel, previous_gyro, disabled, health_reporter.health)
                    previous_accel, previous_gyro = accel.copy(), gyro.copy()
                    display_controls.refresh()

                    if use_gyro_bias and online_bias is not None: gyro = online_bias.correct(accel, gyro)
                    state["accel"][0], state["gyro"][0] = accel, gyro
                    
                    raw_quaternions = update_madgwick_quaternions(quaternions, accel, gyro, gain, SAMPLE_PERIOD_S)
                    calibrated_quaternions = apply_sensor_alignment(raw_quaternions, sensor_alignment) if sensor_alignment else apply_reference(raw_quaternions, corrections)
                    state["site_quats"][0] = calibrated_quaternions
                    model_qpos = axis_controls.apply_wrist_world_qpos(rel_quat_to_qpos2(rel_quat_compute(state)))
                    hand_qpos = model_qpos[WRIST_WORLD_DOF:WRIST_WORLD_DOF + NUM_QPOS]
                    hand_qpos = hand_qpos.copy()
                    mapped_qpos = hand_qpos if display_controls.raw_mode else apply_joint_mapping(hand_qpos, calibration)
                    display_qpos = apply_display_qpos(mapped_qpos, model.jnt_range[WRIST_WORLD_DOF:WRIST_WORLD_DOF + NUM_QPOS], axis_controls, display_controls)
                    model_qpos[WRIST_WORLD_DOF:WRIST_WORLD_DOF + NUM_QPOS] = display_qpos
                    data.qpos[:] = model_qpos
                    recorder.append(frame, CHANNEL_TO_SITE)
                    last_processed_sequence, processed = frame.sequence, processed + 1
                if processed: mujoco.mj_forward(model, data)
                viewer.sync()
                now = time.monotonic()
                if now - last_status_time >= 1.0:
                    connection = f"connected on {receiver.port_name}" if receiver.port_name else (f"error: {receiver.error}" if receiver.error else "waiting for MainBoard serial")
                    print(f"Serial {connection} | last frame {'none' if last_processed_sequence is None else last_processed_sequence} | queued {receiver.frames.qsize()} | overruns {receiver.consumer_overruns} | gaps {receiver.sequence_gaps} | invalid {receiver.invalid_packets}")
                    health_reporter.maybe_emit()
                    last_status_time = now
                time.sleep(0.001)
    finally:
        receiver.stop()
        recorder.close()
        print(f"Saved {recorder.count} live frames to {recorder.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default=MADGWICK_FILTER, help=argparse.SUPPRESS)
    parser.add_argument("--gain", type=float, default=DEFAULT_MADGWICK_GAIN)
    parser.add_argument("--use-gyro-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recording-folder", default=None)
    arguments = parser.parse_args()
    try: run(arguments.gain, arguments.use_gyro_bias, arguments.recording_folder)
    except KeyboardInterrupt: print("Live tracking stopped.")




















