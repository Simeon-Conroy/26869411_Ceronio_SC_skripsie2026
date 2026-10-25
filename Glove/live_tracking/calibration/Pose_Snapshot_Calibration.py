"""Calibrate the live glove against named MuJoCo pose snapshots."""
from __future__ import annotations
import argparse, json, queue, re, sys, threading
from pathlib import Path
PROJECT_FOLDER = Path(__file__).resolve().parents[2]
LIVE_FOLDER = Path(__file__).resolve().parents[1]
if str(PROJECT_FOLDER) not in sys.path: sys.path.insert(0, str(PROJECT_FOLDER))
import h5py, mujoco, mujoco.viewer
import numpy as np
from live_tracking.calibration.IMU_Guided_Calibration import GoBackRequested, Stage1Preview, collect_stationary, command_reader, next_valid_frame, read_disabled, review_step, snap_viewer_right, wait_for_command, wait_for_start
from common.Const_INFO import IMU_ORDER, Joint_ORDER, MODEL_PATH, NUM_IMUS, NUM_QPOS, WRIST_WORLD_DOF
from common.Sweeps_Errors import rel_quat_compute
from live_tracking.live_common.Live_IMU_Calibration import CalibrationData, apply_sensor_alignment, fit_sensor_alignment, quaternion_mean
from live_tracking.live_common.Live_IMU_Axis_Controls import AxisControls
from live_tracking.live_common.Live_IMU_Display_Controls import DisplayControls, locked_qpos_indices
from live_tracking.live_common.Live_IMU_Joint_Calibration import apply_calibration_locks
from live_tracking.live_common.Live_IMU_Mapping import frame_to_arrays
from live_tracking.live_common.Live_IMU_Serial import LiveIMUSerial
from common.Quat_Operation import rel_quat_to_qpos2

CALIBRATION_FILE = LIVE_FOLDER / "live_imu_calibration_candidate.json"
AXIS_CONTROL_FILE = LIVE_FOLDER / "joint_axis_controls.json"
DISPLAY_CONTROL_FILE = LIVE_FOLDER / "display_controls.json"
IMAGE_FOLDER = LIVE_FOLDER / ".pose_calibration_images"
SAMPLE_RATE_HZ = 100
POSE_SECONDS = 2.5
PALM_WRIST_BLOCK = "|".join(Joint_ORDER[:3])
PALM_WRIST_MIN_RANGE_DEG = np.asarray([20.0, 20.0, 15.0])
BLOCKS = [(0, 1, 2), (3, 4), (5, 6), (7,), (8, 9), (10,), (11,), (12, 13), (14,), (15,), (16, 17), (18,), (19,), (20, 21), (22,), (23,)]
FLEXION_JOINTS = (3, 5, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23)
ABDUCTION_JOINTS = (4, 6, 9, 13, 17, 21)
FLEXION_MIN_GAIN = 0.10
FLEXION_MAX_GAIN = 4.5

def safe_code(index: int, label: str) -> str: return f"pose_{index + 1:03d}_{re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_') or 'unnamed'}"


def load_targets(path: Path) -> list[dict]:
    with h5py.File(path, "r") as file:
        if "snapshots/qpos" not in file or "snapshots/label" not in file or "snapshots/world_quat" not in file: raise ValueError("This pose-snapshot file must contain snapshots/qpos, snapshots/label and snapshots/world_quat.")
        qpos = np.asarray(file["snapshots/qpos"], dtype=float)
        world_quat = np.asarray(file["snapshots/world_quat"], dtype=float)
        relative_quat = np.asarray(file["snapshots/relative_quat"], dtype=float) if "snapshots/relative_quat" in file else None
        wrist_world_quat = np.asarray(file["snapshots/wrist_world_quat"], dtype=float) if "snapshots/wrist_world_quat" in file else np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (len(qpos), 1))
        wrist_world_qpos = np.asarray(file["snapshots/wrist_world_qpos"], dtype=float) if "snapshots/wrist_world_qpos" in file else np.zeros((len(qpos), 3))
        labels = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in file["snapshots/label"][:]]
        if qpos.ndim != 2 or qpos.shape[1] != len(Joint_ORDER): raise ValueError(f"Snapshot qpos must have {len(Joint_ORDER)} columns; found {qpos.shape}.")
        if len(labels) != len(qpos) or not labels: raise ValueError("The snapshot file contains no usable named poses.")
        if world_quat.shape != (len(qpos), len(IMU_ORDER), 4): raise ValueError(f"Snapshot world_quat must have shape ({len(qpos)}, {len(IMU_ORDER)}, 4); found {world_quat.shape}.")
        if "meta/imu_order" in file:
            imu_order = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in file["meta/imu_order"][:]]
            if imu_order != IMU_ORDER: raise ValueError("Snapshot IMU order does not match this glove model.")
        if "meta/qpos_order" in file:
            order = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in file["meta/qpos_order"][:]]
            if order != Joint_ORDER: raise ValueError("Snapshot qpos order does not match this glove model.")
    return [{"code": safe_code(index, label), "label": label, "qpos": qpos[index], "world_quat": world_quat[index], "relative_quat": None if relative_quat is None else relative_quat[index], "wrist_world_quat": wrist_world_quat[index], "wrist_world_qpos": wrist_world_qpos[index]} for index, label in enumerate(labels)]


def emit_reference(target: dict) -> None:
    image = target.get("image")
    if image is not None:
        print("CALIBRATION_REFERENCE_IMAGE " + json.dumps({"path": str(image), "label": target["label"]}), flush=True)

def qpos_from_site_quaternions(quaternions: np.ndarray) -> np.ndarray:
    state = {"frame": 0, "site_quats": np.asarray([quaternions], dtype=float)}
    return rel_quat_to_qpos2(rel_quat_compute(state))[WRIST_WORLD_DOF:WRIST_WORLD_DOF + NUM_QPOS]

def reference_target(targets: list[dict]) -> dict:
    return next((target for target in targets if any(word in target["label"].lower() for word in ("reference", "refrence", "neutral", "open hand", "open_hand"))), targets[0])


def render_targets(model, targets: list[dict]) -> None:
    IMAGE_FOLDER.mkdir(exist_ok=True)
    data = mujoco.MjData(model)
    with mujoco.Renderer(model, height=300, width=560) as renderer:
        wrist_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
        wrist_mocap = int(model.body_mocapid[wrist_body])
        for target in targets:
            data.qpos[WRIST_WORLD_DOF:WRIST_WORLD_DOF + NUM_QPOS] = np.clip(target["qpos"], model.jnt_range[WRIST_WORLD_DOF:WRIST_WORLD_DOF + NUM_QPOS, 0], model.jnt_range[WRIST_WORLD_DOF:WRIST_WORLD_DOF + NUM_QPOS, 1])
            if wrist_mocap >= 0: data.mocap_quat[wrist_mocap] = target["wrist_world_quat"]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="render_camera")
            image = renderer.render()
            path = IMAGE_FOLDER / f"{target['code']}.ppm"
            with path.open("wb") as file: file.write(f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii") + np.ascontiguousarray(image).tobytes())
            target["image"] = path


def fit_pose_mapping(targets: list[np.ndarray], measured: list[np.ndarray], pose_masks: np.ndarray | None = None) -> dict:
    matrix, offset = np.eye(len(Joint_ORDER)), np.zeros(len(Joint_ORDER))
    warnings, details = [], {}
    if not targets: return {"matrix": matrix.tolist(), "offset": offset.tolist(), "details": details, "warnings": ["No pose targets were accepted."], "method": "named_pose_snapshots"}
    wanted, observed = np.asarray(targets, dtype=float), np.asarray(measured, dtype=float)
    for block in BLOCKS:
        indices = np.asarray(block)
        valid_rows = np.ones(len(wanted), dtype=bool) if pose_masks is None else np.all(np.asarray(pose_masks)[:, indices], axis=1)
        x, y = observed[valid_rows][:, indices], wanted[valid_rows][:, indices]
        if x.ndim == 1: x, y = x[:, None], y[:, None]
        dimensions = len(indices)
        x_range, y_range = np.ptp(x, axis=0), np.ptp(y, axis=0)
        if len(x) < dimensions + 1 or float(max(np.max(x_range), np.max(y_range))) < np.deg2rad(3.0):
            block_matrix = np.eye(dimensions)
            block_offset = np.mean(y - x, axis=0)
            warnings.append(f"{', '.join(Joint_ORDER[i] for i in indices)} had too little pose variation; only its neutral offset was aligned")
            accepted = np.ones(len(x), dtype=bool)
        else:
            def solve(selected):
                x_fit, y_fit = x[selected], y[selected]
                x_centre, y_centre = np.mean(x_fit, axis=0), np.mean(y_fit, axis=0)
                xc, yc = x_fit - x_centre, y_fit - y_centre
                regularisation = max(1e-7, 0.005 * float(np.trace(xc.T @ xc)) / dimensions)
                fitted = np.linalg.solve(xc.T @ xc + regularisation * np.eye(dimensions), xc.T @ yc + regularisation * np.eye(dimensions)).T
                return fitted, y_centre - fitted @ x_centre
            accepted = np.ones(len(x), dtype=bool)
            block_matrix, block_offset = solve(accepted)

            singular = np.linalg.svd(block_matrix, compute_uv=False)
            if np.any(singular < 0.25) or np.any(singular > 3.0):
                warnings.append(f"{', '.join(Joint_ORDER[i] for i in indices)} produced an extreme scale; add poses that move this joint through more of its range")
        matrix[np.ix_(indices, indices)] = block_matrix
        offset[indices] = block_offset
        prediction = x @ block_matrix.T + block_offset
        details["|".join(Joint_ORDER[i] for i in indices)] = {"rms_error_deg": float(np.rad2deg(np.sqrt(np.mean((prediction[accepted] - y[accepted]) ** 2)))), "used_poses": int(np.count_nonzero(accepted)), "rejected_poses": int(len(accepted) - np.count_nonzero(accepted)), "measured_range_deg": np.rad2deg(x_range).tolist(), "target_range_deg": np.rad2deg(y_range).tolist()}
    return {"matrix": matrix.tolist(), "offset": offset.tolist(), "details": details, "warnings": warnings, "method": "named_pose_snapshots"}


def fit_sensor_primary_mapping(targets: list[dict], measured: list[np.ndarray], pose_masks: np.ndarray | None = None) -> dict:
    matrix, offset = np.eye(len(Joint_ORDER)), np.zeros(len(Joint_ORDER))
    warnings, details = [], {}
    if not targets: return {"matrix": matrix.tolist(), "offset": offset.tolist(), "details": details, "warnings": ["No pose targets were accepted."], "method": "sensor_primary_reference_offsets"}
    wanted, observed = np.asarray([target["qpos"] for target in targets], dtype=float), np.asarray(measured, dtype=float)
    reference = np.asarray([any(word in f"{target.get('code', '')} {target.get('label', '')}".lower() for word in ("reference", "refrence", "neutral", "open hand", "open_hand")) for target in targets], dtype=bool)
    selected = reference if np.count_nonzero(reference) else np.ones(len(targets), dtype=bool)
    for index in range(len(Joint_ORDER)):
        valid = selected if pose_masks is None else (selected & np.asarray(pose_masks)[:, index])
        if np.any(valid): offset[index] = np.median(wanted[valid, index] - observed[valid, index])
    prediction = observed + offset
    for block in BLOCKS:
        indices = np.asarray(block)
        valid_rows = np.ones(len(wanted), dtype=bool) if pose_masks is None else np.all(np.asarray(pose_masks)[:, indices], axis=1)
        error = prediction[valid_rows][:, indices] - wanted[valid_rows][:, indices]
        details["|".join(Joint_ORDER[i] for i in indices)] = {"rms_error_deg": float(np.rad2deg(np.sqrt(np.mean(error ** 2)))), "used_poses": int(len(targets)), "rejected_poses": 0, "measured_range_deg": np.rad2deg(np.ptp(observed[:, indices], axis=0)).tolist(), "target_range_deg": np.rad2deg(np.ptp(wanted[:, indices], axis=0)).tolist()}
    details["reference_offset_source"] = {"reference_poses": int(np.count_nonzero(reference)), "total_poses": int(len(targets))}
    if not np.count_nonzero(reference): warnings.append("No reference-labelled pose was found; sensor-primary offsets used all poses.")
    return {"matrix": matrix.tolist(), "offset": offset.tolist(), "details": details, "warnings": warnings, "method": "sensor_primary_reference_offsets"}


def add_secondary_pose_fit(sensor_mapping: dict, pose_mapping: dict) -> dict:
    matrix = np.asarray(sensor_mapping["matrix"], dtype=float).copy()
    offset = np.asarray(sensor_mapping["offset"], dtype=float).copy()
    pose_matrix = np.asarray(pose_mapping["matrix"], dtype=float)
    pose_offset = np.asarray(pose_mapping["offset"], dtype=float)
    applied, skipped, abduction_applied, abduction_skipped = [], [], [], []
    for index in FLEXION_JOINTS:
        gain = float(pose_matrix[index, index])
        if np.isfinite(gain):
            matrix[index] = 0.0
            matrix[index, index] = gain
            offset[index] = pose_offset[index]
            applied.append({"joint": Joint_ORDER[index], "gain": gain, "offset_deg": float(np.rad2deg(offset[index]))})
        else:
            skipped.append({"joint": Joint_ORDER[index], "gain": gain})
    for index in ABDUCTION_JOINTS:
        pose_gain = float(pose_matrix[index, index])
        if np.isfinite(pose_gain):
            raw_gain = float(matrix[index, index])
            blended_gain = 1.0
            blended_offset = 0.0
            matrix[index] = 0.0
            matrix[index, index] = blended_gain
            offset[index] = blended_offset
            abduction_applied.append({"joint": Joint_ORDER[index], "raw_gain": raw_gain, "pose_gain": pose_gain, "blended_gain": blended_gain, "offset_deg": float(np.rad2deg(blended_offset)), "mode": "reference_offset_only"})
        else:
            abduction_skipped.append({"joint": Joint_ORDER[index], "gain": pose_gain})
    combined = dict(sensor_mapping)
    combined["matrix"] = matrix.tolist()
    combined["offset"] = offset.tolist()
    combined["method"] = "sensor_primary_plus_flexion_pose_fit_and_abduction_reference_offset"
    combined["secondary_joint_fit"] = {"mode": "flexion_pose_fit_and_abduction_reference_offset_only", "applied": applied, "skipped": skipped, "abduction_applied": abduction_applied, "abduction_skipped": abduction_skipped, "abduction_pose_blend": 0.0, "gain_limits": [FLEXION_MIN_GAIN, FLEXION_MAX_GAIN]}
    if skipped: combined["warnings"] = [*combined.get("warnings", []), "Flexion pose fit skipped extreme gain for: " + ", ".join(item["joint"] for item in skipped)]
    return combined
def fit_pose_calibration(targets: list[dict], measurements: list[dict], enabled_imus: list[str], locked_qpos: list[int] | None = None) -> dict:
    if not targets: return fit_pose_mapping([], [])
    locked_qpos = locked_qpos or []
    alignment = fit_sensor_alignment(np.asarray([target["world_quat"] for target in targets]), np.asarray([measurement["live_filter_quat"] for measurement in measurements]), enabled_imus)
    measured_qpos = [qpos_from_site_quaternions(apply_sensor_alignment(np.asarray(measurement["live_filter_quat"]), alignment)) for measurement in measurements]
    fit_targets = []
    fit_measured = []
    pose_masks = []
    for target, measured in zip(targets, measured_qpos):
        target_qpos = np.asarray(target["qpos"], dtype=float).copy()
        measured_copy = np.asarray(measured, dtype=float).copy()
        if locked_qpos:
            target_qpos[locked_qpos] = 0.0
            measured_copy[locked_qpos] = 0.0
        fit_targets.append(target_qpos)
        fit_measured.append(measured_copy)
        pose_name = f"{target.get('code', '')} {target.get('label', '')}".lower()
        is_finger_flex = "flex_fingers" in pose_name
        mask = np.ones(len(Joint_ORDER), dtype=bool)
        if is_finger_flex:
            mask[:] = False
            
            mask[[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]] = True
        pose_masks.append(mask)
    pose_masks = np.asarray(pose_masks, dtype=bool)
    full_joint_mapping = fit_pose_mapping(fit_targets, fit_measured, pose_masks)
    
    if len(fit_targets) >= 2:
        strict_matrix = np.zeros((len(Joint_ORDER), len(Joint_ORDER)), dtype=float)
        strict_offset = np.zeros(len(Joint_ORDER), dtype=float)
        reference_index = next((pose_index for pose_index, target in enumerate(targets) if any(word in f"{target.get('code', '')} {target.get('label', '')}".lower() for word in ("reference", "refrence", "neutral", "open hand", "open_hand"))), 0)
        for index in range(len(Joint_ORDER)):
            valid_rows = np.asarray(pose_masks)[:, index]
            measured = np.asarray([row[index] for row, valid in zip(fit_measured, valid_rows) if valid], dtype=float)
            target = np.asarray([row[index] for row, valid in zip(fit_targets, valid_rows) if valid], dtype=float)
            if len(measured) == 0:
                strict_matrix[index, index] = 1.0
                continue
            design = np.column_stack((measured, np.ones(len(measured))))
            gain, offset = np.linalg.lstsq(design, target, rcond=None)[0]
            strict_matrix[index, index] = float(gain)
            strict_offset[index] = float(offset)
        full_joint_mapping["matrix"] = strict_matrix.tolist()
        full_joint_mapping["offset"] = strict_offset.tolist()
        full_joint_mapping["method"] = "strict_multi_pose_diagonal_qpos_fit_all_joints"
    sensor_mapping = fit_sensor_primary_mapping([{**target, "qpos": fit_targets[index]} for index, target in enumerate(targets)], fit_measured, pose_masks)
    
    reference_index = next((pose_index for pose_index, target in enumerate(targets) if any(word in f"{target.get('code', '')} {target.get('label', '')}".lower() for word in ("reference", "refrence", "neutral", "open hand", "open_hand"))), 0)
    mapping = add_secondary_pose_fit(sensor_mapping, full_joint_mapping)
    mapping = apply_calibration_locks(mapping, locked_qpos, "pose snapshot calibration")
    
    mapping["joint_output_fit"] = {"details": full_joint_mapping.get("details", {}), "method": full_joint_mapping.get("method", "named_pose_snapshots")}
    mapping["sensor_alignment"] = alignment
    mapping["measured_qpos"] = [value.tolist() for value in measured_qpos]
    mapping["piecewise_mapping"] = {}
    mapping["palm_wrist_calibration"] = palm_wrist_calibration_report(mapping, alignment)
    poor = [site for site, detail in alignment["details"].items() if detail["used_poses"] and detail["rms_error_deg"] > 12.0]
    mapping["alignment_valid"] = not any(detail["used_poses"] and detail["rms_error_deg"] > 25.0 for detail in alignment["details"].values())
    if poor: mapping["warnings"].append("Sensor-to-segment fit exceeds 12 degrees for: " + ", ".join(poor))
    if mapping["palm_wrist_calibration"]["status"] != "PASS": mapping["warnings"].append(mapping["palm_wrist_calibration"]["summary"])
    return mapping


def relative_from_site_quaternions(quaternions: np.ndarray) -> np.ndarray:
    return rel_quat_compute({"frame": 0, "site_quats": np.asarray([quaternions], dtype=float)})


def palm_wrist_calibration_report(mapping: dict, alignment: dict) -> dict:
    details = mapping.get("details", {}).get(PALM_WRIST_BLOCK, {})
    target_range = np.asarray(details.get("target_range_deg", [0.0, 0.0, 0.0]), dtype=float)
    measured_range = np.asarray(details.get("measured_range_deg", [0.0, 0.0, 0.0]), dtype=float)
    coverage = target_range >= PALM_WRIST_MIN_RANGE_DEG
    wrist_rms = float(alignment.get("details", {}).get("imu_wrist", {}).get("rms_error_deg", np.inf))
    palm_rms = float(alignment.get("details", {}).get("imu_palm", {}).get("rms_error_deg", np.inf))
    joint_rms = float(details.get("rms_error_deg", np.inf))
    passed = bool(np.all(coverage) and max(wrist_rms, palm_rms, joint_rms) <= 12.0)
    axis_text = ", ".join(f"{axis} {value:.1f}°{' OK' if ok else ' LOW'}" for axis, value, ok in zip(("roll", "pitch", "yaw"), target_range, coverage))
    summary = f"Palm/Wrist {'PASS' if passed else 'WARN'} — {axis_text}; fit RMS wrist {wrist_rms:.1f}°, palm {palm_rms:.1f}°, joints {joint_rms:.1f}°."
    return {"status": "PASS" if passed else "WARN", "summary": summary, "target_range_deg": target_range.tolist(), "measured_range_deg": measured_range.tolist(), "minimum_range_deg": PALM_WRIST_MIN_RANGE_DEG.tolist(), "axis_coverage": coverage.tolist(), "wrist_alignment_rms_deg": wrist_rms, "palm_alignment_rms_deg": palm_rms, "joint_mapping_rms_deg": joint_rms}



def collect_pose_measurement(receiver, exclusions, preview, code: str, seconds: float) -> dict:
    while receiver.get_nowait() is not None: pass
    required = max(50, int(round(seconds * SAMPLE_RATE_HZ)))
    quaternions, acceleration, angular_rate, last_progress = [], [], [], -1
    while len(quaternions) < required:
        frame = next_valid_frame(receiver, exclusions, preview)
        accel, gyro = frame_to_arrays(frame, exclusions)
        acceleration.append(accel); angular_rate.append(gyro)
        quaternions.append(preview.quaternions.copy())
        progress = int(100 * len(quaternions) / required)
        if progress != last_progress: print(f"CALIBRATION_STEP_PROGRESS {code} {progress} {len(quaternions)} {required}", flush=True); last_progress = progress
    samples = np.asarray(quaternions)
    mean_quaternions, dispersion = np.zeros((NUM_IMUS, 4)), np.zeros(NUM_IMUS)
    for index in range(NUM_IMUS):
        mean_quaternions[index] = quaternion_mean(samples[:, index])
        dots = np.clip(np.abs(samples[:, index] @ mean_quaternions[index]), 0.0, 1.0)
        dispersion[index] = np.rad2deg(np.median(2.0 * np.arccos(dots)))
    return {"live_filter_quat": mean_quaternions, "hold_dispersion_deg": dispersion, "mean_accel_m_s2": np.median(np.asarray(acceleration), axis=0), "mean_gyro_rad_s": np.median(np.asarray(angular_rate), axis=0)}


def capture_target(receiver, exclusions, preview, commands, target: dict, allow_back: bool, accepted_targets: list[dict], accepted_measurements: list[dict], enabled_imus: list[str], calibration_locked_qpos: list[int]) -> dict | None:
    while True:
        emit_reference(target)
        instruction = f"Match the pictured pose: {target['label']}. Hold it comfortably still, then start the capture."
        if not wait_for_start(receiver, preview, commands, target["code"], instruction, POSE_SECONDS, skippable=True, allow_back=allow_back): return None
        measured = collect_pose_measurement(receiver, exclusions, preview, target["code"], POSE_SECONDS)
        provisional = fit_pose_calibration([*accepted_targets, target], [*accepted_measurements, measured], enabled_imus, calibration_locked_qpos)
        preview.set_sensor_alignment(provisional["sensor_alignment"])
        preview.set_mapping(provisional)
        if review_step(receiver, preview, commands, target["code"], "The new correction is active in the live viewer. Compare it with the reference picture, then accept/next or redo this pose."): return measured
        previous = fit_pose_calibration(accepted_targets, accepted_measurements, enabled_imus, calibration_locked_qpos) if accepted_targets else None
        preview.set_sensor_alignment(previous.get("sensor_alignment") if previous else None)
        preview.set_mapping(previous)
        print(f"CALIBRATION_STEP_REDO {target['code']}", flush=True)


def strict_pose_targets(targets: list[dict]) -> list[dict]:
    """Use every pose in the selected snapshot file as a strict qpos anchor."""
    if not targets: return []
    reference = reference_target(targets)
    return [reference] + [target for target in targets if target["code"] != reference["code"]]

def save_result(calibration: CalibrationData, targets: list[dict], measured_by_code: dict[str, dict | None], mapping: dict, source: Path) -> None:
    alignment = mapping["sensor_alignment"]
    paired = []
    for target in targets:
        measurement = measured_by_code.get(target["code"])
        measured_qpos = None if measurement is None else qpos_from_site_quaternions(apply_sensor_alignment(np.asarray(measurement["live_filter_quat"]), alignment)).tolist()
        aligned = None if measurement is None else apply_sensor_alignment(np.asarray(measurement["live_filter_quat"]), alignment)
        paired.append({"code": target["code"], "label": target["label"], "target_qpos": target["qpos"].tolist(), "target_wrist_world_qpos": target["wrist_world_qpos"].tolist(), "target_wrist_world_quat": target["wrist_world_quat"].tolist(), "target_world_quat": target["world_quat"].tolist(), "target_relative_quat": None if target["relative_quat"] is None else target["relative_quat"].tolist(), "measured_qpos": measured_qpos, "live_filter_quat": None if measurement is None else measurement["live_filter_quat"].tolist(), "aligned_site_quat": None if aligned is None else aligned.tolist(), "measured_relative_quat": None if aligned is None else relative_from_site_quaternions(aligned).tolist(), "hold_dispersion_deg": None if measurement is None else measurement["hold_dispersion_deg"].tolist(), "mean_accel_m_s2": None if measurement is None else measurement["mean_accel_m_s2"].tolist(), "mean_gyro_rad_s": None if measurement is None else measurement["mean_gyro_rad_s"].tolist()})
    calibration.guided_imu = {"sample_rate_hz": SAMPLE_RATE_HZ, "neutral_mean_accel_m_s2": calibration.mean_accel.tolist(), "neutral_mean_gyro_rad_s": calibration.gyro_bias.tolist(), "neutral_qpos": reference_target(targets)["qpos"].tolist(), "flexion": {"pose_snapshot_fit": True}, "abduction": {"pose_snapshot_fit": True}, "ranges": mapping["details"], "locked_qpos": mapping.get("locked_qpos", []), "sensor_to_segment_alignment": alignment, "stage1_endpoint_mapping": mapping, "pose_snapshot_calibration": {"source_file": str(source), "poses": paired}}
    calibration.save(CALIBRATION_FILE)


def run(snapshot_file: Path, duration: float, gain: float, use_gyro_bias: bool) -> None:
    axis_controls_for_locks = AxisControls(AXIS_CONTROL_FILE)
    display_controls_for_locks = DisplayControls(DISPLAY_CONTROL_FILE)
    calibration_locked_qpos = locked_qpos_indices(axis_controls_for_locks.values, display_controls_for_locks.disable_abduction)
    all_targets = load_targets(snapshot_file)
    targets = strict_pose_targets(all_targets)
    model = mujoco.MjModel.from_xml_path(str(PROJECT_FOLDER / MODEL_PATH))
    render_targets(model, targets)
    plan = [{"code": "stationary", "name": "Stationary bias and gravity hold"}, *[{"code": target["code"], "name": target["label"]} for target in targets]]
    print("CALIBRATION_PLAN " + json.dumps(plan), flush=True)
    print("CALIBRATION_STAGE1_BEGIN Strict reference and first-pose qpos calibration.", flush=True)
    exclusions, receiver, commands = read_disabled(), LiveIMUSerial(), queue.Queue()
    enabled_imus = [site for site in IMU_ORDER if site not in exclusions]
    receiver.start()
    threading.Thread(target=command_reader, args=(commands,), daemon=True).start()
    calibration, measured_by_code = None, {}
    try:
        data = mujoco.MjData(model)
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "render_camera")
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            hwnd = snap_viewer_right()
            if hwnd is not None: print(f"MUJOCO_HWND {hwnd}", flush=True)
            preview = Stage1Preview(viewer, model, data, exclusions, gain, use_gyro_bias)
            emit_reference(reference_target(targets))
            while True:
                calibration = collect_stationary(receiver, exclusions, preview, commands, duration)
                preview.configure(calibration)
                if review_step(receiver, preview, commands, "stationary", "Bias and gravity captured. Accept or redo the stationary hold."): break
            accepted_targets, accepted_measurements = [], []
            target_index = 0
            while target_index < len(targets):
                target = targets[target_index]
                try:
                    measured_by_code[target["code"]] = capture_target(receiver, exclusions, preview, commands, target, target_index > 0, accepted_targets, accepted_measurements, enabled_imus, calibration_locked_qpos)
                except GoBackRequested:
                    previous_index = max(0, target_index - 1)
                    if accepted_targets:
                        previous_code = accepted_targets[-1]["code"]
                        measured_by_code.pop(previous_code, None)
                        accepted_targets.pop()
                        accepted_measurements.pop()
                    target_index = previous_index
                    print(f"CALIBRATION_WENT_BACK {target['code']}", flush=True)
                    continue
                if measured_by_code[target["code"]] is not None:
                    accepted_targets.append(target); accepted_measurements.append(measured_by_code[target["code"]])
                target_index += 1
            mapping = fit_pose_calibration(accepted_targets, accepted_measurements, enabled_imus, calibration_locked_qpos)
            preview.set_sensor_alignment(mapping["sensor_alignment"]); preview.set_mapping(mapping)
            preview.set_mapping(mapping)
            for warning in mapping.get("warnings", []): print(f"CALIBRATION_MAPPING_WARNING {warning}", flush=True)
            print(f"CALIBRATION_STAGE1_FINAL_REVIEW {len(accepted_targets)} strict qpos poses fitted. Inspect the live viewer, then save.", flush=True)
            wait_for_command(receiver, preview, commands, {"ACCEPT_STAGE1"})
            save_result(calibration, targets, measured_by_code, mapping, snapshot_file)
            print(f"CALIBRATION_COMPLETE {CALIBRATION_FILE}", flush=True)
            return
    finally: receiver.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--filter", default="Madgwick", help=argparse.SUPPRESS)
    parser.add_argument("--gain", type=float, default=0.1)
    parser.add_argument("--use-gyro-bias", action=argparse.BooleanOptionalAction, default=True)
    arguments = parser.parse_args()
    try: run(arguments.snapshots.resolve(), arguments.duration, arguments.gain, arguments.use_gyro_bias)
    except KeyboardInterrupt: print("CALIBRATION_CANCELLED", flush=True)
    except Exception as error: print(f"CALIBRATION_ERROR {type(error).__name__}: {error}", flush=True); raise


































