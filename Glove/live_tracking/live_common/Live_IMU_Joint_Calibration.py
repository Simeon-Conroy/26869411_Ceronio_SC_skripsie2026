"""Stage 1 endpoint calibration and joint-output correction."""
from __future__ import annotations
import numpy as np
from common.Const_INFO import NUM_QPOS, Joint_ORDER

PROXIMAL_QPOS = {"thumb": (5, 6), "index": (8, 9), "middle": (12, 13), "ring": (16, 17), "pinky": (20, 21)}
HINGE_QPOS = {"thumb": (7,), "index": (10, 11), "middle": (14, 15), "ring": (18, 19), "pinky": (22, 23)}


def fit_stage1_endpoint_mapping(straight: dict[str, np.ndarray | None], flexed: dict[str, np.ndarray | None], abduction_left: dict[str, np.ndarray | None], abduction_neutral: dict[str, np.ndarray | None], abduction_right: dict[str, np.ndarray | None], joint_ranges: np.ndarray) -> dict:
    matrix = np.eye(NUM_QPOS)
    offset = np.zeros(NUM_QPOS)
    details, warnings = {}, []
    for finger, proximal_indices in PROXIMAL_QPOS.items():
        indices = list(proximal_indices)
        flex_available = straight.get(finger) is not None and flexed.get(finger) is not None
        flex_delta = flexed[finger][indices] - straight[finger][indices] if flex_available else np.zeros(2)
        flex_target = float(joint_ranges[indices[0], 1])
        correction = np.eye(2)
        calibrated_rows = np.zeros(2, dtype=bool)
        abduction_available = abduction_left.get(finger) is not None and abduction_right.get(finger) is not None
        abduction_delta = abduction_right[finger][indices] - abduction_left[finger][indices] if abduction_available else np.zeros(2)
        if flex_available and abduction_available and np.linalg.norm(flex_delta) >= np.deg2rad(5.0) and np.linalg.norm(abduction_delta) >= np.deg2rad(3.0):
            measured = np.column_stack((flex_delta, abduction_delta))
            target = np.array([[flex_target, 0.0], [0.0, float(joint_ranges[indices[1], 1] - joint_ranges[indices[1], 0])]])
            condition = float(np.linalg.cond(measured))
            if condition < 12.0:
                correction = target @ np.linalg.inv(measured)
                calibrated_rows[:] = True
            else:
                correction[0] = flex_target * flex_delta / np.dot(flex_delta, flex_delta)
                calibrated_rows[0] = True
                warnings.append(f"{finger} proximal abduction was skipped because flexion and abduction measurements were too similar; flexion calibration still applied")
        elif flex_available and np.linalg.norm(flex_delta) >= np.deg2rad(5.0):
            correction[0] = flex_target * flex_delta / np.dot(flex_delta, flex_delta)
            calibrated_rows[0] = True
            warnings.append(f"{finger} proximal abduction was skipped or too small; flexion calibration still applied")
        else:
            warnings.append(f"{finger} proximal flexion was skipped or too small; its raw joint output will be left unchanged")
        matrix[np.ix_(indices, indices)] = correction
        baseline_pose = abduction_neutral.get(finger) if abduction_neutral.get(finger) is not None else straight.get(finger)
        baseline = baseline_pose[indices] if baseline_pose is not None else np.zeros(2)
        calibrated_offset = -correction @ baseline
        offset[np.asarray(indices)[calibrated_rows]] = calibrated_offset[calibrated_rows]
        details[finger] = {"proximal_condition": float(np.linalg.cond(np.column_stack((flex_delta, abduction_delta)))) if flex_available and abduction_available and np.linalg.norm(flex_delta) > 0 and np.linalg.norm(abduction_delta) > 0 else float("inf"), "flex_delta_deg": np.rad2deg(flex_delta).tolist(), "abduction_delta_deg": np.rad2deg(abduction_delta).tolist(), "abduction_calibration": "enabled" if calibrated_rows[1] else "skipped_or_raw"}
        for qpos_index in HINGE_QPOS[finger]:
            delta = float(flexed[finger][qpos_index] - straight[finger][qpos_index]) if flex_available else 0.0
            if not flex_available or abs(delta) < np.deg2rad(5.0):
                details[finger][f"qpos_{qpos_index}_delta_deg"] = float(np.rad2deg(delta))
                warnings.append(f"{finger} joint {qpos_index} was skipped or changed by only {np.rad2deg(abs(delta)):.1f} degrees; its raw joint output will be left unchanged")
                continue
            target = float(joint_ranges[qpos_index, 1])
            scale = target / delta
            matrix[qpos_index] = 0.0
            matrix[qpos_index, qpos_index] = scale
            offset[qpos_index] = -scale * straight[finger][qpos_index]
            details[finger][f"qpos_{qpos_index}_delta_deg"] = float(np.rad2deg(delta))
    return {"matrix": matrix.tolist(), "offset": offset.tolist(), "details": details, "warnings": warnings, "method": "static_endpoint_holds"}


def apply_calibration_locks(mapping: dict, locked_qpos: list[int] | tuple[int, ...] | None, reason: str = "display locks") -> dict:
    if not locked_qpos: return mapping
    matrix = np.asarray(mapping.get("matrix", np.eye(NUM_QPOS)), dtype=float).copy()
    offset = np.asarray(mapping.get("offset", np.zeros(NUM_QPOS)), dtype=float).copy()
    locked = sorted({int(index) for index in locked_qpos if 0 <= int(index) < NUM_QPOS})
    for index in locked:
        matrix[index] = 0.0
        offset[index] = 0.0
    output = dict(mapping)
    output["matrix"] = matrix.tolist()
    output["offset"] = offset.tolist()
    output["locked_qpos"] = locked
    names = ", ".join(str(index) for index in locked)
    warning = f"Strict display lock active: qpos {names} forced to zero and excluded from calibration mapping ({reason})."
    output["warnings"] = [*output.get("warnings", []), warning]
    return output


def apply_joint_mapping(qpos: np.ndarray, calibration, joint_ranges: np.ndarray | None = None) -> np.ndarray:
    saved = calibration.guided_imu.get("stage1_endpoint_mapping", calibration.guided_imu.get("stage1_qpos_mapping", {})) if calibration.guided_imu else {}
    if not saved: return qpos
    output = np.asarray(saved["matrix"], dtype=float) @ np.asarray(qpos, dtype=float) + np.asarray(saved["offset"], dtype=float)
    curves = saved.get("piecewise_mapping", {})
    for index, name in enumerate(Joint_ORDER):
        curve = curves.get(name)
        if curve and len(curve.get("measured", [])) >= 2:
            output[index] = float(np.interp(float(qpos[index]), curve["measured"], curve["target"]))
    for index in saved.get("locked_qpos", []): output[int(index)] = 0.0
    if joint_ranges is not None: output = np.clip(output, joint_ranges[:, 0], joint_ranges[:, 1])
    return output
