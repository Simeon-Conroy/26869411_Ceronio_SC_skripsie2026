"""Capture single-frame MuJoCo pose snapshots for later calibration."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
PROJECT_FOLDER = Path(__file__).resolve().parents[2]
LIVE_FOLDER = Path(__file__).resolve().parents[1]
if str(PROJECT_FOLDER) not in sys.path: sys.path.insert(0, str(PROJECT_FOLDER))

import h5py
import mujoco
import mujoco.viewer
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from scipy.spatial.transform import Rotation as Rot

from common.Const_INFO import IMU_ORDER, Joint_ORDER, MODEL_PATH, SITE_PAIRS
from common.H5_file_Handling import Save_File_H5
from common.MJ_utils import discover_sensors

DEFAULT_MODEL = PROJECT_FOLDER / MODEL_PATH
WRIST_WORLD_JOINTS = ["wrist_world_roll", "wrist_world_pitch", "wrist_world_yaw"]
WRIST_WORLD_ORDER = ["Wrist_World_Roll", "Wrist_World_Pitch", "Wrist_World_Yaw"]
WRIST_WORLD_DOF = 3


def scipy_to_mujoco(quaternion_xyzw): return np.asarray([quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]], dtype=np.float64)


def site_rotation(model, data, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0: raise KeyError(f"MuJoCo site not found: {site_name}")
    return Rot.from_matrix(np.asarray(data.site_xmat[site_id]).reshape(3, 3))


def capture_snapshot(model, data, accel_lookup, gyro_lookup):
    mujoco.mj_forward(model, data)
    accel = np.zeros((len(IMU_ORDER), 3))
    gyro = np.zeros((len(IMU_ORDER), 3))
    world_pos = np.zeros((len(IMU_ORDER), 3))
    world_quat = np.zeros((len(IMU_ORDER), 4))
    relative_quat = np.zeros((len(SITE_PAIRS), 4))
    for index, site in enumerate(IMU_ORDER):
        accel_address, gyro_address = accel_lookup[site]["adr"], gyro_lookup[site]["adr"]
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        accel[index] = data.sensordata[accel_address:accel_address + 3]
        gyro[index] = data.sensordata[gyro_address:gyro_address + 3]
        world_pos[index] = data.site_xpos[site_id]
        world_quat[index] = scipy_to_mujoco(site_rotation(model, data, site).as_quat())
    for index, (child, parent) in enumerate(SITE_PAIRS): relative_quat[index] = scipy_to_mujoco((site_rotation(model, data, parent).inv() * site_rotation(model, data, child)).as_quat())
    wrist_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
    return {"captured_at": datetime.now().astimezone().isoformat(), "sim_time": float(data.time), "world_pos": world_pos, "world_quat": world_quat, "accel": accel, "gyro": gyro, "relative_quat": relative_quat, "wrist_world_qpos": np.asarray(data.qpos[:WRIST_WORLD_DOF]).copy(), "wrist_world_quat": np.asarray(data.xquat[wrist_body]).copy(), "qpos": np.asarray(data.qpos[WRIST_WORLD_DOF:WRIST_WORLD_DOF + len(Joint_ORDER)]).copy(), "qvel": np.asarray(data.qvel[WRIST_WORLD_DOF:WRIST_WORLD_DOF + len(Joint_ORDER)]).copy(), "qacc": np.asarray(data.qacc[WRIST_WORLD_DOF:WRIST_WORLD_DOF + len(Joint_ORDER)]).copy(), "sensordata": np.asarray(data.sensordata).copy()}


def save_snapshots(snapshots, model_path, timestep, filename=None):
    filename = filename or Save_File_H5()
    if not filename:
        print("Save cancelled. Snapshots are still held in memory.")
        return
    path = Path(filename)
    if path.suffix.lower() not in {".h5", ".hdf5"}: path = path.with_suffix(".h5")
    with h5py.File(path, "w") as file:
        meta = file.create_group("meta")
        meta.attrs["format"] = "Sharpa pose snapshots"
        meta.attrs["format_version"] = 2
        meta.attrs["created_at"] = datetime.now().astimezone().isoformat()
        meta.attrs["model_path"] = str(model_path)
        meta.attrs["timestep"] = timestep
        meta.attrs["snapshot_count"] = len(snapshots)
        meta.attrs["units_accel"] = "m/s^2"
        meta.attrs["units_gyro"] = "rad/s"
        meta.attrs["quaternion_order"] = "w,x,y,z"
        meta.create_dataset("imu_order", data=np.asarray(IMU_ORDER, dtype="S"))
        meta.create_dataset("relative_pairs", data=np.asarray([f"{child}|{parent}" for child, parent in SITE_PAIRS], dtype="S"))
        meta.create_dataset("qpos_order", data=np.asarray(Joint_ORDER, dtype="S"))
        meta.create_dataset("wrist_world_order", data=np.asarray(WRIST_WORLD_ORDER, dtype="S"))
        group = file.create_group("snapshots")
        group.create_dataset("label", data=np.asarray([snapshot["label"] for snapshot in snapshots], dtype=h5py.string_dtype("utf-8")))
        group.create_dataset("captured_at", data=np.asarray([snapshot["captured_at"] for snapshot in snapshots], dtype=h5py.string_dtype("utf-8")))
        group.create_dataset("sim_time", data=np.asarray([snapshot["sim_time"] for snapshot in snapshots]))
        for key in ("world_pos", "world_quat", "accel", "gyro", "relative_quat", "wrist_world_qpos", "wrist_world_quat", "qpos", "qvel", "qacc", "sensordata"): group.create_dataset(key, data=np.asarray([snapshot[key] for snapshot in snapshots]), compression="gzip")
    print(f"Saved {len(snapshots)} snapshots to:\n{path}\n")


def append_snapshots(snapshots, filename=None):
    if filename is None:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        filename = filedialog.askopenfilename(title="Choose snapshot H5 to append to", filetypes=[("HDF5 snapshot files", "*.h5 *.hdf5")], parent=root)
        root.destroy()
    if not filename:
        print("Append cancelled. Snapshots are still held in memory.")
        return
    path = Path(filename)
    temporary_path = path.with_name(f".{path.stem}.append_tmp{path.suffix}")
    try:
        shutil.copy2(path, temporary_path)
        with h5py.File(temporary_path, "r+") as file:
            if "meta" not in file or "snapshots" not in file: raise ValueError("the selected file is not a pose-snapshot recording")
            meta, group = file["meta"], file["snapshots"]
            required = ("label", "captured_at", "sim_time", "world_pos", "world_quat", "accel", "gyro", "relative_quat", "qpos", "qvel", "qacc", "sensordata")
            missing = [key for key in required if key not in group]
            if missing: raise ValueError(f"missing snapshot data: {', '.join(missing)}")
            stored_imu_order = [value.decode() if isinstance(value, bytes) else str(value) for value in meta["imu_order"][:]]
            stored_qpos_order = [value.decode() if isinstance(value, bytes) else str(value) for value in meta["qpos_order"][:]]
            stored_pairs = [value.decode() if isinstance(value, bytes) else str(value) for value in meta["relative_pairs"][:]]
            if stored_imu_order != list(IMU_ORDER): raise ValueError("IMU order differs from the current recorder")
            if stored_qpos_order != list(Joint_ORDER): raise ValueError("joint order differs from the current recorder")
            if stored_pairs != [f"{child}|{parent}" for child, parent in SITE_PAIRS]: raise ValueError("relative-quaternion order differs from the current recorder")
            old_count = len(group["label"])
            combined = {}
            for key in required:
                old_values = group[key][:]
                new_values = np.asarray([snapshot[key] for snapshot in snapshots], dtype=h5py.string_dtype("utf-8")) if key in {"label", "captured_at"} else np.asarray([snapshot[key] for snapshot in snapshots])
                combined[key] = np.concatenate((old_values, new_values), axis=0)
            old_wrist_qpos = group["wrist_world_qpos"][:] if "wrist_world_qpos" in group else np.zeros((old_count, WRIST_WORLD_DOF))
            old_wrist_quat = group["wrist_world_quat"][:] if "wrist_world_quat" in group else np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (old_count, 1))
            combined["wrist_world_qpos"] = np.concatenate((old_wrist_qpos, np.asarray([snapshot["wrist_world_qpos"] for snapshot in snapshots])), axis=0)
            combined["wrist_world_quat"] = np.concatenate((old_wrist_quat, np.asarray([snapshot["wrist_world_quat"] for snapshot in snapshots])), axis=0)
            for key in (*required, "wrist_world_qpos", "wrist_world_quat"):
                if key in group: del group[key]
            group.create_dataset("label", data=combined["label"], dtype=h5py.string_dtype("utf-8"))
            group.create_dataset("captured_at", data=combined["captured_at"], dtype=h5py.string_dtype("utf-8"))
            group.create_dataset("sim_time", data=combined["sim_time"])
            for key in ("world_pos", "world_quat", "accel", "gyro", "relative_quat", "wrist_world_qpos", "wrist_world_quat", "qpos", "qvel", "qacc", "sensordata"): group.create_dataset(key, data=combined[key], compression="gzip")
            if "wrist_world_order" not in meta: meta.create_dataset("wrist_world_order", data=np.asarray(WRIST_WORLD_ORDER, dtype="S"))
            meta.attrs["format_version"] = 2
            meta.attrs["snapshot_count"] = old_count + len(snapshots)
            meta.attrs["modified_at"] = datetime.now().astimezone().isoformat()
        os.replace(temporary_path, path)
        print(f"Appended {len(snapshots)} snapshots to {path.name} — the file now contains {old_count + len(snapshots)} snapshots.\n")
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showerror("Could not append snapshots", f"{path.name}\n\n{error}", parent=root)
        root.destroy()
        print(f"Append failed: {error}. Snapshots are still held in memory.")


def print_controls():
    print("\nPOSE SNAPSHOT CONTROLS — press these keys inside the MuJoCo viewer")
    print("-----------------------------------------------------------------")
    print("C : Name and capture the current hand pose")
    print("U : Undo the most recent snapshot")
    print("S : Save the held snapshots as a new HDF5 file")
    print("A : Append the held snapshots to an existing snapshot HDF5 file")
    print("Q : Quit")
    print("P / N : Select previous / next joint")
    print(", / . : Decrease / increase the selected joint")
    print("1 / 2 / 3 : Set adjustment size to 0.1 / 1 / 5 degrees")
    print("E : Enter an exact angle for the selected joint")
    print("Z : Set the selected joint target to zero degrees")
    print("V : Print all targets: 3 Wrist-to-World + 24 Palm/finger joints")
    print("-----------------------------------------------------------------")
    print("Use the degree-based Pose Snapshot Controller or the MuJoCo controls before capturing.\n")


def ask_snapshot_name(default_name):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    name = simpledialog.askstring("Snapshot name", "Enter a descriptive name for this pose:", initialvalue=default_name, parent=root)
    root.destroy()
    return name.strip() if name and name.strip() else None


def ask_joint_angle(joint_name, current_degrees):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    angle = simpledialog.askfloat("Exact joint angle", f"{joint_name}\nEnter target angle in degrees:", initialvalue=round(current_degrees, 2), parent=root)
    root.destroy()
    return angle


def load_and_check(model_path):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    accel_lookup, gyro_lookup = discover_sensors(model)
    missing = [site for site in IMU_ORDER if site not in accel_lookup or site not in gyro_lookup]
    if missing: raise RuntimeError(f"Missing simulated IMU sensors: {', '.join(missing)}")
    missing_joints = [name for name in WRIST_WORLD_JOINTS if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0]
    if missing_joints: raise RuntimeError(f"Pose-recorder model is missing Wrist-to-World joints: {', '.join(missing_joints)}")
    if model.nq < WRIST_WORLD_DOF + len(Joint_ORDER): raise RuntimeError(f"Pose-recorder model must contain at least 3 Wrist-to-World plus {len(Joint_ORDER)} hand qpos values; found {model.nq}")
    return model, data, accel_lookup, gyro_lookup


def validate(model_path):
    model, data, accel_lookup, gyro_lookup = load_and_check(model_path)
    snapshot = capture_snapshot(model, data, accel_lookup, gyro_lookup)
    expected = {"wrist_world_qpos": (WRIST_WORLD_DOF,), "wrist_world_quat": (4,), "qpos": (len(Joint_ORDER),), "accel": (len(IMU_ORDER), 3), "gyro": (len(IMU_ORDER), 3), "world_quat": (len(IMU_ORDER), 4), "relative_quat": (len(SITE_PAIRS), 4)}
    for key, shape in expected.items():
        if snapshot[key].shape != shape: raise RuntimeError(f"{key} has shape {snapshot[key].shape}; expected {shape}")
    print(f"Validation passed: {len(IMU_ORDER)} IMUs, 3 Wrist-to-World controls and {len(Joint_ORDER)} saved hand qpos values.")


def main(model_path):
    model, data, accel_lookup, gyro_lookup = load_and_check(model_path)
    snapshots, viewer_holder = [], {}
    actuator_for_joint = {int(model.actuator_trnid[index, 0]): index for index in range(model.nu)}
    controlled_joints = sorted(actuator_for_joint)
    fine_control = {"joint": controlled_joints[0], "step_degrees": 1.0}

    def joint_name(index): return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) or f"joint_{index}"

    def show_selected():
        index = fine_control["joint"]
        actuator = actuator_for_joint[index]
        section = "Wrist → World" if index < WRIST_WORLD_DOF else "Palm / fingers"
        print(f"Selected {controlled_joints.index(index) + 1:02d}/{len(controlled_joints)} [{section}] {joint_name(index)} | target {np.rad2deg(data.ctrl[actuator]):.2f}° | step {fine_control['step_degrees']:g}°")

    def set_joint_degrees(degrees):
        index = fine_control["joint"]
        actuator = actuator_for_joint[index]
        low, high = model.actuator_ctrlrange[actuator]
        data.ctrl[actuator] = np.clip(np.deg2rad(degrees), low, high)
        show_selected()

    def key_callback(keycode):
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        if key == "C":
            label = ask_snapshot_name(f"pose_{len(snapshots) + 1:03d}")
            if label is None:
                print("Snapshot capture cancelled.")
                return
            snapshot = capture_snapshot(model, data, accel_lookup, gyro_lookup)
            snapshot["label"] = label
            snapshots.append(snapshot)
            print(f"Captured {snapshot['label']} — {len(snapshots)} snapshots held in memory.")
        elif key == "U":
            if snapshots: print(f"Removed {snapshots.pop()['label']} — {len(snapshots)} snapshots remain.")
            else: print("Nothing to undo.")
        elif key == "S":
            if snapshots: save_snapshots(snapshots, model_path, float(model.opt.timestep))
            else: print("Nothing to save. Press C in the viewer to capture a pose first.")
        elif key == "A":
            if snapshots: append_snapshots(snapshots)
            else: print("Nothing to append. Press C in the viewer to capture a pose first.")
        elif key == "Q": viewer_holder["viewer"].close()
        elif key == "P": fine_control["joint"] = controlled_joints[(controlled_joints.index(fine_control["joint"]) - 1) % len(controlled_joints)]; show_selected()
        elif key == "N": fine_control["joint"] = controlled_joints[(controlled_joints.index(fine_control["joint"]) + 1) % len(controlled_joints)]; show_selected()
        elif key in {",", "."}:
            actuator = actuator_for_joint[fine_control["joint"]]
            change = fine_control["step_degrees"] * (-1.0 if key == "," else 1.0)
            set_joint_degrees(np.rad2deg(data.ctrl[actuator]) + change)
        elif key in {"1", "2", "3"}:
            fine_control["step_degrees"] = {"1": 0.1, "2": 1.0, "3": 5.0}[key]
            show_selected()
        elif key == "E":
            actuator = actuator_for_joint[fine_control["joint"]]
            angle = ask_joint_angle(joint_name(fine_control["joint"]), np.rad2deg(data.ctrl[actuator]))
            if angle is not None: set_joint_degrees(angle)
        elif key == "Z": set_joint_degrees(0.0)
        elif key == "V":
            print("\nJoint targets:")
            for number, joint in enumerate(controlled_joints, start=1): print(f"  {number:02d} {joint_name(joint):22s} {np.rad2deg(data.ctrl[actuator_for_joint[joint]]):8.2f}°")

    print_controls()
    show_selected()
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer_holder["viewer"] = viewer
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
    if snapshots: print(f"Viewer closed with {len(snapshots)} snapshot(s) in memory. Use Save new H5 before closing next time.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    model_path = arguments.model.resolve()
    validate(model_path) if arguments.validate_only else main(model_path)
