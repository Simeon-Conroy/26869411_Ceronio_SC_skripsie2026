import h5py
import tkinter as tk
from tkinter import filedialog
from .Const_INFO import * 
import numpy as np
from .H5_file_Handling import *

def save_hdf5_Recording(trajectory, sample_rate, timestep):
    dataset = Save_File_H5()
    
    if not dataset:
        print("No file selected.")
        return

    with h5py.File(dataset, "w") as file:
        # Metadata
        meta = file.create_group("meta")
        meta.attrs["sample_rate_hz"] = sample_rate
        meta.attrs["timestep"] = timestep
        meta.create_dataset("imu_order",data=np.array(IMU_ORDER,dtype="S"))
        pair_strings = [f"{child}|{parent}"for child, parent in SITE_PAIRS]
        meta.create_dataset("relative_pairs",data=np.array(pair_strings,dtype="S"))
        # Trajectories
        trajectory_group = file.create_group("trajectory")
        sim_time = np.array([frame["sim_time"] for frame in trajectory])
        world_pos = np.array([frame["world_pos"] for frame in trajectory])
        accel = np.array([frame["accel"] for frame in trajectory])
        gyro = np.array([frame["gyro"] for frame in trajectory])
        world_quat = np.array([frame["world_quat"] for frame in trajectory])
        relative_quat = np.array([frame["relative_quat"] for frame in trajectory])
        qpos = np.array([frame["qpos"] for frame in trajectory]) 
        trajectory_group.create_dataset("sim_time",data=sim_time,compression="gzip",)
        trajectory_group.create_dataset("world_pos",data=world_pos,compression="gzip",)
        trajectory_group.create_dataset("world_quat",data=world_quat,compression="gzip",)
        trajectory_group.create_dataset("accel",data=accel,compression="gzip",)
        trajectory_group.create_dataset("gyro",data=gyro,compression="gzip",)
        trajectory_group.create_dataset("relative_quat",data=relative_quat,compression="gzip",)
        trajectory_group.create_dataset("qpos",data=qpos,compression="gzip",)
    print("HDF5 saved.")
# Trajectory Loading

def load_trajectory(state, f):
    trajectory = f["trajectory"]

    if "qpos" in trajectory:
        state["qpos"] = trajectory["qpos"][:]

    if "relative_quat" in trajectory:
        state["relative_quat"] = trajectory["relative_quat"][:]

    if "accel" in trajectory:
            state["accel"] = trajectory["accel"][:]

    if "gyro" in trajectory:
            state["gyro"] = trajectory["gyro"][:]

    print("Trajectory Loaded")

def save_recovered_Data(save_state): 
    dataset = Save_File_H5()
    if not dataset:
            print("No file selected.")
            return
    with h5py.File(dataset, "w") as f:
        trajectory = f.create_group("trajectory")
        if save_state.get("recovered_qpos") is not None:
            trajectory.create_dataset("qpos",data=save_state["recovered_qpos"],compression="gzip")

        if save_state.get("site_quats") is not None:
            trajectory.create_dataset("site_quats",data=save_state["site_quats"],compression="gzip")

        if save_state.get("relative_quat") is not None:
            trajectory.create_dataset("relative_quat",data=save_state["relative_quat"],compression="gzip")
       
        print("Recovered data saved") 

