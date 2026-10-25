import numpy as np
from .Const_INFO import * 
from .Quat_Operation import *

import time

def Error_Metrics2(ground, recovered):
    error_mat = recovered - ground 
    error_mat = (error_mat + np.pi) % (2 * np.pi) - np.pi
    RMSE_O = np.sqrt(np.mean(error_mat ** 2))
    MAE_O = np.mean(np.abs(error_mat))
    RMSE_pj = np.sqrt(np.mean(error_mat ** 2, axis=0))          
    MAE_pj = np.mean(np.abs(error_mat), axis=0)                 
    return RMSE_O, MAE_O, RMSE_pj, MAE_pj

def run_monte_carlo_sweep(param_list, param_name, monte_Runs, num_frames,
                           num_joints, qpos_raw, state, save_state, dt,
                           gyro_noise, accel_noise, joint_names,
                           site_quats_Compute, rel_quat_compute, rel_quat_to_qpos2,
                           NUM_IMUS):

    results_main = []
    start = time.perf_counter()

    for param_index, param_val in enumerate(param_list, start=1):
        print(f"Starting {param_name} {param_index}/{len(param_list)} = {param_val:.5f}", flush=True)
        RMSE_monte, MAE_monte = [], []
        RMSE_pj_monte = np.zeros((monte_Runs, num_joints))
        MAE_pj_monte = np.zeros((monte_Runs, num_joints))

        for i in range(monte_Runs):
            run_start = time.perf_counter()

            state["site_quats"] = np.zeros((num_frames, NUM_IMUS, 4))
            save_state["recovered_qpos"] = np.zeros_like(state["qpos"])

            for frame in range(num_frames):
                state["frame"] = frame
                site_quats_Compute(state, dt, param_val, gyro_noise, accel_noise)
                rel_quat = rel_quat_compute(state)
                qpos = rel_quat_to_qpos2(rel_quat)
                save_state["recovered_qpos"][state["frame"]] = qpos
                
            RMSE_O, MAE_O, RMSE_pj, MAE_pj = Error_Metrics2(qpos_raw, save_state["recovered_qpos"])

            RMSE_monte.append(RMSE_O)
            MAE_monte.append(MAE_O)

            RMSE_pj_monte[i] = RMSE_pj
            MAE_pj_monte[i] = MAE_pj
            print(f"  {param_name} {param_index}/{len(param_list)} run {i + 1}/{monte_Runs} done in {time.perf_counter() - run_start:.2f} s", flush=True)

        RMSE_mean, RMSE_std = np.mean(RMSE_monte), np.std(RMSE_monte)
        MAE_mean, MAE_std = np.mean(MAE_monte), np.std(MAE_monte)

        RMSE_pj_mean = RMSE_pj_monte.mean(axis=0) 
        RMSE_pj_std = RMSE_pj_monte.std(axis=0)
        MAE_pj_mean = MAE_pj_monte.mean(axis=0)
        MAE_pj_std = MAE_pj_monte.std(axis=0)

        print(f"{param_name}: {param_val:.5f}  "
              f"RMSE: {RMSE_mean:.5f}±{RMSE_std:.5f}  "
              f"MAE: {MAE_mean:.5f}±{MAE_std:.5f}")
        print(f"Elapsed: {time.perf_counter() - start:.2f} s")

        results_main.append({
            param_name: param_val,
            "RMSE_mean": RMSE_mean, "RMSE_std": RMSE_std,
            "MAE_mean": MAE_mean, "MAE_std": MAE_std,
            "RMSE_pj_mean": RMSE_pj_mean, "RMSE_pj_std": RMSE_pj_std,
            "MAE_pj_mean": MAE_pj_mean, "MAE_pj_std": MAE_pj_std,
        })

    return results_main

def run_monte_carlo_sweep_mahony(Kp_list, Ki_list, monte_Runs, num_frames,
                                    num_joints, qpos_raw, state, save_state, dt,
                                    gyro_noise, accel_noise, joint_names,
                                    site_quats_Compute, rel_quat_compute, rel_quat_to_qpos2,
                                    NUM_IMUS):
        results_main = []
        start = time.perf_counter()
    
        for kp_index, Kp in enumerate(Kp_list, start=1):
            for ki_index, Ki in enumerate(Ki_list, start=1):
                print(f"Starting Mahony Kp {kp_index}/{len(Kp_list)} = {Kp:.5f}, Ki {ki_index}/{len(Ki_list)} = {Ki:.5f}", flush=True)
                RMSE_monte, MAE_monte = [], []
                RMSE_pj_monte = np.zeros((monte_Runs, num_joints))
                MAE_pj_monte = np.zeros((monte_Runs, num_joints))
    
                for i in range(monte_Runs):
                    run_start = time.perf_counter()
                    state["site_quats"] = np.zeros((num_frames, NUM_IMUS, 4))
                    save_state["recovered_qpos"] = np.zeros_like(state["qpos"])
    
                    for frame in range(num_frames):
                        state["frame"] = frame
                        site_quats_Compute(state, dt, Kp, Ki, gyro_noise, accel_noise)
                        rel_quat = rel_quat_compute(state)
                        qpos = rel_quat_to_qpos2(rel_quat)
                        save_state["recovered_qpos"][state["frame"]] = qpos

                    
                    RMSE_O, MAE_O, RMSE_pj, MAE_pj = Error_Metrics2(qpos_raw, save_state["recovered_qpos"])
    
                    RMSE_monte.append(RMSE_O)
                    MAE_monte.append(MAE_O)
    
                    RMSE_pj_monte[i] = RMSE_pj
                    MAE_pj_monte[i] = MAE_pj
                    print(f"  Mahony Kp {kp_index}/{len(Kp_list)} Ki {ki_index}/{len(Ki_list)} run {i + 1}/{monte_Runs} done in {time.perf_counter() - run_start:.2f} s", flush=True)
    
                RMSE_mean, RMSE_std = np.mean(RMSE_monte), np.std(RMSE_monte)
                MAE_mean, MAE_std = np.mean(MAE_monte), np.std(MAE_monte)
    
                RMSE_pj_mean = RMSE_pj_monte.mean(axis=0)
                RMSE_pj_std = RMSE_pj_monte.std(axis=0)
                MAE_pj_mean = MAE_pj_monte.mean(axis=0)
                MAE_pj_std = MAE_pj_monte.std(axis=0)
    
                print(f"Kp: {Kp:.5f} Ki: {Ki:.5f}  "
                    f"RMSE: {RMSE_mean:.5f}±{RMSE_std:.5f}  "
                    f"MAE: {MAE_mean:.5f}±{MAE_std:.5f}")
                print(f"Elapsed: {time.perf_counter() - start:.2f} s")
    
                results_main.append({
                    "Kp": Kp, "Ki": Ki,
                    "RMSE_mean": RMSE_mean, "RMSE_std": RMSE_std,
                    "MAE_mean": MAE_mean, "MAE_std": MAE_std,
                    "RMSE_pj_mean": RMSE_pj_mean, "RMSE_pj_std": RMSE_pj_std,
                    "MAE_pj_mean": MAE_pj_mean, "MAE_pj_std": MAE_pj_std,
                })
    
        return results_main


