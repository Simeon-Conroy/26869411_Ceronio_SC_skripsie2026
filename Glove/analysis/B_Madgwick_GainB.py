from common import *
import time

def site_quats_Compute(state,delta_t,Beta,gyro_noise,accel_noise):
    for i, site in enumerate(IMU_ORDER):
        
        noise_accel = np.random.normal(0, accel_noise, 3)
        noise_gyro = np.random.normal(0, gyro_noise, 3)

        accel_raw = state["accel"][state["frame"],i].copy()
        accel_raw[0] = noise_accel[0] + accel_raw[0]
        accel_raw[1] = noise_accel[1] + accel_raw[1]
        accel_raw[2] = noise_accel[2] + accel_raw[2]

        accel = Vect3_Normalise(accel_raw)

        w = state["gyro"][state["frame"],i].copy()
        w[0] = noise_gyro[0] + w[0]
        w[1] = noise_gyro[1] + w[1]
        w[2] = noise_gyro[2] + w[2]

        if state["frame"] == 0:
            q_est_old = Site_Initial_quat[site]
        else:
            q_est_old = state["site_quats"][state["frame"]-1,i]

        q_est = madgwick_filter(w,q_est_old,accel,delta_t,Beta)
        state["site_quats"][state["frame"],i] = q_est
    

def main(DATASET, run_name="", output_dir="filter_results", noise_enabled=False, monte_runs=None):
    start = time.perf_counter()
    results_main = []
    
    with h5py.File(DATASET, "r") as f:
        qpos_raw = f["trajectory"]["qpos"][:]
        sample_rate = 100
        dt = 1.0 / sample_rate
        
        state = {"frame": 0,"relative_quat": None,"qpos" : None,"accel":None,"gyro":None,"site_quats":None}
        print(f"\n============================================================")
        load_trajectory(state,f)
        num_frames = state["qpos"].shape[0]
        save_state = {"recovered_qpos":np.zeros_like(state["qpos"])}
        
        gyro_noise = SIM_GYRO_NOISE_STD if noise_enabled else 0.0
        accel_noise = SIM_ACCEL_NOISE_STD if noise_enabled else 0.0
        num_joints = qpos_raw.shape[1]
        results_main = run_monte_carlo_sweep(
                        param_list=Beta_list, param_name="Beta", monte_Runs=monte_Runs if monte_runs is None else monte_runs,
                        num_frames=num_frames, num_joints=num_joints, qpos_raw=qpos_raw,
                        state=state, save_state=save_state, dt=dt,
                        gyro_noise=gyro_noise, accel_noise=accel_noise,
                        joint_names=MODEL_QPOS_ORDER,
                        site_quats_Compute=site_quats_Compute,
                        rel_quat_compute=rel_quat_compute,
                        rel_quat_to_qpos2=rel_quat_to_qpos2,
                        NUM_IMUS=NUM_IMUS,
                    )

    elapsed = time.perf_counter() - start
    print(f"Elapsed: {elapsed:.2f} s")
    result_name = f"{run_name}_{'noise' if noise_enabled else 'no_noise'}"
    save_results_h52(results_main, MODEL_QPOS_ORDER, result_filename("Madgwick", result_name, output_dir))
                
                
if __name__ == "__main__":
    main()

