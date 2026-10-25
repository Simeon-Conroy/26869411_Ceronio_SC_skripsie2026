from common import *
from common.Const_INFO import MODEL_QPOS

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

def main():
    dataset = choose_file()
    if not dataset:
        print("No file selected.")
        return

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    with h5py.File(dataset, "r") as f:
        sample_rate = f["meta"].attrs["sample_rate_hz"]
        dt = 1.0 / sample_rate
 
        state = {"frame": 0,"relative_quat": None,"qpos" : None,"accel":None,"gyro":None,"site_quats":None,"paused":False,"speed_index":3}
        load_trajectory(state,f)
        num_frames = state["qpos"].shape[0]
        state["site_quats"] = np.zeros((num_frames, NUM_IMUS, 4))

        save_state = {"recovered_qpos":np.zeros((num_frames, MODEL_QPOS)),"site_quats":state["site_quats"],"relative_quat":np.zeros((num_frames, NUM_RELATIVE_QUATS + 1, 4)),"saved": False}

        viewer_holder = {}
        with mujoco.viewer.launch_passive(model,data,key_callback=lambda key: key_callback(key,state,viewer_holder)) as viewer:
            viewer_holder["viewer"] = viewer
            Controls_Print_Playback()
            while viewer.is_running():
                site_quats_Compute(state, dt, 0.01, 0, 0)
                rel_quat = rel_quat_compute(state)
                qpos = rel_quat_to_qpos2(rel_quat)
                save_state["recovered_qpos"][state["frame"]] = qpos
                save_state["relative_quat"][state["frame"]] = rel_quat
                data.qpos[:] = qpos
                
                mujoco.mj_forward(model, data)
                viewer.sync()

                if state["paused"]:
                    time.sleep(0.01)
                    continue
                time.sleep(dt / SPEEDS[state["speed_index"]])

                if (state["frame"] == len(state["qpos"]) - 1 and not save_state["saved"]):
                    save_recovered_Data(save_state)
                    save_state["saved"] = True

                state["frame"] = (state["frame"] + 1) % len(state["qpos"])
                
if __name__ == "__main__":
    main()