from common import *

def read_imus(data, accel_lookup, gyro_lookup):

    accel = np.zeros((NUM_IMUS, 3))
    gyro = np.zeros((NUM_IMUS, 3))

    for i, site in enumerate(IMU_ORDER):
        a = accel_lookup[site]
        g = gyro_lookup[site]
        accel[i] = data.sensordata[a["adr"] : a["adr"] + 3]
        gyro[i] = data.sensordata[g["adr"] : g["adr"] + 3]
    return accel, gyro

def site_rotation(model, data, site_name):
    site_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,site_name,)
    xmat = data.site_xmat[site_id].reshape(3, 3)
    return Rot.from_matrix(xmat)
def all_site_world_quats(model, data):
    world = np.zeros((NUM_IMUS, 4))
    for i, site in enumerate(IMU_ORDER):
        q = site_rotation(model,data,site)
        world[i] = scipy_to_mj(q.as_quat())
    return world
def all_site_world_positions(model, data):
    positions = np.zeros((NUM_IMUS, 3))
    for i, site in enumerate(IMU_ORDER):
        site_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,site)
        positions[i] = data.site_xpos[site_id]
    return positions
def all_site_relative_quats(model, data):
    relative = np.zeros((NUM_RELATIVE_QUATS, 4))
    for i, (child, parent) in enumerate(SITE_PAIRS):
        q_child = site_rotation(model, data, child)
        q_parent = site_rotation(model, data, parent)
        q_rel = q_parent.inv() * q_child
        relative[i] = scipy_to_mj(q_rel.as_quat())
    return relative
def capture_frame(model, data, accel_lookup, gyro_lookup):
    accel, gyro = read_imus(data,accel_lookup,gyro_lookup,)
    world_pos = all_site_world_positions( model,data)
    world_quat = all_site_world_quats(model,data)
    relative_quat = all_site_relative_quats(model,data,)
    frame = {"sim_time": float(data.time),"world_pos": world_pos.tolist(),"world_quat": world_quat.tolist(),"accel": accel.tolist(),"gyro": gyro.tolist(),"relative_quat": relative_quat.tolist(),"qpos": data.qpos.tolist()}
    return frame

def main():

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    accel_lookup, gyro_lookup = discover_sensors(model)
    print(f"\nDiscovered {NUM_IMUS} IMUs")

    timestep = float(model.opt.timestep)
    steps_per_sample = max(1,round(1.0 / (SAMPLE_RATE_HZ * timestep)))
    actual_rate = 1.0 / (steps_per_sample * timestep)
    print(f"Simulation timestep : {timestep:.6f} s")
    print(f"Requested rate      : {SAMPLE_RATE_HZ} Hz")
    print(f"Actual rate         : {actual_rate:.2f} Hz\n")

    recording = False
    trajectory = []
    sample_counter = 0
    viewer_holder = {}

    def key_callback(keycode):
        nonlocal recording
        nonlocal sample_counter
        nonlocal trajectory
        key = chr(keycode).upper() if 0 <= keycode < 256 else ""
        if key == "R":
            if not recording:
                recording = True
                trajectory = []
                sample_counter = 0
                print("\nRecording started.")
            else:
                recording = False
                if len(trajectory):
                    duration = len(trajectory) / actual_rate
                    print(f"Recording stopped.\n"f"Frames   : {len(trajectory)}\n"f"Duration : {duration:.2f} s\n")
                else:
                    print("No frames recorded.")
        elif key == "S":
            if trajectory:
                save_hdf5_Recording(trajectory,actual_rate,timestep,)
                print("Recording saved.\n")
            else:
                print("Nothing to save.\n")
        elif key == "Q":
            print("Closing viewer...")
            viewer_holder["viewer"].close()

    with mujoco.viewer.launch_passive(model,data,key_callback=key_callback,) as viewer:
        viewer_holder["viewer"] = viewer
        while viewer.is_running():
            mujoco.mj_step(model, data)
            if recording:
                sample_counter += 1
                if sample_counter >= steps_per_sample:
                    sample_counter = 0
                    frame = capture_frame(model,data,accel_lookup,gyro_lookup,)
                    trajectory.append(frame)
            viewer.sync()

if __name__ == "__main__":
    main()