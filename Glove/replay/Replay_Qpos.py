from common import *

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
        state = {"frame": 0,"paused": False,"speed_index": 3,"qpos": None}
        load_trajectory(state,f)

        viewer_holder = {}
        with mujoco.viewer.launch_passive(model,data,key_callback=lambda key: key_callback(key,state,viewer_holder)) as viewer:
            viewer_holder["viewer"] = viewer
            Controls_Print_Playback();
            while viewer.is_running():
                data.qpos[:] = state["qpos"][state["frame"]]
                mujoco.mj_forward(model, data)
                viewer.sync()

                if state["paused"]:
                    time.sleep(0.01)
                    continue

                time.sleep(dt / SPEEDS[state["speed_index"]])

                state["frame"] = (state["frame"] + 1) % len(state["qpos"])
                    


if __name__ == "__main__":
    main()