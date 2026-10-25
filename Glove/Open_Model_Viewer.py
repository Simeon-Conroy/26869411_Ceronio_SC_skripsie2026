import mujoco
import mujoco.viewer

MODEL_PATH = "Sharpa_Hand3.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()