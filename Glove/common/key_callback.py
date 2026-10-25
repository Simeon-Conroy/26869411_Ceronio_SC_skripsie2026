from .Trajectory_Load_Save import *
from .Const_INFO import *


def key_callback(keycode,state,viewer_holder):
    key = chr(keycode) if 0 <= keycode < 256 else ""
    # Pause / Resume
    if key == " ":
        state["paused"] = not state["paused"]
        if state["paused"]:
            print("Paused")
        else:
            print("Playing")
    # Restart
    elif key.upper() == "R":
        state["frame"] = 0
        print("Restarted")
    # Faster Playback
    elif key == "+":
        if state["speed_index"] < len(SPEEDS) - 1:
            state["speed_index"] += 1
        print(
            f"Playback : {SPEEDS[state['speed_index']]}x"
        )
    # Slower Playback
    elif key == "-":
        if state["speed_index"] > 0:
            state["speed_index"] -= 1
        print(
            f"Playback : {SPEEDS[state['speed_index']]}x"
        )
    # Step Forward (Paused)
    elif keycode == 262 and state["paused"]:
        if state["frame"] < len(state["qpos"]) - 1:
            state["frame"] += 1
    # Step Backward (Paused)
    elif keycode == 263 and state["paused"]:

        if state["frame"] > 0:
            state["frame"] -= 1
    # Quit
    elif key.upper() == "Q":
        viewer_holder["viewer"].close()