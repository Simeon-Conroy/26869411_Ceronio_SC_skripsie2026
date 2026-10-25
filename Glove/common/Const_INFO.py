MODEL_PATH = "Sharpa_Hand3.xml"                 
SPEEDS = [0.1,0.25,0.5,1.0,2.0,4.0,8.0,]
SAMPLE_RATE_HZ = 100
import numpy as np

WRIST_WORLD_DOF = 3
NUM_QPOS = 24
MODEL_QPOS = WRIST_WORLD_DOF + NUM_QPOS
WRIST_WORLD_ORDER = ["Wrist_World_Roll", "Wrist_World_Pitch", "Wrist_World_Yaw"]

Pair_info = { 
"palm-wrist" : {"DOF": 3,"qpos":(0, 1,2)},
"imu_thumb_cmc_imu_palm" :      {"DOF": 2,"qpos":(3, 4)},
"imu_thumb_prox_imu_thumb_cmc" :{"DOF": 2,"qpos":(5, 6)},
"imu_thumb_dist_imu_thumb_prox" :{"DOF": 1,"qpos":7},
"imu_index_prox_imu_palm" :      {"DOF": 2,"qpos":(8, 9)},
"imu_index_mid_imu_index_prox" : {"DOF": 1,"qpos":10},
"imu_index_dist_imu_index_mid" : {"DOF": 1,"qpos":11},
"imu_middle_prox_imu_palm" :     {"DOF": 2,"qpos":(12, 13)},
"imu_middle_mid_imu_middle_prox" :    {"DOF": 1,"qpos":14},
"imu_middle_dist_imu_middle_mid" :    {"DOF": 1,"qpos":15},
"imu_ring_prox_imu_palm" :    {"DOF": 2,"qpos":(16,17)},
"imu_ring_mid_imu_ring_prox" :    {"DOF": 1,"qpos":18},
"imu_ring_dist_imu_ring_mid"  :    {"DOF": 1,"qpos":19},
"imu_pinky_prox_imu_palm" :    {"DOF": 2,"qpos":(20, 21)},
"imu_pinky_mid_imu_pinky_prox" :    {"DOF": 1,"qpos":22},
"imu_pinky_dist_imu_pinky_mid" :    {"DOF": 1,"qpos":23},                                  
}

SITE_PAIRS = [
    ("imu_palm", "imu_wrist"),

    ("imu_thumb_cmc", "imu_palm"),
    ("imu_thumb_prox", "imu_thumb_cmc"),
    ("imu_thumb_dist", "imu_thumb_prox"),

    ("imu_index_prox", "imu_palm"),
    ("imu_index_mid", "imu_index_prox"),
    ("imu_index_dist", "imu_index_mid"),

    ("imu_middle_prox", "imu_palm"),
    ("imu_middle_mid", "imu_middle_prox"),
    ("imu_middle_dist", "imu_middle_mid"),

    ("imu_ring_prox", "imu_palm"),
    ("imu_ring_mid", "imu_ring_prox"),
    ("imu_ring_dist", "imu_ring_mid"),

    ("imu_pinky_prox", "imu_palm"),
    ("imu_pinky_mid", "imu_pinky_prox"),
    ("imu_pinky_dist", "imu_pinky_mid"),
]

def Controls_Print_Playback() :
    print()
    print("Controls")
    print("--------------------------------")
    print("Space : Pause / Resume")
    print("R     : Restart")
    print("L     : Toggle Loop")
    print("+     : Faster")
    print("-     : Slower")
    print("←     : Previous Frame")
    print("→     : Next Frame")
    print("Q     : Quit")
    print("--------------------------------")

IMU_ORDER = [
    "imu_wrist",

    "imu_palm",

    "imu_thumb_cmc",
    "imu_thumb_prox",
    "imu_thumb_dist",

    "imu_index_prox",
    "imu_index_mid",
    "imu_index_dist",

    "imu_middle_prox",
    "imu_middle_mid",
    "imu_middle_dist",

    "imu_ring_prox",
    "imu_ring_mid",
    "imu_ring_dist",

    "imu_pinky_prox",
    "imu_pinky_mid",
    "imu_pinky_dist",
]

Joint_ORDER = [
    "Roll_Palm",
    "Pitch_Palm",
    "Yaw_Palm",

    "thumb_CMC_Flex",
    "thumb_CMC_Abduct",
    
    "thumb_MCP_Flex",
    "thumb_MCP_Abduct",
    "imu_thumb_dist",

    "index_prox_Flex",
    "index_prox_Abduct",
    "imu_index_mid",
    "imu_index_dist",

    "middle_prox_Flex",
    "middle_prox_Abduct",
    "imu_middle_mid",
    "imu_middle_dist",

    "ring_prox_Flex",
    "ring_prox_Abduct",
    "imu_ring_mid",
    "imu_ring_dist",

    "pinky_prox_Flex",
    "pinky_prox_Abduct",
    "imu_pinky_mid",
    "imu_pinky_dist",
]

MODEL_QPOS_ORDER = WRIST_WORLD_ORDER + Joint_ORDER

Site_Initial_quat = {
    "imu_wrist" :np.array([0.5, 0.5, -0.5, 0.5], dtype=float),

    "imu_palm" :np.array([0.5, 0.5, -0.5, 0.5], dtype=float),

    "imu_thumb_cmc":np.array([0.701056475, -0.0922953485, -0.560986742, 0.430459363], dtype=float),
    "imu_thumb_prox":np.array([0.70089669, -0.092890402, -0.560679763, 0.430991191], dtype=float),
    "imu_thumb_dist":np.array([0.700885886, -0.092903227, -0.560676477, 0.431010271], dtype=float),

    "imu_index_prox":np.array([0.499983678, 0.499976038, -0.500016443, 0.500023839], dtype=float),
    "imu_index_mid":np.array([0.499973709, 0.499968191, -0.50002429, 0.500033807], dtype=float),
    "imu_index_dist":np.array([0.499969397, 0.49996388, -0.500028601, 0.500038118], dtype=float),

    "imu_middle_prox":np.array([0.499983678, 0.499976038, -0.500016443, 0.500023839], dtype=float),
    "imu_middle_mid":np.array([0.499973709, 0.499968191, -0.50002429, 0.500033807], dtype=float),
    "imu_middle_dist":np.array([0.499969397, 0.49996388, -0.500028601, 0.500038118], dtype=float),

    "imu_ring_prox":np.array([0.499983678, 0.499976038, -0.500016443, 0.500023839], dtype=float),
    "imu_ring_mid":np.array([0.499973709, 0.499968191, -0.50002429, 0.500033807], dtype=float),
    "imu_ring_dist":np.array([0.499969397, 0.49996388, -0.500028601, 0.500038118], dtype=float),

    "imu_pinky_prox":np.array([0.499989369, 0.499972359, -0.500014425, 0.500023846], dtype=float),
    "imu_pinky_mid":np.array([0.499979401, 0.499964512, -0.500022271, 0.500033813], dtype=float),
    "imu_pinky_dist":np.array([0.499975089, 0.4999602, -0.500026582, 0.500038124], dtype=float),
}

Site_Index = {site: index for index, site in enumerate(IMU_ORDER)}
NUM_IMUS = len(IMU_ORDER)
NUM_RELATIVE_QUATS = len(SITE_PAIRS)

SIM_GYRO_NOISE_STD = 5.42E-04
SIM_ACCEL_NOISE_STD = 1.20E-02