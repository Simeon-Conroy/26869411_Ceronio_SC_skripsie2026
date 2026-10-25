import numpy as np
import math
from .Const_INFO import * 

def scipy_to_mj(quat):
    return [quat[3],quat[0],quat[1],quat[2]]
def mj_to_scipy(quat):
    return [quat[1],quat[2],quat[3],quat[0]]

def quat_Mult(p,q):
    #p⊗q Returns Q
    Q = np.zeros(4)
    Q[0] = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    Q[1] = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    Q[2] = p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1]
    Q[3] = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]
    return Q

def quat_to_R(q):
    # takes in quat returns DCM
    R = np.zeros((3,3))
    R[0,0] = (q[0]*q[0]) + (q[1]*q[1]) - (q[2]*q[2]) - (q[3]*q[3]) 
    R[1,0] = 2*(q[1]*q[2] + q[0]*q[3])
    R[2,0] = 2*(q[1]*q[3] - q[0]*q[2])
    R[0,1] = 2*(q[1]*q[2] - q[0]*q[3])
    R[1,1] = (q[0]*q[0]) - (q[1]*q[1]) + (q[2]*q[2]) - (q[3]*q[3]) 
    R[2,1] = 2*(q[2]*q[3] + q[0]*q[1])
    R[0,2] = 2*(q[1]*q[3] + q[0]*q[2])
    R[1,2] = 2*(q[2]*q[3] - q[0]*q[1])
    R[2,2] = (q[0]*q[0]) - (q[1]*q[1]) - (q[2]*q[2]) + (q[3]*q[3])
    return R

def inv_conj_quat(q):
    q = np.asarray(q, dtype=float)
    norm_sq = np.dot(q, q)
    p = np.array([q[0], -q[1], -q[2], -q[3]])
    return p / norm_sq

def vect3_to_pure_quat(v):
    q = [0,v[0],v[1],v[2]]
    return np.array(q, dtype=float)

def unit_quat():
    return np.array([1,0,0,0], dtype=float)

def quat_Normalise(q):
    return q/np.sqrt(q[0]**2+q[1]**2+q[2]**2+q[3]**2)

def quat_Norm(q):
    return np.sqrt(q[0]**2+q[1]**2+q[2]**2+q[3]**2)

def Vect3_Normalise(q):
    if q[0] == 0 and q[1] ==0 and q[2] == 0 :
        return q 
    else :
        return q/np.sqrt(q[0]**2+q[1]**2+q[2]**2)

def rel_quat_compute(state):
    q_out = np.zeros((NUM_RELATIVE_QUATS + 1, 4))
    q_wrist_site = state["site_quats"][state["frame"],Site_Index["imu_wrist"]]
    q_world_reference = Site_Initial_quat["imu_wrist"]
    q_out[0] = quat_Normalise(quat_Mult(inv_conj_quat(q_world_reference), q_wrist_site))
    for i, (child, parent) in enumerate(SITE_PAIRS, start=1):
        child_i = Site_Index[child]
        parent_i = Site_Index[parent]
        q_child = state["site_quats"][state["frame"],child_i]
        q_parent = state["site_quats"][state["frame"],parent_i]
        q_out[i] = quat_Mult(inv_conj_quat(q_parent),q_child)
    return q_out

CMC_FLEX_Q = quat_Normalise(np.array([0.807155194, 0.466011364, 0.313847759, 0.181200125], dtype=float))
WRIST_PALM_Q = quat_Normalise(np.array([0.5, -0.5, 0.5, -0.5], dtype=float))
PALM_WRIST_Q0 = quat_Normalise(np.array([9.999e-1, -7.914e-6, 1.601e-2, -5.572e-4], dtype=float))

def change_quat_basis(q, offset):
    return quat_Normalise(quat_Mult(quat_Mult(inv_conj_quat(offset), q), offset))

def rel_quat_to_qpos2(rel_quat):
    rel_quat = np.asarray(rel_quat, dtype=float)
    model_qpos = np.zeros(MODEL_QPOS)
    hand_qpos = model_qpos[WRIST_WORLD_DOF:WRIST_WORLD_DOF + NUM_QPOS]
    CMC_Q0 = np.array([7.995e-1,-4.616e-1,3.329e-1,1.922e-1])

    PAIR_INFO = [("wrist-world", {"DOF": 3, "qpos": (0, 1, 2), "target": "model"})] + list(Pair_info.items())
    CMC_Q0_INV = inv_conj_quat(CMC_Q0)
    for i, (pair_name, info) in enumerate(PAIR_INFO):
        Q_rel = rel_quat[i]
        if i == 2 : Q_rel = quat_Mult(CMC_Q0_INV,Q_rel)
        if pair_name in ("wrist-world", "palm-wrist"):
            if pair_name == "palm-wrist":
                Q_rel = quat_Normalise(quat_Mult(PALM_WRIST_Q0, Q_rel))
            Q_rel = change_quat_basis(Q_rel, WRIST_PALM_Q)
        if pair_name == "imu_thumb_cmc_imu_palm":
            Q_rel = change_quat_basis(Q_rel, CMC_FLEX_Q)

        R_10 = 2*(Q_rel[1]*Q_rel[2] + Q_rel[0]*Q_rel[3])
        R_00 = (Q_rel[0]*Q_rel[0]) + (Q_rel[1]*Q_rel[1]) - (Q_rel[2]*Q_rel[2]) - (Q_rel[3]*Q_rel[3]) 
        R_01 = 2*(Q_rel[1]*Q_rel[2] - Q_rel[0]*Q_rel[3])
        R_11 = (Q_rel[0]*Q_rel[0]) - (Q_rel[1]*Q_rel[1]) + (Q_rel[2]*Q_rel[2]) - (Q_rel[3]*Q_rel[3]) 
        R_12 = 2*(Q_rel[2]*Q_rel[3] - Q_rel[0]*Q_rel[1])
        R_20 = 2*(Q_rel[1]*Q_rel[3] - Q_rel[0]*Q_rel[2])
        R_22 = (Q_rel[0]*Q_rel[0]) - (Q_rel[1]*Q_rel[1]) - (Q_rel[2]*Q_rel[2]) + (Q_rel[3]*Q_rel[3])
        R_21 = 2*(Q_rel[2]*Q_rel[3] + Q_rel[0]*Q_rel[1])

        target_qpos = model_qpos if info.get("target") == "model" else hand_qpos
        if info["DOF"] == 1:
            target_qpos[info["qpos"]] = -math.atan2(R_10, R_00)
        elif info["DOF"] == 2:
            dof0 = -math.atan2(-R_01, R_11)
            dof1 = -math.atan2(-R_20, R_22)
            target_qpos[info["qpos"][0]] = dof0
            target_qpos[info["qpos"][1]] = dof1
        elif info["DOF"] == 3:
            target_qpos[info["qpos"][0]] = math.atan2(R_21, R_22)
            target_qpos[info["qpos"][1]] = math.atan2(-R_20, math.sqrt(R_00**2 + R_10**2))
            target_qpos[info["qpos"][2]] = math.atan2(R_10, R_00)

    return model_qpos

   



