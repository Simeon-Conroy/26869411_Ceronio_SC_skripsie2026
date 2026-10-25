import numpy as np
import math
from .Quat_Operation import *
from.Const_INFO import *

def delta_q_acc_calculate(G_g_p):

    norm = np.linalg.norm(G_g_p)
    if norm < 1e-8:
        return unit_quat()
    
    G_g_p = G_g_p / norm

    G_g_p = G_g_p / np.linalg.norm(G_g_p)

    g_x = G_g_p[0]
    g_y = G_g_p[1]
    g_z = G_g_p[2]

    g_z = np.clip(g_z, -1.0, 1.0)

    delta_q_acc = np.zeros(4)

    if g_z < -0.999999:
        delta_q_acc[0] = 0.0
        delta_q_acc[1] = 1.0
        delta_q_acc[2] = 0.0
        delta_q_acc[3] = 0.0
    else:
        denominator = np.sqrt(2.0 * (g_z + 1.0))
        delta_q_acc[0] = denominator / 2.0
        delta_q_acc[1] = -g_y / denominator
        delta_q_acc[2] = g_x / denominator
        delta_q_acc[3] = 0.0

    return quat_Normalise(delta_q_acc)

def Complementary_Filter(w,accel,q_est_old,delta_t,alpha):
    q_est_old_GL = inv_conj_quat(q_est_old) 

    w_q = vect3_to_pure_quat(w)
    q_dot = -0.5*quat_Mult(w_q,q_est_old_GL)

    q_GL_u = q_est_old_GL + q_dot*delta_t
    q_GL = quat_Normalise(q_GL_u)

    R_GL = quat_to_R(q_GL)
    R_LG = R_GL.T
    G_g_p = R_LG@accel
    
    delta_q_acc = delta_q_acc_calculate(G_g_p)

    Eps = 0.9
    q_I = unit_quat()
    Omega = np.arccos(delta_q_acc[0])

    if delta_q_acc[0] > Eps:
        delta_bar_q_acc = (1-alpha)*q_I + alpha*delta_q_acc
        delta_hat_q_acc = delta_bar_q_acc/(quat_Norm(delta_bar_q_acc))
    elif delta_q_acc[0] <= Eps:
        delta_hat_q_acc = ((np.sin((1-alpha)*Omega))/(np.sin(Omega)))*q_I + ((np.sin(alpha*Omega))/(np.sin(Omega)))*delta_q_acc

    q_est_GL = quat_Mult(delta_hat_q_acc,q_GL)

    q_est_LG = inv_conj_quat(q_est_GL) 
    return quat_Normalise(q_est_LG)



