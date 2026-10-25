import numpy as np
import math
from .Quat_Operation import *
from.Const_INFO import *

def mahony_filter(w,q_est_old,a,delta_t,K_p,K_i,int_accum_old):
 
    g_b = np.zeros(3)
    g_b[0] =  2*(q_est_old[1]*q_est_old[3] - q_est_old[0]*q_est_old[2])
    g_b[1] = 2*(q_est_old[2]*q_est_old[3] + q_est_old[0]*q_est_old[1])
    g_b[2] =  q_est_old[0]**2 - q_est_old[1]**2 - q_est_old[2]**2 + q_est_old[3]**2

    e = np.cross(g_b.astype(float), a.astype(float))

    int_accumulation = int_accum_old + e*delta_t

    delta = K_p*e + K_i*int_accumulation

    w_q = w - delta
    w_q = vect3_to_pure_quat(w_q)
    q_dot = 0.5*quat_Mult(q_est_old,w_q)
    q_est = q_est_old + q_dot*delta_t

    q_est = quat_Normalise(q_est)

    return q_est,int_accumulation