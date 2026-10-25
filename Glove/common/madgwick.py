from .Quat_Operation import *

def madgwick_filter(w,q_est_old,a,delta_t,Beta):
    w = vect3_to_pure_quat(w)
    q_dot = 0.5*quat_Mult(q_est_old,w) 

    a = vect3_to_pure_quat(Vect3_Normalise(a))
    f_g  = np.zeros(3)
    f_g[0] = 2*(q_est_old[1]*q_est_old[3] - q_est_old[0]*q_est_old[2]) - a[1]
    f_g[1] = 2*(q_est_old[0]*q_est_old[1] + q_est_old[2]*q_est_old[3]) - a[2]
    f_g[2] = 2*( 0.5 - q_est_old[1]**2 - q_est_old[2]**2 ) - a[3]

    J_g  = np.zeros((3,4))
    J_g[0,0] = -2*q_est_old[2]
    J_g[1,0]= 2*q_est_old[1]
    J_g[2,0] = 0
    J_g[0,1] = 2*q_est_old[3] 
    J_g[1,1] = 2*q_est_old[0]
    J_g[2,1] = -4*q_est_old[1]
    J_g[0,2] = -2*q_est_old[0]
    J_g[1,2] = 2*q_est_old[3]
    J_g[2,2] = -4*q_est_old[2]
    J_g[0,3] = 2*q_est_old[1]
    J_g[1,3] = 2*q_est_old[2]
    J_g[2,3] = 0

    f_grad = J_g.T @ f_g
    f_grad = quat_Normalise(f_grad)

    q_dot_est = q_dot - Beta*f_grad
    q_est = q_est_old + q_dot_est*delta_t 
    q_est = quat_Normalise(q_est)
    
    return q_est
