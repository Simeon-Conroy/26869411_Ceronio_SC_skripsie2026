import mujoco
import mujoco.viewer

def discover_sensors(model):
    accel_lookup = {}
    gyro_lookup = {}
    for sensor_id in range(model.nsensor):
        sensor_type = model.sensor_type[sensor_id]
        if sensor_type not in (mujoco.mjtSensor.mjSENS_ACCELEROMETER,mujoco.mjtSensor.mjSENS_GYRO,):
            continue

        site_id = model.sensor_objid[sensor_id]
        site_name = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_SITE,site_id,)

        sensor = { "adr": int(model.sensor_adr[sensor_id]),"dim": int(model.sensor_dim[sensor_id]),}

        if sensor_type == mujoco.mjtSensor.mjSENS_ACCELEROMETER:
            accel_lookup[site_name] = sensor
        else:
            gyro_lookup[site_name] = sensor

    return accel_lookup, gyro_lookup