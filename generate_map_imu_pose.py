import numpy as np
import math

_EPS = np.finfo(float).eps * 4.0

def euler2Rotation(eulerAngles):
    roll = eulerAngles[0]
    pitch = eulerAngles[1]
    yaw = eulerAngles[2]

    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    RIb = np.array([[cy * cp, cy * sp * sr - sy * cr, sy * sr + cy * cr * sp],
                    [sy * cp, cy * cr + sy * sr * sp, sp * sy * cr - cy * sr],
                    [-sp, cp * sr, cp * cr]])
    return RIb


def eulerRates2bodyRates(eulerAngles):
    roll = eulerAngles[0]
    pitch = eulerAngles[1]

    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)

    R = np.array([[1, 0, -sp], [0, cr, sr * cp], [0, -sr, cr * cp]])
    return R


def generate_simulate_imu_pose_data(t): 
    #  gravity
    gravity = np.array([0., 0., -9.8])
    ellipse_x = 15
    ellipse_y = 20
    z = 1
    K1 = 10
    K = math.pi / 10 # means 20 * K = 2pi 

    # translation
    position = np.array([
        ellipse_x * math.cos(K * t) + 5, ellipse_y * math.sin(K * t) + 5,
        z * math.sin(K1 * K * t) + 5
    ])
    dp = np.array([
        -K * ellipse_x * math.sin(K * t), K * ellipse_y * math.cos(K * t),
        z * K1 * K * math.cos(K1 * K * t)
    ]) 
    K2 = K * K
    ddp = np.array([
        -K2 * ellipse_x * math.cos(K * t), -K2 * ellipse_y * math.sin(K * t),
        -z * K1 * K1 * K2 * math.sin(K1 * K * t)
    ]) 

    # Rotation
    k_roll = 0.1  # 0.1
    k_pitch = 0.2  #0.2
    K = 0.3
    eulerAngles = np.array(
        [k_roll * math.cos(t), k_pitch * math.sin(t), K * math.cos(t)])
    eulerAnglesRates = np.array(
        [-k_roll * math.sin(t), k_pitch * math.cos(t), -K * math.sin(t)])
    Rwb = euler2Rotation(eulerAngles)
    #  euler rates trans to body gyro
    imu_gyro = eulerRates2bodyRates(eulerAngles).dot(
        eulerAnglesRates)  

    imu_acc = Rwb.transpose().dot(ddp - gravity)

    return t, imu_gyro, imu_acc, eulerAngles, position, dp


def run():
    t_start = 0.
    t_end = 20.
    t = t_start
    imu_frequency = 100.
    with open('imu_pose_simulate_data.csv', 'w') as file:
        while (t < t_end):
            time, gyro, acc, R_wb, P_wb, Vw = generate_simulate_imu_pose_data(
                t)

            if t > 0.:
                content = 'IMU ' + str(time)
                for i in gyro:
                    content += ' ' + str(i)
                for i in acc:
                    content += ' ' + str(i)
                content += '\n'
                file.write(content)

            content = 'POSE ' + str(time)
            for i in P_wb:
                content += ' ' + str(i)
            for i in Vw:
                content += ' ' + str(i)
            for i in R_wb:
                content += ' ' + str(i)
            content += '\n'
            file.write(content)

            t += 1.0 / imu_frequency


if __name__ == '__main__':
    run()