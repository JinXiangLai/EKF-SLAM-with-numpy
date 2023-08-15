from os import abort
import numpy as np
import math
import scipy.spatial.transform as sst
from generate_map_imu_pose import euler2Rotation


def str_list2float_list(all_data, data_start_idx=0):
    result = []
    for item in all_data:
        if item[0][0] == '#':
            continue
        data = item.strip().split(' ')  # throw \n and split number
        data_type = data[0]
        data = list(map(float, data[data_start_idx:]))  # str to float
        if data_start_idx == 0:
            result.append(data)
        elif data_start_idx == 1:
            result.append((data_type, data))
    return result


def read_map_points(filename: str):
    map_points = []
    print('\n********** Reading map points **********\n')
    with open(filename) as file:
        map_points = file.readlines()
    map_points = str_list2float_list(map_points)  # OK
    return map_points


def read_imu_pose_data(filename: str):
    imu_pose_data = []
    print('\n********** Reading imu pose data **********\n')
    with open(filename) as file:
        imu_pose_data = file.readlines()
    imu_pose_data = str_list2float_list(imu_pose_data, 1)  # OK
    return imu_pose_data


# quaternion
def quaternion_from_matrix(matrix, isprecise=False):
    '''
        return [qx, qy, qz, qw]
    '''
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    q = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64, copy=True)
    q = q / np.linalg.norm(q)
    return q


# Lie glgebra
def is_so3(r: np.ndarray) -> bool:
    # Check the determinant.
    det_valid = np.allclose(np.linalg.det(r), [1.0], atol=1e-6)
    # Check if the transpose is the inverse.
    inv_valid = np.allclose(r.transpose().dot(r), np.eye(3), atol=1e-6)
    return det_valid and inv_valid


def so3_exp(rotation_vector: np.ndarray):
    return sst.Rotation.from_rotvec(rotation_vector).as_matrix()


def hat(v: np.ndarray) -> np.ndarray:
    # yapf: disable
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])
    # yapf: enable


def so3_log(r: np.ndarray, return_skew: bool = False) -> np.ndarray:
    """
    :param r: SO(3) rotation matrix
    :param return_skew: return skew symmetric Lie algebra element
    :return:
            rotation vector (axis * angle)
        or if return_skew is True:
             3x3 skew symmetric logarithmic map in so(3) (Ma, Soatto eq. 2.8)
    """
    if not is_so3(r):
        print("matrix is not a valid SO(3) group element")
        abort()
    rotation_vector = sst.Rotation.from_matrix(r).as_rotvec()
    if return_skew:
        return hat(rotation_vector)
    else:
        return rotation_vector


def angle_velocity2rotation(w, delta_t):
    wx = w[0] * delta_t
    wy = w[1] * delta_t
    wz = w[2] * delta_t

    # Rodrigues formula
    d2 = wx * wx + wy * wy + wz * wz
    d = np.sqrt(d2)

    v = np.array([wx, wy, wz])
    W = hat(v)
    eps = 1e-4
    if (d < eps):
        deltaR = np.eye(3) + W
    else:
        deltaR = np.eye(3) + W * np.sin(d) / d + W.dot(W) * (1.0 -
                                                             np.cos(d)) / d2
    return deltaR


def InverseRightJacobianSO3(vec):
    I = np.eye(3)
    x = vec[0]
    y = vec[1]
    z = vec[2]
    d2 = x * x + y * y + z * z
    d = np.sqrt(d2)
    W = hat(vec)
    eps = 1e-4
    if (d < eps):
        return I
    else:
        return I + W / 2 + W.dot(W) * (1.0 / d2 - (1.0 + math.cos(d)) /
                                       (2.0 * d * math.sin(d)))


class ExternKalmanFilter(object):

    def __init__(self) -> None:
        # noise
        self.acc_noise = 0.1
        self.gyro_noise = 0.01
        self.Q = np.eye(6)
        self.Q[:3, :3] *= self.acc_noise
        self.Q[3:, 3:] *= self.gyro_noise
        self.obs_noise = 0.01

        #position, velocity, rotation in lie algebra
        self.x_k = np.zeros(9)
        self.gravity = np.array([0, 0, -9.8])
        self.P_k = np.eye(9)
        self.P_k[:3, :3] *= self.acc_noise
        self.P_k[3:6, 3:6] *= self.acc_noise
        self.P_k[6:, 6:] *= self.gyro_noise

        self.map_points = []

    def predict(self, gyro, acc, delta_t):
        '''
            P_k+1 = [P_k + v_k * Δt] + 0.5 * R*a * t^2 + g * t^2
            v_k+1 = [v_k] + R*a * Δt + g * Δt
            # BCH 近似可以对 Log(R * exp(w^)) 进行线性化
            # Log(R * exp(w^)) = Log(R) + Jr(Log(R)).inv * w
            # w = so3(ΔR), ΔR = angle_velocity2rotation(w, delta_t)
            Log(R_k+1) = [Log(R)] + Jr(Log(R)).inv * Log(ΔR)
        '''
        F = np.eye(9)
        # about position
        F[:3, 3:6] = np.eye(3) * delta_t
        # about velocity
        # about rotation

        w = so3_log(angle_velocity2rotation(gyro, delta_t))
        input = np.array([acc[0], acc[1], acc[2], w[0], w[1], w[2]])
        R_k = so3_exp(self.x_k[6:])

        B = np.zeros((9, 6))
        # about position
        B[:3, :3] = 0.5 * R_k * math.pow(delta_t, 2)
        # about velocity
        B[3:6, :3] = R_k * delta_t
        # about rotation vector
        B[6:, 3:] = InverseRightJacobianSO3(self.x_k[6:])

        G = np.zeros((9, 3))
        G[:3, :3] = np.eye(3) * math.pow(delta_t, 2)
        G[3:6, :3] = np.eye(3) * delta_t

        self.x_k = F.dot(self.x_k) + B.dot(input) + G.dot(self.gravity)
        self.P_k = F.dot(self.P_k).dot(F.transpose()) + B.dot(self.Q).dot(
            B.transpose())

    def update(self, p_true, R_true, delta_t=0):
        obs_true = []
        obs_pred = []
        p_pred = self.x_k[:3]
        R_pred = so3_exp(self.x_k[6:])
        p_w_idx = []
        for i in range(len(self.map_points)):
            point_w = np.array(self.map_points[i])
            point_c_true = R_true.transpose().dot(
                point_w) - R_true.transpose().dot(p_true)
            point_c_pred = R_pred.transpose().dot(
                point_w) - R_pred.transpose().dot(p_pred)

            min_z = 0.01
            max_norm = 1
            if (point_c_true[2] > min_z and point_c_pred[2] > min_z):
                p1 = np.array(
                    [point_c_true[0], point_c_true[1], point_c_true[2]])
                p2 = np.array(
                    [point_c_pred[0], point_c_pred[1], point_c_pred[2]])
                norm = np.linalg.norm(p1 - p2)
                if norm < max_norm:
                    obs_true.append(
                        [point_c_true[0], point_c_true[1], point_c_true[2]])
                    obs_pred.append(
                        [point_c_pred[0], point_c_pred[1], point_c_pred[2]])
                    p_w_idx.append(i)

        print("obs num: ", len(p_w_idx))
        if len(p_w_idx) == 0:
            print('No obs')
            abort()
            return

        # linearise observation function
        '''
                  p   v  R
            obs0 
            obs1
            obs2
            ==> 雅可比矩阵为
            ∂obs        ∂obs     ∂Pc
           ------    = ------ * -----
            ∂p(or R)    ∂Pc      ∂p(or R)
            2 X 3    = 2 X 3  * 3 X 3
        '''
        # calculate jacobians
        H = np.zeros((len(obs_pred) * 2, 9))
        t_pred = self.x_k[:3]
        for i in range(len(obs_pred)):
            p = obs_pred[i]
            p_w = np.array(self.map_points[p_w_idx[i]])
            # obs w.r.t position
            obs_wrt_p_c = np.array([[1. / p[2], 0, -p[0] / math.pow(p[2], 2)],
                                    [0, 1. / p[2], -p[1] / math.pow(p[2], 2)]])
            p_wrt_tw = -1. * R_pred.transpose()
            vector = R_pred.transpose().dot(p_w - t_pred)
            p_wrt_Rw = hat(vector)
            # about position
            H[i * 2:i * 2 + 2, :3] = obs_wrt_p_c.dot(p_wrt_tw)
            # obs w.r.t rotation
            H[i * 2:i * 2 + 2, 6:] = obs_wrt_p_c.dot(p_wrt_Rw)

        # add obs noise to every observation
        #     nx1 ny1 nx2 ny2 nx3 ny3 ...
        # nx1  1
        # ny1      1
        # ...
        R = np.eye(2 * len(obs_pred))
        R *= self.obs_noise

        K = 0
        try:
            K = self.P_k.dot(H.transpose()).dot(
                np.linalg.inv(H.dot(self.P_k).dot(H.transpose()) + R))
        except:
            print('H\n', H)
            print('P_k\n', self.P_k)
            print('for inv\n', H.dot(self.P_k).dot(H.transpose()) + R)
            abort()

        x_obs_true = []
        x_obs_pred = []
        for i in range(len(obs_pred)):
            p_c_true = obs_true[i]
            p_c_pred = obs_pred[i]
            x_obs_true.append([
                p_c_true[0] / p_c_true[2] +
                np.random.normal(loc=0, scale=self.obs_noise),
                p_c_true[1] / p_c_true[2] +
                np.random.normal(loc=0, scale=self.obs_noise)
            ])
            x_obs_pred.append(
                [p_c_pred[0] / p_c_pred[2], p_c_pred[1] / p_c_pred[2]])
        x_obs_true = np.array(x_obs_true)
        x_obs_pred = np.array(x_obs_pred)

        # be careful for the array shape
        self.x_k = self.x_k + (K.dot(
            (x_obs_true - x_obs_pred).reshape(-1, 1))).reshape(1, -1)[0]
        self.P_k = (np.eye(9) - K.dot(H)).dot(self.P_k)

    def run(self, imu_pose_data):
        idx = -1
        cur_time = 0
        # initialize state
        while idx < len(imu_pose_data) - 1:
            idx += 1
            name = imu_pose_data[idx][0]
            data = imu_pose_data[idx][1]
            cur_time = data[0]
            if (name == 'POSE'):
                self.x_k[:3] = np.array(data[1:4])
                self.x_k[3:6] = np.array(data[4:7])
                R = euler2Rotation(data[7:])
                self.x_k[6:] = so3_log(R)

                break

        # process every data
        true_file = open('true_pose.csv', 'w')
        pred_file = open('pred_pose.csv', 'w')
        while idx < len(imu_pose_data) - 1:
            idx += 1
            name = imu_pose_data[idx][0]
            data = imu_pose_data[idx][1]
            delta_t = data[0] - cur_time
            cur_time = data[0]
            # predict
            if name == 'IMU':
                gyro = data[1:4]
                acc = data[4:]
                # add noise to gyro and acc
                gyro = [
                    i + np.random.normal(loc=0, scale=self.gyro_noise)
                    for i in gyro
                ]
                acc = [
                    i + np.random.normal(loc=0, scale=self.acc_noise)
                    for i in acc
                ]
                self.predict(gyro, acc, delta_t)
            # update
            elif name == 'POSE':
                p_true = np.array(data[1:4])
                R_true = euler2Rotation(data[7:])
                state_true = np.array(data[1:])
                print('diff.norm: ', np.linalg.norm(state_true - self.x_k))
                true_pose = str(cur_time)
                pred_pose = str(cur_time)
                for i in range(3):
                    true_pose += ' ' + str(p_true[i])
                    pred_pose += ' ' + str(self.x_k[i])

# TUM style
                quat_true = quaternion_from_matrix(R_true)
                quat_pred = quaternion_from_matrix(euler2Rotation(
                    self.x_k[6:]))
                for i in range(4):
                    true_pose += ' ' + str(quat_true[i])
                    pred_pose += ' ' + str(quat_pred[i])
                true_pose += '\n'
                pred_pose += '\n'
                true_file.write(true_pose)
                pred_file.write(pred_pose)
                # self.update(p_true, R_true, delta_t)
        true_file.close()
        pred_file.close()


def run():
    map_file = 'house.txt'
    simulate_file = 'imu_pose_simulate_data.csv'
    map_points = read_map_points(map_file)
    imu_pose_data = read_imu_pose_data(simulate_file)
    ekf = ExternKalmanFilter()
    ekf.map_points = map_points
    ekf.run(imu_pose_data)


if __name__ == '__main__':
    run()