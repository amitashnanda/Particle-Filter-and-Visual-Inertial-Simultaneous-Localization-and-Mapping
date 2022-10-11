import numpy as np
from pr3_utils import *
from scipy.linalg import expm

if __name__ == '__main__':

    # Load the measurements
    filename = "D:/ECE_276A_Project/ECE-276PR3/code/data/10.npz"
    t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(
        filename)

    # (a) IMU Localization via EKF Prediction

    # hatmap of a vector
    def hatmap(x):
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])

    # u^
    def uhat(v, w):
        return np.block([[hatmap(w), v.reshape((3, 1))],
                         [np.zeros((1, 3)), 0]])

    # uc^
    def ucurly(v, w):
        return np.block([[hatmap(w), hatmap(v)],
                         [np.zeros((3, 3)), hatmap(w)]])

    # create arrays
    print(t.shape[1])
    pose = np.zeros((4, 4, t.shape[1]), dtype=float)
    print(pose)
    mu = np.zeros((4, 4, t.shape[1]), dtype=float)
    delmu = np.zeros((6, 1, t.shape[1]), dtype=float)
    sigma = np.zeros((6, 6, t.shape[1]), dtype=float)

    # intialize values
    mu0 = np.identity(4)
    sigma0 = np.zeros((6, 6))
    delmu0 = np.diag(np.random.normal(0, sigma0)).reshape((6, 1))

    pose[:, :, 0] = mu0
    # print(mu0)
    mu[:, :, 0] = mu0
    delmu[:, :, 0] = delmu0
    sigma[:, :, 0] = sigma0

    # noise
    W = np.diag(np.zeros(6))
    # W = np.diag([0.3, 0.3, 0.3, 0.05, 0.05, 0.05])

    # trajectory prediction
    for i in range(t.shape[1] - 1):
        v = linear_velocity[:, i]
        w = angular_velocity[:, i]
        uh = uhat(v, w)
        uc = ucurly(v, w)
        tau = t[0, i+1] - t[0, i]
        wt = np.diag(np.random.normal(0, W)).reshape((6, 1))

        mu[:, :, i+1] = np.matmul(mu[:, :, i], expm(tau*uh))
        delmu[:, :, i+1] = np.matmul(expm(-tau*uc), delmu[:, :, i]) + wt
        sigma[:, :, i+1] = expm(-tau*uc) @ sigma[:, :, i] @ expm(-tau*uc).T + W

        delmuh = uhat(delmu[:3, 0, i], delmu[3:, 0, i])
        pose[:, :, i+1] = np.matmul(mu[:, :, i], expm(delmuh))

    # visualize trajectory
    visualize_trajectory_2d(pose, path_name="Unknown", show_ori=True)
