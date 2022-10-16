import numpy as np
import matplotlib.pyplot as plt
from pr2_utils import *


'''
    variable declaration
    Lidar to Body Transformation provided values
    Lidar Parameters
    Other Variable Definition
    
'''
rpy = np.array([[142.759,0.0584636, 89.9254]])
r_l2b = np.array([[0.00130201, 0.796097 ,0.605167],[0.999999, -0.000419027, -0.00160026],[-0.00102038 ,0.605169 ,-0.796097 ]])
t_l2b = np.array([[0.8349 , -0.0126869 , 1.76416]])

fov = 190
start_angle = -5
end_angle = 185
angular_resolution = 0.666
max_range = 80


def lidar_data_process(lidar_data, particle_state):


    coord = list()
    angle = 0
    x_s_0 = np.zeros(286)
    y_s_0 = np.zeros(286)

    '''
        x, y coordinates are calculated from lidar data
    
    '''
    for i in range(len(lidar_data)):
        angle = start_angle + (i*angular_resolution)
        a = [0,0]
        
        '''
        lidar data is filtered with the range 2<d<75

        '''

        if lidar_data[i] > 2 and lidar_data[i] < 75:
            a[0] = lidar_data[i] * (np.cos(np.deg2rad(angle)))
            a[1] = lidar_data[i] * (np.sin(np.deg2rad(angle)))
            coord.append(a)

    '''
    convereted to lidar to world frame
    pose matrix is formed from lidar to body frame transfromation
    
    '''

    pose_l2b = np.concatenate((r_l2b, t_l2b.T), axis = 1)
    pose_l2b = np.concatenate((pose_l2b, np.array([[0,0,0,1]])), axis = 0)

    theta = particle_state[2]

    '''
    pose matrix is formed from body to world frame transfromation
    '''

    r_b2w = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0, 0, 1]])


    pose_b2w = np.concatenate((r_b2w, np.array([[particle_state[0], particle_state[1], 0]]).T), axis = 1)
    pose_b2w = np.concatenate((pose_b2w, np.array([[0,0,0,1]])), axis = 0)

    '''
    final pose formulated for lidar to world frame transformation

    '''

    pose = np.dot(pose_b2w, pose_l2b)


    for i in range(len(coord)):
        a = [coord[i][0], coord[i][1], 0, 1 ]
        a = np.dot(pose, a)
        x_s_0[i] = a[0]
        y_s_0[i] = a[1]

    x_s_0 = x_s_0[x_s_0 != 0]
    y_s_0 = y_s_0[y_s_0 != 0]

    return  x_s_0, y_s_0

