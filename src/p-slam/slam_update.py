import numpy as np
import matplotlib.pyplot as plt
from pymacaroons import Macaroon
from pr2_utils import *
from lidar_process import lidar_data_process

'''
all parameters declaration

'''

'''
encoder parameters
'''
encoder_res = 4096
left_dia = 0.623479
right_dia = 0.622806

'''
particles parameters
'''

particle_count = 10
particle_weights = np.zeros((1, particle_count))
particle_state = np.zeros((particle_count, 3))

'''
variable definitions
'''

MAP = {}
log_odds_ratio = np.log(4)
dt_yaw = np.zeros(116048)
dt = np.zeros(116048)
X = np.zeros((116048, 3))
X[0] = np.array([0,0,0])
traj = np.zeros((1,2))
theta = 0
encoder_counter = 0
lidar_counter = 0

'''
all function definitions

'''


'''
sensor data initialize 

'''


def map_data_initialize():

    lidar_timestamp, lidar_data = read_data_from_csv('/home/amitash/Documents/UCSD /ECE-276A/ECE276A_PR2/code/data/sensor_data/lidar.csv')
    encoder_timestamp, encoder_data = read_data_from_csv('/home/amitash/Documents/UCSD /ECE-276A/ECE276A_PR2/code/data/sensor_data/encoder.csv')
    fog_timestamp, fog_data = read_data_from_csv('/home/amitash/Documents/UCSD /ECE-276A/ECE276A_PR2/code/data/sensor_data/fog.csv')
    return lidar_timestamp, lidar_data, encoder_timestamp, encoder_data, fog_timestamp, fog_data


'''
encoder and fog data syncronization

'''


def encoder_fog_data_sync(fog_data_n, encoder_timestamp_n):

    dt_yaw_n = np.zeros(116048)
    dt = np.zeros(116048)
    for i in range( len(encoder_timestamp_n) - 1 ):
        dt_yaw_n[i] = sum( fog_data_n[ ((i - 1)*10 + 1) : (i*10 + 1), 2 ] )
        dt[i] = (encoder_timestamp_n[i+1] - encoder_timestamp_n[i] ) * 10**(-9) 
    return dt_yaw_n, dt


'''
added noise
'''   

def map_noise(particle_state_n, particle_count_n, dt):

    state_arr = np.zeros((particle_count_n, 5))
    for i in range(particle_count_n):
        noise_linval = np.random.normal(0, 0.5)
        noise_angval = np.random.normal(0, 0.05)
        state_arr[i,0] = particle_state_n[i,0] + (noise_linval * dt)
        state_arr[i,1] = particle_state_n[i,1] + (noise_linval * dt)
        state_arr[i,2] = particle_state_n[i,2] + (noise_angval * dt)
    return state_arr

'''
map initialization
'''


def map_initialize(MAP):

    MAP['res'] = 1 # Meters
    MAP['xmin'] = -100  # Meters
    MAP['ymin'] = -1200 # Meters
    MAP['xmax'] = 1300 # Meters
    MAP['ymax'] = 200  # Meters
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float32) #DATA TYPE: char or int8

    x_im_n = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
    y_im_n = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

    x_range_n = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res'])
    y_range_n = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res'])


    return MAP, x_im_n, y_im_n, x_range_n, y_range_n

'''
calculation of map correlation
updated particle weights with Softmax function
determined the particles best matches with the map

'''


def map_correlation(lidar_data_n, particle_count_n, particle_state_n, particle_weights_n, MAP, x_im_n, y_im_n, x_range_n, y_range_n):

    correlation = np.zeros(particle_count_n)
    for i in range(particle_count_n):
        xs0_n, ys0_n = lidar_data_process(lidar_data_n, particle_state_n[i])
        Y = np.stack((xs0_n,ys0_n))
        corr = mapCorrelation(MAP['map'], x_im_n , y_im_n , Y, x_range_n, y_range_n )
        correlation[i] = np.max(corr)


    max_corr = np.max(correlation)
    a = np.exp(correlation - max_corr)
    b = a/a.sum()
    particle_weights_n = particle_weights_n* (b/ np.sum(particle_weights_n * b))

   
    match_particle_pos  = np.argmax(particle_weights_n)
    match_particle_state_n = particle_state_n[match_particle_pos, :]

    return match_particle_state_n

'''
updated the map
converted from meters to cells
updated map using log-odds ratio
decreased the log-odds for free cells
increased the log-odds for occupied cells
clipped map to the max and min values 

'''

def map_update(MAP, xs0_n, ys0_n, particle_state_n, log_odds):


    xis = np.ceil((xs0_n - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((ys0_n - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

    x_o = np.ceil((particle_state_n[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    y_o = np.ceil((particle_state_n[1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

    out = np.zeros((2,0))
    for i in range(len(xis)):
            bresenham_output = bresenham2D(x_o, y_o, xis[i], yis[i])
            out = np.hstack((out, bresenham_output))

    bres_x = out[0,:].astype(int)
    bres_y = out[1,:].astype(int)
    
    indGood = np.logical_and(np.logical_and(np.logical_and((bres_x > 1), (bres_y > 1)), (bres_x < MAP['sizex'])), (bres_y < MAP['sizey']))

    MAP['map'][ bres_x[indGood] , bres_y[indGood]] += log_odds


    for i in range(len(xis)):
            if (xis[i] > 1) and (xis[i] < MAP['sizex']) and yis[i] > 1 and (yis[i] < MAP['sizey']):
                MAP['map'][ xis[i] , yis[i] ] -= log_odds


    MAP['map'] = np.clip(MAP['map'], -10*log_odds, 10*log_odds)


    return MAP

'''
draws the map trajectory
'''

def map_trajectory(traj_n, MAP):

    x_traj = np.ceil((traj_n[:,0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    y_traj = np.ceil((traj_n[:,1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

    return x_traj, y_traj

'''
plots the output map and output wall


'''

def map_plot(MAP, x_traj, y_traj):

    # final_map = ((1 - 1 / (1 + np.exp(MAP['map']))) < 0.1).astype(np.int)
    final_wall = ((1 - 1 / (1 + np.exp(MAP['map']))) > 0.9).astype(np.int)

    
    # plt.imshow(MAP['map'], cmap = 'gray')
    # plt.imshow(final_map, cmap = 'gray')
    plt.imshow(final_wall, cmap = 'gray')
    plt.plot(y_traj, x_traj, color='orangered', linewidth=0.5)
    plt.title("Occupancy Grid Map")
    plt.xlabel("x grid-cell coordinates")
    plt.ylabel("y grid-cell coordinates")
    plt.show(block=True)
    return 0

'''
function call


'''

'''
loaded data from the sensor
initialized the map
synced encoder and fog data
initialzed particle weights
updated the first lidar scan

'''
lidar_timestamp, lidar_data, encoder_timestamp, encoder_data, fog_timestamp, fog_data = map_data_initialize()
MAP, x_im, y_im, x_range, y_range = map_initialize(MAP)
dt_yaw,dt =  encoder_fog_data_sync(fog_data,encoder_timestamp)
particle_weights[0, 0:particle_count] = 1/particle_count
xs0, ys0 = lidar_data_process(lidar_data[lidar_counter,:], X[0] )
MAP = map_update( MAP, xs0, ys0, X[0], log_odds_ratio)

'''
slam_update running main loop

'''


for encoder_counter in range(len(encoder_timestamp)-1):

    l_wheel_dis = ((encoder_data[encoder_counter+1,0] - encoder_data[encoder_counter,0]) * np.pi * left_dia )/ encoder_res
    r_wheel_dis = ((encoder_data[encoder_counter+1,1] - encoder_data[encoder_counter,1]) * np.pi * right_dia )/ encoder_res

    dt_distance = (l_wheel_dis + r_wheel_dis) / 2

    theta = theta + dt_yaw[encoder_counter]

    # particle_state = particle_state + np.array( [dt_distance * np.cos(theta), dt_distance * np.sin(theta), theta ])
    # particle_state = map_noise( particle_state, particle_count, dt[encoder_counter] )


 # particle state change
    del_x = dt_distance * np.cos(particle_state[:,2] + dt_yaw[encoder_counter])
    del_y = dt_distance * np.sin(particle_state[:,2] + dt_yaw[encoder_counter])

    # new particle state
    particle_state[:,0] += del_x
    particle_state[:,1] += del_y
    particle_state[:,2] += dt_yaw[encoder_counter]

    # add noise to particles
    particle_state[:,0] += np.random.normal(0, abs(np.max(del_x)) / 10, particle_count)
    particle_state[:,1] += np.random.normal(0, abs(np.max(del_y)) / 10, particle_count)
    particle_state[:,2] += np.random.normal(0, abs(dt_yaw[encoder_counter]) / 10, particle_count)
    


    if lidar_timestamp[lidar_counter] < encoder_timestamp[encoder_counter]:

        match_particle_state = map_correlation(lidar_data[lidar_counter,:], particle_count, particle_state, particle_weights, MAP, x_im, y_im, x_range, y_range)
        '''
        code to run with particles
        '''
        
        xs0, ys0 = lidar_data_process(lidar_data[lidar_counter,:], match_particle_state)
        MAP = map_update(MAP, xs0, ys0, match_particle_state,log_odds_ratio )

        '''
        code to run without particles
        '''
        # xs0, ys0 = lidar_data_process(lidar_data[lidar_counter,:], X[encoder_counter] )
        # MAP = map_update( MAP, xs0, ys0, X[encoder_counter], log_odds_ratio )

        print('Lidar Counter:', lidar_counter)
        lidar_counter += 5

    X[encoder_counter] = match_particle_state
    traj = np.concatenate((traj, np.array([[X[encoder_counter,0], X[encoder_counter,1]]])), axis = 0    )

    if (lidar_counter >= len(lidar_data)):
        break

    if encoder_counter % 50000 == 0:

        x_final,y_final = map_trajectory(traj, MAP)
        map_plot(MAP, x_final,y_final)
        print('Encoder Counter:', encoder_counter)
        print('Lidar Counter:', lidar_counter)

    print('Encoder Counter:', encoder_counter)


X = X[~np.all(X == 0, axis=1)]
x_final,y_final = map_trajectory(X, MAP)
map_plot(MAP, x_final,y_final)


    
