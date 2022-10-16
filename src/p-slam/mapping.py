import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from pr2_utils import read_data_from_csv
from pr2_utils import bresenham2D

class mapping():
    def __init__(self):

        '''
        variable declaration
        Lidar to Body Transformation provided values
        Lidar Parameters
        Other Variable Definition
        
        '''
        self.rpy = np.array([[142.759,0.0584636, 89.9254]])
        self.r = np.array([[0.00130201, 0.796097 ,0.605167],[0.999999, -0.000419027, -0.00160026],[-0.00102038 ,0.605169 ,-0.796097 ]])
        self.t = np.array([[0.8349 , -0.0126869 , 1.76416]])

        self.fov = 190
        self.start_angle = -5
        self.end_angle = 185
        self.angular_resolution = 0.666
        self.max_range = 80

        self.timestamp = np.zeros([115865,1])
        self.data = np.zeros([115865,286])

        self.timestamp, self.data= read_data_from_csv('/home/amitash/Documents/UCSD /ECE-276A/ECE276A_PR2/code/data/sensor_data/lidar.csv')

        self.angle = 0
        self.theta = 0
        self.lidar_frame_cod = list()
        self.world_frame_cod = list()
        self.temp_array = np.array([[0,0,0,1]]) 
        self.x_sf = np.zeros(286)
        self.y_sf = np.zeros(286)
        self.final = np.zeros((2,0))

        

    def lidar_first_scan(self):
        '''
        Initialize the MAP

        '''

        MAP = {}
        MAP['res']   = 0.1 
        MAP['xmin']  = -50  
        MAP['ymin']  = -50
        MAP['xmax']  =  50
        MAP['ymax']  =  50 
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) 
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
        MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) 

        

        for i in range(self.data.shape[1]):

            '''
            Calculate x and y co-ordinates from the Lidar Value

            '''
            
            self.angle = self.start_angle + self.theta
            lst = [0,0]
            lst[0] = self.data[0,i]*(np.cos(np.deg2rad(self.angle)))
            lst[1] = self.data[0,i]*(np.sin(np.deg2rad(self.angle)))
            self.lidar_frame_cod.append(lst)
            self.theta += self.angular_resolution 

        '''
        Lidar to Body Frame conversion   
        '''
        # print(self.t.T.shape)

        pose = np.concatenate((self.r,self.t.T),axis = 1)
        pose = np.concatenate((pose,self.temp_array),axis = 0)

        for i in range(len(self.lidar_frame_cod)):
            lst = [self.lidar_frame_cod[i][0],self.lidar_frame_cod[i][1],0,1]
            lst = np.dot(pose,lst)
            self.x_sf[i] = lst[0]
            self.y_sf[i] = lst[1]

        '''
        Convert from meters to cells
        '''
        xis = np.ceil((self.x_sf - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        yis = np.ceil((self.y_sf - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1    

        x_origin = np.ceil((0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        y_origin = np.ceil((0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1 


        

        output = np.zeros((2,0))

        for i in range(len(xis)):
            bresenham_output = bresenham2D(x_origin, y_origin, xis[i], yis[i])
            output = np.hstack((output, bresenham_output))

        xis= output[0,:].astype(int)
        yis= output[1,:].astype(int)

        '''
        build an arbitrary map

        '''



        indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
        MAP['map'][xis[indGood],yis[indGood]]=1


        '''
        plot a map
        '''

        fig = plt.figure(figsize=[10,10])
        plt.imshow(MAP['map'],cmap="gray")
        plt.title("Lidar First Scan")
        plt.xlabel("x grid-cell coordinates")
        plt.ylabel("y grid-cell coordinates")
        plt.show(block=True)


        '''
        Mapping Class call
        Lidar_First_Scan function call
        '''

c = mapping()
c.lidar_first_scan()


                





