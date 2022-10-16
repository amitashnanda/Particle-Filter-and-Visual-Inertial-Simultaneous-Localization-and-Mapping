import pandas as pd
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from pr2_utils import read_data_from_csv
from pr2_utils import bresenham2D

class prediction():

    def __init__(self):

        self.encoder_res = 4096
        self.left_dia = 0.623479
        self.right_dia = 0.622806

        self.encoder_time, self.encoder_data = read_data_from_csv('/home/amitash/Documents/UCSD /ECE-276A/ECE276A_PR2/code/data/sensor_data/encoder.csv')
        self.fog_time, self.fog_data = read_data_from_csv('/home/amitash/Documents/UCSD /ECE-276A/ECE276A_PR2/code/data/sensor_data/fog.csv')

        self.X = np.zeros((len(self.encoder_time),3))
        self. theta = 0
        self.index = 0
        self.counter = 0


    def dead_reck(self):
    
    

        for i in range (1, len(self.encoder_time)):
            X_left_encoder = math.pi*self.left_dia*(self.encoder_data[i][0]-self.encoder_data[i-1][0])/self.encoder_res
            X_right_encoder = math.pi*self.right_dia*(self.encoder_data[i][1]-self.encoder_data[i-1][1])/self.encoder_res
            X_average = (X_left_encoder + X_right_encoder)/2

            for self.counter in range(10):
                if self.fog_time[i+self.index+self.counter] >= self.encoder_time[i]:
                    break
                self. theta += self.fog_data[i+self.index+self.counter][2]  

            self.index += self.counter

            read_value = np.array([X_average*math.cos(self.theta),X_average*math.sin(self.theta),self. theta])
            self.X[i]=self.X[i-1]+read_value

        return self.X  

class_call = prediction()
p = class_call.dead_reck()
plt.plot(p[:,0],p[:,1], color = 'orangered')
plt.gca().set_aspect("equal")
plt.show(block=True)



