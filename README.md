# Particle-Filter-and-Visual-Inertial-Simultaneous-Localization-and-Mapping
Autonomous navigation requires precise and robust mapping and localization solutions in real-world scenarios. Simultaneous Localization and Mapping (SLAM) is widely used in solving this problem. SLAM usage ranges from mobile robotics and self-driving cars to unmanned aerial and underwater autonomous vehicles. It uses data from the sensors to perform mapping and localization simultaneously. Particle filter is one of the most adapted estimation algorithms for SLAM apart from the Kalman Filter and Extended Kalman Filter. This project discusses an approach to solving the SLAM problem for an autonomous vehicle and attempts to understand the given implementation's shortcomings. We successfully implemented the differential-drive motion and scan-grid correlation observation models for simultaneous localization and occupancy-grid mapping. We also implemented visual-inertial simultaneous localization and mapping (SLAM) using python's extended Kalman filter (EKF). EKF is used over popular sparse SLAM algorithms like Particle Filter and Factor Graphs SLAM. This is because EKF uses prediction and update steps to track an autonomous system over time and estimate landmark positions. Our project involves three main steps:

A. IMU Localization via EKF Prediction
B. Landmark Mapping via EKF Update
C. Visual Inertial SLAM

## Folder Structure

```
Visual-Inertial-Simultaneous-Localization-and-Mapping
│   _pycache_
└───data
│   │   p-slam
|   └───v-slam   
└───report
└───results
└───src   
|    |   p-slam
|    └───v-slam
└───LICENCE 
└───README.md
└───requirments_pslam.txt
└───requirements_vslam.txt


```

## Prerequisites
The dependencies are listed under requirements.txt and are all purely python based. To install them simply run
```
pip install -r requirements.txt
```

## Dataset
The datset is there inside the data folder.

## Running
For a local installation, make sure you have pip installed and run: 
```
pip install notebook
```
For running jupyter
```
jupyter notebook
```

The entire code is written in the vslam.py file.

```visual_inertial_slam()``` is the main class consists of two function ```predict.ekf_prediction()``` ```predict.vslam_prediction()```
To run the IMU Localization via EKF prediction to get deadreckon images
comment ```predict.vslam_prediction()``` and run the main file

To run the Landmark Mapping and V-SLAM comment the ```predict.ekf_prediction()``` and run the main file.


## Results
<p align = "justify">
In this project we have implemented the particle filter SLAM for 𝑛 = 5, 10, 20 particles. We have plotted the log-odds occupancy map and binary map at various intervals for particularly 𝑛 = 5 particle case. Also, we have updated the map only at every 5 LiDAR readings for every case. Gaussian
noise is used for each particles state using zero mean and variance of max (∆𝑥/10), max (∆𝑦/10) and max (∆𝜃/10). The dead-reckoning trajectory path is also plotted prior to using particle filter. We noticed that particle filter SLAM gives almost same trajectory as the dead-reckoning trajectory
with some variations. One particular thing to note is the realization of the lane on which the vehicle traverses which becomes very conspicuous from the SLAM implementation. We observed that the change in the number of particles doesn’t have much of an effect on the map or the vehicle trajectory. The trajectory aligns well through out the journey. All the plots are plotted below. The EKF SLAM created a more accurate map and trajectory compared to just the IMU localization prediction and landmark mapping via EKF update. The algorithm is performed on two datasets provided to us (03.npz and 10.npz). IMU based localization, the trajectory is determined as follows, the dead reckoning Map for both datasets are shown below. The trajectory from the IMU-based localization is assumed to be correct and EKF-update is performed for landmark mapping. The world frame is plotted using all the features. After the successful implementation of the algorithm, the map displays the car trajectory and environment landmarks. The landmark features are tried to down sampled for 10.npz datasets but with full feature the code has ran. While for 03.npz it has been shown till the 1900 iteration. The results of EKF landmark mapping, the, blue points are closely aligned with the trajectory of the car in red colour. With the variation in noise the best map is created as shown below.
</p>
  
 Dead_reckoning            |  First LiDAR Scan
:-------------------------:|:-------------------------:
 ![](/results/p-slam/Dead_Reckoning.png)  |  ![](/results/p-slam/First_Lidar_Scan.png)
![](/results/p-slam/binary_map/Figure_1.png)  |  ![](/results/p-slam/binary_map/Figure_2.png)
![](/results/p-slam/binary_map/Figure_3.png)  |  ![](/results/p-slam/binary_map/Figure_4.png)
![](/results/p-slam/binary_map/Figure_5.png)  |  ![](/results/p-slam/binary_map/Figure_7.png)
 
<!--   Occupancy Grid Map          |  Occupancy Grid Map 
:-------------------------:|:-------------------------: -->
 ![](/results/p-slam/binary_map/Figure_1.png)  |  ![](/results/p-slam/binary_map/Figure_2.png)

  
  Occupancy Grid Map          |  Occupancy Grid Map 
:-------------------------:|:-------------------------: -->
 ![](/results/p-slam/binary_map/Figure_3.png)  |  ![](/results/p-slam/binary_map/Figure_4.png)
 
  
  Occupancy Grid Map          |  Occupancy Grid Map 
:-------------------------:|:-------------------------: -->
 ![](/results/p-slam/binary_map/Figure_5.png)  |  ![](/results/p-slam/binary_map/Figure_7.png)



 -->


