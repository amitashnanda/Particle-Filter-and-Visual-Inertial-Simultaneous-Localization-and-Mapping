# Visual-Inertial-Simultaneous-Localization-and-Mapping
Implemented visual-inertial SLAM using an extended Kalman filter using IMU and stereo camera measurements from an autonomous car. First performed IMU localization via EKF prediction, then landmark mapping via EKF update.

## Folder Structure

```
Visual-Inertial-Simultaneous-Localization-and-Mapping
│   _pycache_
└───data
│   │   03.npz
|   └───10.npz   
└───report
└───results
└───src
|    |   dead_reck.py
|    |   vslam.py
|    └───pr3_utils.py
└───LICENCE 
└───README.md
└───requirments.txt


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
The EKF SLAM created a more accurate map and trajectory compared to just the IMU localization prediction and landmark mapping via EKF update. The algorithm is performed on two datasets provided to us (03.npz and 10.npz). IMU based localization, the trajectory is determined as follows, the dead reckoning Map for both datasets are shown below. The trajectory from the IMU-based localization is assumed to be correct and EKF-update is performed for landmark mapping. The world frame is plotted using all the features. After the successful implementation of the algorithm, the map displays the car trajectory and environment landmarks. The landmark features are tried to down sampled for 10.npz datasets but with full feature the code has ran. While for 03.npz it has been shown till the 1900 iteration. The results of EKF landmark mapping, the, blue points are closely aligned with the trajectory of the car in red colour. With the variation in noise the best map is created as shown below.
</p>
  

![](/results/Selection_003.png)
![](/results/Dead_reck_03.png)
![](/results/Dead_Reck_10.png)
![](/results/Figure_03npz.png)
![](/results/Figure_1.png)

