import numpy as np
from pr3_utils import *
from scipy.linalg import expm


class dead_reck_slam():
	def __init__(self):
		# load the data provided to us # 
		filename = "D:/ECE_276A_Project/ECE-276PR3/code/data/10.npz"
		# filename = "D:/ECE_276A_Project/ECE-276PR3/code/data/03.npz"

		# call the load_data function from utils.py #
		self.t, self.features, self.linear_velocity, self.angular_velocity, self.K, self.b, self.imu_T_cam = load_data(filename)

		# defining matrices for variables used in prediction step       

		self.shape_t = self.t.shape[1]	
		self.pose_matrix = np.zeros((4, 4 , self.shape_t), dtype = float)
		self.mu_matrix =  np.zeros((4, 4 , self.shape_t), dtype = float)
		self.delta_mu_matrix = np.zeros((6, 1, self.shape_t), dtype = float)
		self.sigma_matrix = np.zeros((6, 6, self.shape_t), dtype = float)
		self.mu_matrix_new = np.identity(4)
		self.sigma_matrix_new = np.zeros((6,6))
		self.delta_mu_matrix_new = np.diag(np.random.normal(0,self.sigma_matrix_new)).reshape((6, 1))

		self.pose_matrix[:,:,0] = self.mu_matrix_new
		self.mu_matrix [:,:,0] = self.mu_matrix_new
		self.delta_mu_matrix[:,:,0] =self.delta_mu_matrix_new
		self.sigma_matrix[:,:,0] = self.sigma_matrix_new
		self.W_n = np.diag(np.zeros(6))


# creates the hat of the provided vector
	def hatmap_vector(self,x):
		self.get_hat = np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])
		return self.get_hat

# returns u_hat and u_curlyhat and takes linear and angular velocities as input through self declaration
	def ekf_operators(self, lin_vel, ang_vel):
		self.lin_vel_hat = self.hatmap_vector(lin_vel)
		self.ang_vel_hat = self.hatmap_vector(ang_vel)

		self.u_hat = np.block([[self.ang_vel_hat, lin_vel.reshape((3, 1))],[np.zeros((1, 3)), 0]])
		self.u_curly_hat = np.block([[self.ang_vel_hat, self.lin_vel_hat],[np.zeros((3, 3)),self.ang_vel_hat]])
		return self.u_hat, self.u_curly_hat

# IMU localization via EKF prediction (prediction of the trajectory) #

	def ekf_prediction(self):
		for step in range(self.shape_t - 1):
			lin_vel = self.linear_velocity[:, step]
			ang_vel = self.angular_velocity[:,step]
			self.u_hat, self.u_curly_hat = self.ekf_operators(lin_vel, ang_vel)
			self.tau = self.t[0,step+1] - self.t[0,step]
			self.w_t = np.diag(np.random.normal(0,self.W_n)).reshape((6,1))
			self.mu_matrix[:,:,step+1] = np.matmul(self.mu_matrix[:,:,step], expm(self.tau*self.u_hat))
			self.delta_mu_matrix[:, :, step+1] = np.matmul(expm(-self.tau*self.u_curly_hat),self.delta_mu_matrix[:, :, step]) + self.w_t
			self.sigma_matrix[:, :, step+1] = expm(-self.tau*self.u_curly_hat) @ self.sigma_matrix[:, :, step] @ expm(-self.tau*self.u_curly_hat).T + self.W_n
			self.delta_mu_hat, self.delta_muc_hat = self.ekf_operators(self.delta_mu_matrix[:3, 0, step],self.delta_mu_matrix[3:, 0, step])
			self.pose_matrix[:, :, step+1] = np.matmul(self.mu_matrix[:, :, step], expm(self.delta_mu_hat))

		return visualize_trajectory_2d_deadreckon(self.pose_matrix, path_name="Unknown", show_ori=True)
			

predict = dead_reck_slam()
predict.ekf_prediction()









		

		









	

