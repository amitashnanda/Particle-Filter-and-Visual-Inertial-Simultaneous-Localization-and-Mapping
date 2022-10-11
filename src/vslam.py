import numpy as np
from pr3_utils import *
from scipy.linalg import expm, inv 
 

class visual_inertial_slam():
	def __init__(self):
		# load the data provided to us # 
		filename = "D:/ECE_276A_Project/ECE-276PR3/code/data/10.npz"
		# filename = "D:/ECE_276A_Project/ECE-276PR3/code/data/03.npz"

		# call the load_data function from pr3_utils.py #
		self.t, self.features, self.linear_velocity, self.angular_velocity, self.K, self.b, self.imu_T_cam = load_data(filename)

		# printing the parameters values #

		print(self.t)
		print(self.features)
		print(self.linear_velocity)
		print(self.angular_velocity)
		print(self.K)
		print(self.b)
		print(self.imu_T_cam)

		# Checking the parameter shape #

		print(np.shape(self.t))                
		print(np.shape(self.features))   
		print(np.shape(self.linear_velocity))  
		print(np.shape(self.angular_velocity))  
		print(np.shape(self.K))                
		print(np.shape(self.b))                
		print(np.shape(self.imu_T_cam)) 

    
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



		# initializing parameters features, mean and covariance for landmark #
		
		self.main_f_count = np.shape(self.features)[1]
		self.imu2cam = inv(self.imu_T_cam)
		self.f = self.features[:, 0:self.main_f_count:5,:]
		self.f_time = np.shape(self.f)[2]
		self.f_count = np.shape(self.f)[1]
		self.var_n = 90
		self.mu_lm =  -1 * np.ones((4, self.f_count))
		self.sigma_lm = np.identity(3*self.f_count) * self.var_n
		self.t_array =  np.array([[1, 0, 0], 
								[0, 1, 0], 
								[0, 0, 1],
								[0, 0, 0]])
	
		# initializing mean and variance for imu pose and trajectory
		self.imu_mu = np.identity(4)
		self.imu_sig = np.identity(6)
		self.pose_size = (4, 4,self.f_time)
		self.imu_traj = np.zeros(self.pose_size)
		self.imu_traj[:,:,0] = self.imu_mu 

		self.linear_velocity_n = np.transpose(self.linear_velocity)
		self.angular_velocity_n = np.transpose(self.angular_velocity)
		self.ini = np.kron(np.identity(self.f_count ), self.t_array)

		# Extrinsics of stereo camera definition via the Matrix M #

		self.stereo_fsu = self.K[0][0]
		self.stereo_cu = self.K[0][2]
		self.stereo_fsv = self.K[1][1]
		self.stereo_cv = self.K[1][2]

		self.M_matrix = np.array([[self.stereo_fsu, 0, self.stereo_cu, 0],[0, self.stereo_fsv , self.stereo_cv, 0],[self.stereo_fsu, 0, self.stereo_cu, -self.stereo_fsu * self.b],[0, self.stereo_fsv,self.stereo_cv, 0]])




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
			
	# Landmark mapping via EKF Update and Visual Inertial SLAM in one function #
	
	
	def vslam_prediction(self):
		for step in range(self.f_time):
			print('No of Iteration:', step)
			self.lv_t = self.linear_velocity_n[step,:]
			self.av_t = self.angular_velocity_n[step,:]
			self.um_t = np.vstack((self.lv_t,self.av_t))

			self.lv_hat = np.array([[0, -self.lv_t[2], self.lv_t[1]], [self.lv_t[2], 0, -self.lv_t[0]], [-self.lv_t[1], self.lv_t[0], 0]])
			self.av_hat = np.array([[0, -self.av_t[2], self.av_t[1]], [self.av_t[2], 0, -self.av_t[0]], [-self.av_t[1], self.av_t[0], 0]])

			self.um_hat = np.array([[self.av_hat[0][0], self.av_hat[0][1], self.av_hat[0][2], self.lv_t[0]], [self.av_hat[1][0],self.av_hat[1][1], self.av_hat[1][2],self.lv_t[1]], [self.av_hat[2][0], self.av_hat[2][1],self.av_hat[2][2], self.lv_t[2]], [0, 0, 0, 0]])

			self.um_adj = np.array([[self.av_hat[0][0], self.av_hat[0][1], self.av_hat[0][2], self.lv_hat[0][0], self.lv_hat[0][1], self.lv_hat[0][2]],
                                [self.av_hat[1][0], self.av_hat[1][1], self.av_hat[1][2], self.lv_hat[1][0], self.lv_hat[1][1], self.lv_hat[1][2]],
                                [self.av_hat[2][0], self.av_hat[2][1], self.av_hat[2][2], self.lv_hat[2][0], self.lv_hat[2][1], self.lv_hat[2][2]],
                                [0, 0, 0, self.av_hat[0][0], self.av_hat[0][1], self.av_hat[0][2]],
                                [0, 0, 0, self.av_hat[1][0], self.av_hat[1][1], self.av_hat[1][2]],
                                [0, 0, 0, self.av_hat[2][0], self.av_hat[2][1], self.av_hat[2][2]]])

			self.tau_m = self.t[0, step] - self.t[0, step-1]
			self.W_t = np.diag(np.random.normal(0,1,6))
			self.W = self.tau_m * self.tau_m * self.W_t

			self.mu_p = np.dot(expm(-self.tau_m * self.um_hat ),self.imu_mu)
			self.sigma_p = np.dot(np.dot(expm(-self.tau_m * self.um_adj), self.imu_sig), np.transpose(expm(-self.tau_m * self.um_adj))) + self.W
			
			# world to camera an dcamer to world transformation #
			self.t_w2c = np.dot(self.imu2cam, self.imu_mu)
			self.t_c2w =  inv(self.t_w2c)

			# current landmark feature #

			self.f_c = self.f[:, :, step]
			self.s_f = np.sum(self.f_c[:, :], 0)

			# finding noticeable feature and it's coordinates #

			self.f_idx = np.array(np.where(	self.s_f != -4))
			self.f_idx_c = np.size(self.f_idx)
			self.f_up =  np.zeros((4, 0))
			self.f_up_idx = np.zeros((0, 0), dtype=np.int8)

			if self.f_idx_c > 0:

				self.f_n_cord = self.f_c[:, self.f_idx].reshape(4, self.f_idx_c)
				self.f_n = np.ones((4, np.shape(self.f_n_cord)[1]))
				self.f_n[0, :] = (self.f_n_cord[0, :] - self.stereo_cu) * self.b / (self.f_n_cord[0, :] - self.f_n_cord[2, :])
				self.f_n[1, :] = (self.f_n_cord[1, :] - self.stereo_cv) * (-self.M_matrix[2, 3]) / (self.M_matrix[1, 1] * (self.f_n_cord[0, :] - self.f_n_cord[2, :]))
				self.f_n[2, :] = -(-self.stereo_fsu * self.b) / (self.f_n_cord[0, :] - self.f_n_cord[2, :])
				self.f_n = np.dot(self.t_c2w, self.f_n)

			# noticeable features landmark estimation # 
				for i in range(self.f_idx_c):
					index = self.f_idx[0, i] 

					if np.array_equal(self.mu_lm[:, index], [-1, -1, -1, -1]):
						self.mu_lm[:, index] = self.f_n[:, i]

					else:
						self.f_up = np.hstack((self.f_up, self.f_n[:, i].reshape(4, 1)))
						self.f_up_idx = np.append(self.f_up_idx, index)

				self.f_up_count = np.shape(self.f_up_idx)[0]

				# looking for new noticeable features #

				if self.f_up_count != 0:
					
					self.mu_lm_d = (4,  self.f_up_count)
					self.mu_lm_n = self.mu_lm[:, self.f_up_idx]
					self.mu_lm_n.reshape(self.mu_lm_d)
					self.f_tot = self.f_count 
					self.proj_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])


					for j in range(self.f_up_count):
						
						# projection equation and derivate calculation #
						proj_i = np.dot(self.t_w2c, self.mu_lm[:, j])
						proj_q_1 = proj_i[0]
						proj_q_2 = proj_i[1]
						proj_q_3 = proj_i[2]
						proj_q_4 = proj_i[3]

						der_proj = (1/proj_q_3) * (np.array([[1, 0, -proj_q_1/proj_q_3, 0],[0, 1, -proj_q_2/proj_q_3, 0],[0, 0, 0, 0],[0, 0, -proj_q_4/proj_q_3, 1]]))
						
						# jacobian of the noticeable model #
						self.h =  np.zeros((4 * self.f_up_count, 3 * self.f_tot))
						self.h[4*j:4*(j+1), 3*index:3*(index+1)] = np.dot(np.dot(np.dot(self.M_matrix, der_proj),self.t_w2c), np.transpose(self.proj_mat))

					# landmark mean, variance update and coordinates in world frame #
					inverse = inv(np.dot(np.dot(self.h, self.sigma_lm), np.transpose(self.h)) + np.identity(4 * self.f_up_count) * self.var_n )
					K_new = np.dot(np.dot(self.sigma_lm, np.transpose(self.h)),inverse)
					q_mat = np.dot(self.t_w2c, self.mu_lm_n)
					proj_m = q_mat /q_mat[2,:]
					self.zh= np.dot(self.M_matrix, proj_m)
					self.lm = self.f_c[:, self.f_up_idx].reshape((4, self.f_up_count))
					self.mu_lm = (self.mu_lm.reshape(-1, 1, order='F') + np.dot(np.dot(self.ini, K_new), (self.lm - self.zh).reshape(-1, 1, order='F'))).reshape(4, -1, order='F')
					self.sigma_lm = np.dot((np.identity(3 * np.shape(self.f )[1]) - np.dot(K_new, self.h)), self.sigma_lm)

			# IMU update step based on stereo camera observation model #
			self.imu_traj[:, :, step] = inv(self.mu_p)
			self.imu_mu = self.mu_p
			self.imu_sig = self.sigma_p
		# visualizes the trajectory and landmark, call the function in Pr3.utils.py #
		return visualize_trajectory_2d(self.imu_traj, self.mu_lm, show_ori=True)


# class call #

predict = visual_inertial_slam()
predict.ekf_prediction()
# predict.vslam_prediction()