import numpy as np


class Pca():

	def __init__(self,x):
		self.x = x
		self.m = x.shape[0]


	def sigEval(self):
		"""
		Covariance matrix
		"""
		self.sigma = (1/self.m) * np.dot(np.transpose(self.x),self.x)

	def dim_reduction(self, k):
		"""
		Reducing dimensions from n(input) to k using PCA
		"""
		self.sigEval()
		self.u, self.s, _ = np.linalg.svd(self.sigma)
		self.u_red = self.u[:,:k]
		self.z = np.dot(self.x,self.u_red)

		covar_retain = (np.sum(self.s[:k])/np.sum(self.s[:]))*100

		print(covar_retain,"percent of covariance retained")

		return self.z

	def mean_normalization(self):                       
		"""
		Feature scaling using mean normalization method
		"""
		self.avg = np.mean(self.x, axis = 0)
		self.ran = np.max(self.x, axis = 0)-np.min(self.x, axis = 0)
		self.x = (self.x - self.avg)/self.ran



