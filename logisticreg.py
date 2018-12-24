import numpy as np



class Log_reg():
	def __init__(self,x,y):
		self.y = y
		self.m = len(y)
		self.n = x.shape[1]
		xo = np.ones(shape = (m,1))
		self.x = np.hstack((xo,x))
		self.t = np.zeros(shape = (self.n+1,1))

	@np.vectorize
	def cost_function(self):
		"""
		Evaluates the cost 
		"""
		self.yt = np.transpose(y)
		self.hypo = 1/(1+np.exp(-np.matmul(self.x,self.t)))
		self.cost = (-1/self.m)*(np.matmul(self.yt,np.log(self.hypo))+np.matmul((1-self.yt),np.log(1-self.hypo)))
		return self.cost

	@np.vectorize
	def gradient_descent(self,alpha = 0.01, stop = 0.001):
		"""
		Optimization function
		alpha is the learning rate and stop determines the smallest gradient
		"""
		while True:
			print("Cost at current state: ",self.cost_function())
			old_t = self.t
			self.t = self.t - (alpha/self.m)*np.matmul(np.transpose(self.x),(self.hypo-self.y))   #gradient calculation
			cond = (np.abs(old_t - self.t) > stop)
			if not cond.any():                                                #stopping condition
				print("gradient is small so the process is stopped")
				break
		return self.t


	@np.vectorize
	def regularisation(self,l = 100, alpha = 0.01, stop = 0.001):
		"""
		Regularizing to prevent overfitting
		regularization by shrinking parameters
		"""
		
		while True:
			self.cost = self.cost_function()+(l/(2*self.m))*np.sum(self.t[1:,]**2)
			print("Cost at current state: ",self.cost)
			old_t = self.t
			self.t = self.t - (alpha/self.m)*np.matmul(np.transpose(self.x),(self.hypo-self.y))   #gradient calculation
			self.t[1:,] = self.t[1:,] - (alpha/self.m)*l*old_t[1:,]
			cond = (np.abs(old_t - self.t) > stop)
			if not cond.any():                                                #stopping condition
				print("gradient is small so the process is stopped")
				break
		return self.t





