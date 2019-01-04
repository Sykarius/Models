import numpy as np
import matplotlib.pyplot as plt


class Linear_Reg():

	def __init__(self,x,y):
		self.y = y  #Label
		self.m = len(y)       #number of training samples
		self.n = x.shape[1]  #number of features
		xo = np.ones(shape = (m,1))
		self.x = np.hstack((xo,x))  #adding first column as ones(Data)
		self.t = np.zeros(shape = (self.n+1,1))   #regression parameters
	
	@np.vectorize
	def cost_function(self):
		"""
		Evaluates cost at current state 
		"""
		self.hypo = np.matmul(self.x,self.t)
		self.cost = (1/(2*self.m))*np.sum((self.hypo - self.y)**2)        #cost calculation
		return self.cost

	@np.vectorize
	def gradient_descent(self,aplha = 0.001,stop = 0.0001):  #optimization method
		"""
		optimization function
		alpha is the learning rate and stop is the smallest gradient limit
		"""
		while True:
			print("Current cost = ",self.cost_function())
			old_t = self.t
			self.t = self.t - (alpha/self.m)*np.matmul(np.transpose(self.x),(self.hypo - self.y))    #descent
			cond = (np.abs(old_t - self.t) > stop)
			if not cond.any():                                                #stopping condition
				print("gradient is small so the process is stopped")
				break
		return self.t

	@np.vectorize
	def mean_normalization(self):                       
		"""
		Feature scaling using mean normalization method
		"""
		self.avg = np.mean(self.x, axis = 0)  #avg
		self.ran = np.max(self.x, axis = 0)-np.min(self.x, axis = 0) #range
		self.x = (self.x - self.avg)/self.ran

	@np.vectorize
	def cramer_eq(self):
		"""
		Another optimization function that uses cramer rule.
		"""
		x_trans = np.transpose(self.x)    #solving takes a lot of time as size of input matrix increase
		deno = np.inverse(np.matmul(x_trans,self.x))
		numer = np.matmul(x_trans,self.y)
		self.t = np.matmul(deno,numer)
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


