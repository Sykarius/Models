import numpy as np



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
			print("Current cost = ",cost_function())
			old_t = self.t
			self.t = self.t - (alpha/self.m)*matmul(np.transpose(self.x),(self.hypo - self.y))    #descent
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
		avg = np.mean(self.x, axis = 0)  #avg
		ran = np.max(self.x, axis = 0)-np.min(self.x, axis = 0) #range
		self.x = (self.x - avg)/ran

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
