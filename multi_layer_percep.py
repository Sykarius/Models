import numpy as np
import matplotlib.pyplot as plt


class Perceptron():

	def __init__(self,x,y):
		xo = np.ones(shape = (x.shape[0],1))
		self.x = np.hstack((xo,x))
		self.y = y
		self.m = x.shape[0]
		self.n = x.shape[1]


	def set_params(self, n_nodes, n_hlayers, n_classes):
		"""
		Setting parameters for Neural network model
		n_nodes is the number of nodes in each hidden layer in a form of an iterable
		n_hlayer gives number of hidden layer
		n_classes is the number of output nodes or the number of classes to classify
		"""
		self.n_nodes = list(n_nodes)
		self.n_nodes.insert(0,self.n)
		self.n_nodes.insert(-1,n_classes)
		self.n_hlayers = n_hlayers
		self.n_classes = n_classes

		if len(self.n_nodes)!=self.n_hlayers+2:
			print("Number nodes and layers mismatch")
			raise ValueError


	def init_weights(self):
		"""
		Initializing weights to random values.(small values)
		"""
		self.theta = []
		for i in range(len(self.n_nodes)-1):
			a = np.random.normal(size = (self.n_nodes[i+1],self.n_nodes[i]+1))
			self.theta.append(a)

	
	
	def sigmoid(self,z):
		return 1/(1+np.exp(-z))

	
	
	def grad_sigmoid(self,z):
		return self.sigmoid(z)*(1-self.sigmod(z))


	def tanh(self,z):
		return np.tanh(z)

	
	def grad_tanh(self,z):
		return 1-np.tanh(z)**2

	
	def set_activation(self,act):
		"""
		choosing the activation functions
		0 -> sigmoid
		1 ->tanh
		2->relu
		"""
		self.act = act


	
	def activate(self,z):
		"""
		Runs the chosen activation function
		"""
		if self.act == 0:
			self.sigmoid(z)
		elif self.act == 1:
			self.tanh(z)
		else:
			ValueError


	
	def activate_grad(self,z):
		"""
		Gradient of the chosen activation
		"""
		if self.act == 0:
			self.grad_sigmoid(z)
		elif self.act == 1:
			self.grad_tanh(z)
		else:
			raise ValueError


	
	def feedforward(self, inp = None):
		"""
		Feed forward through the network
		"""
		if inp == None:
			a = self.bx
		else:
			a  = inp

		zlist = []
		
		for i in range(self.n_hlayers):
			z = np.dot(a,np.transpose(self.theta[i]))
			zlist.append(z)
			a = self.activate(z)
			ao = np.ones(shape = (a.shape[0],1))
			a = np.hstack((ao,a))

		z = np.dot(a,np.transpose(self.theta[-1]))
		return z,zlist

	
	def sigmoid_cross_entropy(self):
		z,_ = self.feedforward()
		a = self.sigmoid(z)
		self.cost = (-1/self.batch_size)*np.sum(self.by*np.log(a) + (1-self.by)*np.log(1 - a))

	
	def sigmoid_squared_error(self):
		z,_ = self.feedforward()
		a = self.sigmoid(z)
		self.cost = (1/(2*self.batch_size))*(self.by-a)**2

	
	def softmax_cross_entropy(self):
		pass


	def set_cost_function(self,cf):
		"""
		Setting which cost function to use
		0-> sigmoid cross entropy
		1-> sigmoid cross
		"""
		self.cf = cf


	def cost_function(self):
		if self.cf == 0:
			self.sigmoid_cross_entropy()
		elif self.cf == 1:
			self.sigmoid_squared_error()
		else:
			raise ValueError

	
	def cost_grad(self,z):
		if self.cf == 0:
			a = self.sigmoid(z)
			return a - self.by
		elif self.cf ==1:
			a = self.sigmoid(z)
			return  (a-self.by) * a * (1-a)
		else:
			raise ValueError


	
	def backprop(self):
		"""
		Backpropogation function for finiding the gradients.
		"""
		d = []
		z,zlist = self.feedforward()
		a = self.activate(z)
		d.append(self.cost_grad())
		
		self.theta_grad = [np.zeros(shape = t.shape) for t in self.theta]
		
		for i in range(n_hlayers):
			a_grad = self.activate_grad(zlist[::-1][i])
			d.append(np.dot(d[i],self.theta[::-1][i])[:,1:]*a_grad)

		d.reverse()

		self.theta_grad[0] = self.theta_grad[0] + np.dot(np.transpose(d[0]),self.bx)
		for i in range(1,len(self.theta_grad)):
			a = self.activate(zlist[i-1])
			ao = np.ones(shape = (a.shape[0],1))
			a = np.hstack(ao,a)
			self.theta_grad[i] = self.theta_grad[i] + np.dot(np.transpose(d[i]),a)

		self.theta_grad = [(1/self.batch_size)*tg for tg in self.theta_grad]






	def train(self, alpha = 0.01, batch_size = 100,stop  = 0.000001):
		"""
		Traines the network
		Split to batches
		Mini-Batch gradient descent
		"""
		self.batch_size = batch_size
		self.init_weights()

		while True:
			self.bx = self.x[i:i+self.batch_size,:]
			self.by = self.y[i:i+self.batch_size,:]

			print("Cost at current state: ",self.cost_function())

			self.backprop()

			for i in range(len(self.theta)):
				self.theta[i] = self.theta[i] - (alpha*self.theta_grad[i])   #updating weights

			i+=self.batch_size
			if i+batch_size>self.m:
				break

			for i in self.theta_grad:
				cond = (np.abs(i)>stop)
				if not cond.any():
					break
			else:
				print("Gradient is small so the process is stopped")
				break

		print("Trained!!")


	def predict(self,inp):
		"""
		Run this function to get predictions
		"""
		z,_ = self.feedforward(inp)
		i = np.argmax(z)

		return i









