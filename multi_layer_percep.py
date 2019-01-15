import numpy as np
import matplotlib.pyplot as plt


class Perceptron():
	
	"""
	Functions to call after initializing the object are:
	
	set_params -> to set parameters of the network
	set_activation -> to choose the activation function
	set_cost_function -> to choose how the cost function is calculated
	train -> to train the network
	predict -> after the training to get the predicted value

	If the label is not encoded then you can use the one_hot_encoder
	for one hot encoding
	"""

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


	def relu(self,z):
		return np.maximum(0,z)


	def grad_relu(self,z):
		return z>0


	
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
		
		elif self.act == 2:
			self.relu(z)
		
		else:
			raise ValueError


	
	def activate_grad(self,z):
		"""
		Gradient of the chosen activation function
		"""
		if self.act == 0:
			self.grad_sigmoid(z)
		
		elif self.act == 1:
			self.grad_tanh(z)
		
		elif self.act ==2:
			self.grad_relu(z)
		
		else:
			raise ValueError


	
	def feedforward(self, inp = None):
		"""
		Feed forward through the network
		Return the output and also the input of each layer
		"""
		if inp == None:
			a = self.bx
		else:
			a  = inp

		zlist = []
		
		for i in range(self.n_hlayers):
			z = np.dot(a,np.transpose(self.theta[i]))   #Getting input from previous layer and summation
			zlist.append(z)
			a = self.activate(z)                        #passing the summation value through the activation function
			ao = np.ones(shape = (a.shape[0],1))		
			a = np.hstack((ao,a))						#adding activation value of bias unit(1)

		z = np.dot(a,np.transpose(self.theta[-1]))
		return z,zlist

	
	
	def sigmoid_cross_entropy(self):
		z,_ = self.feedforward()
		a = self.sigmoid(z)
		t_sum = sum([np.sum(t[:,1:]**2) for t in self.theta])   #Regularisation term
		self.cost = (-1/self.batch_size)*np.sum(self.by*np.log(a) + (1-self.by)*np.log(1 - a)) + (self.lam/(2*self.batch_size))*t_sum

	
	
	def sigmoid_squared_error(self):
		z,_ = self.feedforward()
		a = self.sigmoid(z)
		t_sum = sum([np.sum(t[:,1:]**2) for t in self.theta])   #Regularisation term
		self.cost = (1/(2*self.batch_size))*np.sum(self.by-a)**2 + (self.lam/(2*self.batch_size))*t_sum

	
	
	def softmax_cross_entropy(self):
		z,_ = self.feedforward()
		a = np.exp(z)
		a = a/np.sum(a,axis = 1)
		t_sum = sum([np.sum(t[:,1:]**2) for t in self.theta])   #Regularisation term
		self.cost = (-1/self.batch_size)*np.sum(self.by*np.log(a) + (1-self.by)*np.log(1 - a)) + (self.lam/(2*self.batch_size))*t_sum


	
	def set_cost_function(self,cf):
		"""
		Setting which cost function to use
		0-> sigmoid cross entropy
		1-> sigmoid squared error
		2-> softmax cross entropy
		"""
		self.cf = cf


	
	def cost_function(self):
		"""
		Running the chosen cost function
		"""
		
		if self.cf == 0:
			self.sigmoid_cross_entropy()
		
		elif self.cf == 1:
			self.sigmoid_squared_error()
		
		elif self.cf == 2:
			self.softmax_cross_entropy()
		
		else:
			raise ValueError

	
	
	def cost_grad(self,z):
		"""
		Chosing how to calculate error at output layer depending
		cost function used
		"""
		
		if self.cf == 0:
			a = self.sigmoid(z)
			return a - self.by
		
		elif self.cf ==1:
			a = self.sigmoid(z)
			return  (a-self.by) * a * (1-a)
		
		elif self.cf ==2:
			a = np.exp(z)
			a = a/np.sum(a,axis = 1)
			return a-self.by
		
		else:
			raise ValueError


	
	def backprop(self):
		"""
		Backpropogation function for finiding the gradients.
		"""
		d = []
		z,zlist = self.feedforward()
		d.append(self.cost_grad(z))    #error at output(depends on the cost_function used)
		
		self.theta_grad = [np.zeros(shape = t.shape) for t in self.theta]  #initializing the gradients
		
		for i in range(n_hlayers):
			a_grad = self.activate_grad(zlist[::-1][i])
			d.append(np.dot(d[i],self.theta[::-1][i])[:,1:]*a_grad)

		d.reverse()

		self.theta_grad[0] = self.theta_grad[0] + np.dot(np.transpose(d[0]),self.bx)	#gradient calculation of input weights(inp-next layer weights)
		for i in range(1,len(self.theta_grad)):
			a = self.activate(zlist[i-1])
			ao = np.ones(shape = (a.shape[0],1))
			a = np.hstack(ao,a)
			self.theta_grad[i] = self.theta_grad[i] + np.dot(np.transpose(d[i]),a)  #gradient calculation of all the other weights


		#Regularisation

		for i in range(len(self.theta_grad)):
			self.theta_grad[i][:,1:]+=(self.lam*self.theta[:,1:])


		self.theta_grad = [(1/self.batch_size)*tg for tg in self.theta_grad]





	def train(self, alpha = 0.01, batch_size = 100,epochs = 10,lam = 0):
		"""
		Traines the network
		Split to batches
		Mini-Batch gradient descent
		alpha is the learning rate
		epochs is the number of iterations of the entire dataset
		lam is the regularization factor by default it is zero
		"""
		self.lam = lam
		self.batch_size = batch_size
		self.init_weights()
		
		j = 1
		i=0

		while j<=epochs:
			while i+self.batch_size<self.m:
				self.bx = self.x[i:i+self.batch_size,:]  #Splitting up batches
				self.by = self.y[i:i+self.batch_size,:]

				print("Cost at current state: ",self.cost_function())

				self.backprop()

				for i in range(len(self.theta)):
					self.theta[i] = self.theta[i] - (alpha*self.theta_grad[i])   #updating weights

				i+=self.batch_size
			j+=1



		print("Trained!!")


	
	def predict(self,inp):
		"""
		Run this function to get predictions
		Return the index of the class predicted
		"""
		z,_ = self.feedforward(inp)
		i = np.argmax(z)			#finiding index of max value

		return i


	def one_hot_encoder(self):
		"""
		One hot encoding of labels
		Given y value is from 1 to n (no of classes)
		"""
		
		ny = np.zeros(shape = (self.m,self.n_classes))
		
		for i in range(m):
			ny[i,self.y[i]-1] = 1    #The n-1 th index will be 1 for each example [where n is the class]

		self.y = ny









