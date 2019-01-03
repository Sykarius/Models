import numpy as np
import matplotlib.pyplot as plt
import math

class Anamoly():

	def __init__(self,x):
		self.x = x
		self.sig = np.zeros(self.x.shape[1],1)
		self.mu = np.zeros(self.x.shape[1],1)
		self.epsilon = 0.5

	def fit_model(self):
		"""
		Finding average and standard deviation so as to fit it to a gaussian model
		"""
		self.mu = np.mean(self.x, axis = 0)
		self.sig = np.std(self.x, axis = 0)

	def plot_data(self,feature_idx):
		"""
		Plotting a feature to check if it is gaussian
		"""
		fig,ax = plt.subplots()
		ax.set(title = 'Gaussian Histogram',ylabel = 'Frequency',xlabel = "Chosen feature")
		ax.hist(self.x[feature_idx], edgecolor = 'black', facecolor = 'r')
		plt.show()

	def set_epsilon(self,epsilon):
		"""
		Setting epsilon value which is 0.5 by default
		Epsilon is the boudary between anomalous and non anamalous
		"""
		self.epsilon = epsilon

	
	def anamoly_detec(self,data):
		"""
		Returns true if the given data is anamalous or else false is returned
		"""
		p = (1/(((2*math.pi)**0.5)* self.sig)) * np.prod(np.exp(-((data - self.mu)**2)/( 2 * self.sig**2)))
		if p >= self.epsilon:
			return False
		else:
			return True






