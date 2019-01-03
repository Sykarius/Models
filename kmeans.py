import numpy as numpy
import matplotlib.pyplot as pyplot

class Kmeans():

	def __init__(self, x):
		self.x = x
		self.c = np.zeros(x.shape[0],1)


	def cassign(self):
		"""
		Assigning each point to a cluster
		"""
		for i,p in enumerate(self.x):
			r = (p - self.mu)**2
			self.c[i] = np.argmin(np.sum(r, axis =1))

	def centroid_update(self):
		"""
		Updating centroid of each cluster
		"""
		for i,_ in enumerate(self.mu):
			b_array = c==i
			self.mu[i] = np.mean(self.x[b_array], axis = 0)

	def centroid_init(self):
		"""
		Initializing cluster centroid to k random datapoints
		"""
		self.mu = np.zeros(self.k,x.shape[1])
		idx = np.random.randint(self.x.shape[0],size(self.k,1))
		self.mu  = self.x[idx,:]

	def clustering(self,k,stop = 0.01):
		"""
		running the kmeans clustering for k clusters
		"""
		self.k = k
		self.centroid_init()
		while True:
			self.cassign()
			old_mu = self.mu
			self.centroid_init()
			cond = ((self.mu - old_mu) > stop)
			if not cond.any():
				print("Clustered")
				break
		self.cassign()
		return self.c

	def cost_function(self):
		"""
		Finds cost at current state
		"""
		self.cost = (1/m)*np.sum(self.x - self.mu[self.c])
		return self.cost

	def k_vs_j(self):
		"""
		Plotting to find optimal value of k (elbow)
		"""
		fig,ax = plt.subplots()
		ax.set(xlabel = 'K',ylabel = 'J',title = 'Cost vs K', xlim = [0,self.x.shape[0]])
		k_vals = list(range(1,self.x.shape[0]+1))
		cost_vals = []
		for k in k_vals:
			self.clustering(k)
			cost_vals.append(self.cost_function())

		ax.plot(k_vals,cost_vals,linewidth = 3, color = 'k')
		plt.show()












