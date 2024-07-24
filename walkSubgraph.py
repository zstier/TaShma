# experiement: generate a random Ramanujan graph. generate many nonbacktracking walks. for each walk, compute the subgraph induced by the edges AND vertices visited. compute its nonbacktracking random walk operator (unnormalized), and use the trace-power method to compute the number of closed walks in this graph.
# N.B.: earlier files in this repo used networkx3.1 due to an issue on my local system which i've now resolved, so this and all subsequent files will use networkx3.3.
import networkx as nx
import numpy as np
from numpy.linalg import norm
from numpy import sqrt, log
import random
from random import choice
import matplotlib.pyplot as plt
import scipy as sp

seed = 1 # starting seed value
random.seed(seed) # starting seed value

def nb(G): # create the nonbacktracking matrix (unnormalized) of G
	H = nx.DiGraph()
	for u in G.nodes:
		nbhd_u = G.neighbors(u)
		for v in nbhd_u:
			nbhd_v = G.neighbors(v)
			for w in nbhd_v:
				if w != u:
					H.add_edge((u,v),(v,w))
	return nx.adjacency_matrix(H).toarray()

d = 6
# Ramanujan_bound = 2*sqrt(d-1)
# print("Ramanujan bound =", Ramanujan_bound)
n = 300

G = nx.random_regular_expander_graph(n, d, epsilon=0, seed=seed)
print("created G")

W = 20 # number of walks to take
h = 200 # hike length

for k in range(50, 110, 10): # walk length
	print("k =", k)
	lhc = []
	lwc = []
	for _ in range(W):
		# print("starting a new walk")
		v = choice(list(range(n))) # starting vertex, eventually current vertex
		# print("start at", v)
		H = nx.Graph()
		u = None # previous vertex
		for __ in range(k):
			nbhd = list(G.neighbors(v))
			# print("current nbhd", nbhd)
			w = choice(nbhd) # next vertex
			while (w == u): # ensure nonbacktracking
				w = choice(nbhd)
			# print("next vertex is", w)
			H.add_edge(v, w) # add this edge to the subgraph
			u = v
			v = w
		B = nb(H)
		
		"""
		### counting hikes
		C = B @ np.matrix.transpose(B) # hike matrix for 1 step
		hikeCounts = []
		for l in range(h):
			# print(l, np.trace(C))
			hikeCounts.append(np.trace(C))
			C = B @ C @ np.matrix.transpose(B) # hike matrix for one additional step
		# print(hikeCounts)
		# logHikeCounts = [ log(hikeCounts[k])/(k+2) for k in range(h) ]
		logHikeCounts = []
		i = 0
		while i < h and hikeCounts[i] >= 0:
			logHikeCounts.append(0 if hikeCounts[i] == 0 else log(hikeCounts[i])/(i+2))
			i += 1
		lhc.append(logHikeCounts)
		"""
		
		### counting nb walks
		A = nx.adjacency_matrix(H)
		m = H.number_of_nodes()
		# print(H.nodes)
		# print(H.degree)
		# print([v[1] for v in H.degree])
		# print(A.todense())
		D = np.diag([v[1] for v in H.degree])
		# nb matrix from 2 steps ago
		N1 = A # nb matrix from 1 step ago
		Nc = N1 @ A - D # current nb matrix
		walkCounts = []
		for l in range(2, 2*h):
			walkCounts.append(np.trace(Nc))
			N1new = Nc
			Ncnew = Nc @ A - N1 @ D + N1 # why does this work?
			Nc = Ncnew
			N1 = N1new
			# print(Nc)
		# """
		# print(walkCounts)
		logWalkCounts = np.array([])
		i = 1
		while walkCounts[i] >= 0 and i < h:
			logWalkCounts = np.append(logWalkCounts, [np.nan if walkCounts[i] == 0 else log(walkCounts[i])/(i+2)])
			i += 1
				
		""" # code i got from chatgpt for interpolating over gaps
		nans, x_ = np.isnan(logWalkCounts), lambda z: z.nonzero()[0]
		logWalkCounts[nans] = np.interp(x_(nans), x_(~nans), logWalkCounts[~nans])
		"""
		
		lwc.append(logWalkCounts)
		
		
	"""
	for hike in lhc:
		plt.plot(hike)
	plt.show()
	plt.close()
	"""
	for walk in lwc:
		plt.plot(walk)
	# plt.show()
	plt.savefig("figs/seed"+str(seed)+"w"+str(k))
	plt.close()
