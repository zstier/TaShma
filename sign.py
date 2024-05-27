# experiment: for each graph, generate some number of eps-balanced signings of the edges, where eps is the balance of the edge-signing induced by a random vertex-signing, and check how many give a signed operator of norm below the Ramanujan bound
# this is written using Networkx 3.1, even though later versions have much better expander functionality (for some reason by Anaconda installation refuses to update past 3.1)
import networkx as nx
import numpy as np
from numpy.linalg import norm
from numpy import sqrt
import random
from random import sample, shuffle
import matplotlib.pyplot as plt


d = 6
Ramanujan_bound = 2*sqrt(d-1)
print("Ramanujan bound =", Ramanujan_bound)
n = 20
graphs = 100 # how many graphs to study
trials = 1000 # how many signings to consider
seed = 0 # starting seed value
percents = []

def is_Ramanujan(G):
	return sorted(nx.adjacency_spectrum(G),reverse=True)[1] <= Ramanujan_bound

while graphs > 0:
	G = nx.random_regular_graph(d, n, seed=seed)
	if is_Ramanujan(G):
		graphs -= 1
		print("seed =", seed)
		random.seed(seed)
		S = sample(range(n), n//2) # choose the indices of g to be -1
		g = [1,]*n # construct g
		for i in S: # construct g
			g[i] = -1
		# fix an ordering of G's edges (for ease of permuting); construct f
		E = list(nx.to_edgelist(G,nodelist=range(n)))
		edge_to_index = {}
		index_to_edge = {}
		f = {}
		for i in range(len(E)):
			e0 = E[i][0]
			e1 = E[i][1]
			edge_to_index[(e0,e1)] = i
			edge_to_index[(e1,e0)] = i
			index_to_edge[i] = (e0,e1)
			f[(e0,e1)] = g[e0]*g[e1]
			f[(e1,e0)] = g[e0]*g[e1]
		count_Ramanujan_signings = 0
		norms = []
		for _ in range(trials):
			edge_count = list(range(n*d//2))
			shuffle(edge_count) # permute edges
					
			A = nx.to_numpy_array(G,nodelist=range(n))
			for e in E: # construct edge-signed matrix of permutation
				e0 = e[0]
				e1 = e[1]
				i = edge_to_index[(e0,e1)]
				j = edge_count[i] # apply the permutation
				(f0,f1) = index_to_edge[j]
				assert(A[e0][e1] == 1)
				A[e0][e1] = f[(f0,f1)]
				A[e1][e0] = f[(f0,f1)]
			
			op = norm(A,2)
			norms.append(op)
			if op <= Ramanujan_bound:
				count_Ramanujan_signings += 1
		print("% of Ramanujan edge-signings =", 100*count_Ramanujan_signings/trials)
		percents.append(100*count_Ramanujan_signings/trials)
		"""
		plt.plot(norms)
		plt.plot([Ramanujan_bound,]*trials)
		plt.show()
		plt.close()
		"""
	seed += 1
	
plt.plot(percents)
plt.show()
plt.close()
