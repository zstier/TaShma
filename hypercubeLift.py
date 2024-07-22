import networkx as nx
import numpy as np
from numpy.linalg import norm
from numpy import sqrt
import random
from random import sample, shuffle
import matplotlib.pyplot as plt
import scipy as sp

d = 6
Ramanujan_bound = 2*sqrt(d-1)
print("Ramanujan bound =", Ramanujan_bound)
n = 20
seed = 20 # starting seed value
k = int(n/3)
K = 2**k
inc = 10 # increment of epsilon
eps = [x/inc for x in range(inc)]
Graphs = 30 # how many graphs to study
graphs = Graphs
trials = 10 # how many lifts to consider per graph
rams = [0,]*inc # how many eps-Ramanujan graphs are gotten in this way

def is_Ramanujan(G, eps=[0]):
	lambda1 = sorted(nx.adjacency_spectrum(G),reverse=True)[1]
	return [lambda1 <= Ramanujan_bound + delta for delta in eps]

def vertex_index(vert, copy, n, K): # 0<=vert<n, 0<=copy<K
	return n*copy+vert

while graphs > 0:
	G = nx.random_regular_graph(d, n, seed=seed)
	if is_Ramanujan(G):
		graphs -= 1
		print("seed =", seed)
		np.random.seed(seed)
		E = list(nx.to_edgelist(G,nodelist=range(n)))
		# A = nx.to_numpy_array(G,nodelist=range(n))
		# lift = sp.linalg.block_diag([A,]*K)
		for _ in range(trials):
			H = nx.Graph()
			for e in E: # lift the edges in a random way (currently they are "parallel")
				Pe = np.random.permutation(range(K))
				e0 = e[0]
				e1 = e[1]
				for copy in range(K):
					u0 = vertex_index(e0, copy, n, K)
					v0 = vertex_index(e1, copy, n, K)
					v1 = vertex_index(e1, Pe[copy], n, K)
					"""
					lift[u0][v0] = 0
					lift[v0][u0] = 0
					lift[u0][v1] = 1
					lift[v0][u1] = 1
					"""
					H.add_edge(u0, v1)
			"""
			for v in range(n): # add hypercube edges at each vertex
				for copy in range(K):
					u0 = vertex_index(v, copy, n, K)
					for i in range(k):
						v0 = vertex_index(v, copy^(2**i), n, K)
						# lift[u0][v0] = 1
						H.add_edge(u0, v0)
			"""
			new_rams = is_Ramanujan(H, eps)
			for i in range(inc):
				rams[i] += new_rams[i]
	seed += 1

for i in range(inc):
	print("% of", str(eps[i])+"-Ramanujan hypergraph lifts =", 100*rams[i]/(Graphs*trials))
