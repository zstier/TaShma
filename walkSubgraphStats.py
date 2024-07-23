# experiement: generate a random Ramanujan graph. generate many nonbacktracking walks. for each walk, compute the subgraph induced by the edges AND vertices visited. tally statistics about the size of the subgraph and the number of vertices of degree > 2.
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

W = 1000 # number of walks to take

for k in range(50, 110, 10):
	print("k =", k)
	mavg = 0
	havg = 0
	mmax = 0
	hmax = 0
	for _ in range(W):
		# print("  starting a new walk")
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
		
		m = H.number_of_nodes()
		high = sum([v[1]>2 for v in H.degree])
		
		mavg += m
		havg += high
		mmax = mmax if mmax > m else m
		hmax = hmax if hmax > high else high
		
		# print("    ", m, " ", high/n)
	print("avgs", mavg/W, havg/(W*n))
	print("maxs", mmax, "   ", hmax/n)
