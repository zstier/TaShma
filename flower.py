# specify a flower graph and compute data about its perron eigenvector
# requires networkx but any >= 3.1 should suffice
# run this in the command line as:
# python3 flower.py --core 20 --starts 1 10 --ends 5 15 --lengths 8 11
# where these are (respectively) the length of the core, the starting vertices of where each petal first meets the core, the end, and the number of off-core vertices in each petal

import networkx as nx
import numpy as np
from numpy.linalg import norm
from numpy import sqrt, log, abs
import random
from random import choice
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse.linalg import eigs
# import sys
import argparse

CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--core",  # name on the CLI - drop the `--` for positional/required parameters
  nargs=1,
  type=int,
  default=0,  # default if nothing is provided
)
CLI.add_argument(
  "--starts",
  nargs="*",
  type=int,  # any type/callable can be used here
  default=[],
)
CLI.add_argument(
  "--ends",
  nargs="*",
  type=int,  # any type/callable can be used here
  default=[],
)
CLI.add_argument(
  "--lengths",
  nargs="*",
  type=int,  # any type/callable can be used here
  default=[],
)

args = CLI.parse_args()

[core] = args.core
starts = args.starts # one endpoint of each petal on the core
ends = args.ends # the other endpoint of each petal on the core (can be the same)
lengths = args.lengths # number of off-core vertices in the petals. should be >= 2
print(core, starts, ends, lengths)
petal_counts = len(starts)
assert(len(ends) == petal_counts)
assert(len(lengths) == petal_counts)


def flower(core, starts, ends, lengths):
	G = nx.Graph() # create graph
	# create core
	for i in range(core-1):
		G.add_edge(i,i+1)
	G.add_edge(0,core-1)
	core_verts = list(range(core)) + [0] # a cyclic list of the vertices in the core
	
	smallest_index = core-1 # the most recent largest index created
	
	# create petals
	petal_vert_lists = [] # a list of all the cyclic lists of vertices in each petal
	for s,e,l in zip(starts, ends, lengths): # for each petal
		for j in range(smallest_index+1, smallest_index+l): # create l new vertices
			G.add_edge(j,j+1)
		G.add_edge(s, smallest_index+1) # connect s to the first new vertex
		G.add_edge(smallest_index+l,e) # connect last new vertex to e
		petal_verts = list(range(smallest_index+1,smallest_index+l+1)) # start to create cyclic list of vertices in this petal (these are the new ones)
		for j in range(e, s-1, -1): # add the ones along the core
			petal_verts.append(j%core)
		petal_vert_lists.append([s] + petal_verts) # make cyclic, add to list
		smallest_index += l # update
	return G, core_verts, petal_vert_lists

G, core_verts, petal_vert_lists = flower(core, starts, ends, lengths)

"""
nx.draw_planar(G)
plt.show()
plt.close()
"""
def nb(G): # create the nonbacktracking matrix (unnormalized) of G
	H = nx.DiGraph()
	for u in G.nodes:
		nbhd_u = G.neighbors(u)
		for v in nbhd_u:
			nbhd_v = G.neighbors(v)
			for w in nbhd_v:
				if w != u:
					H.add_edge((u,v),(v,w))
	B = nx.adjacency_matrix(H)
	return H.nodes, B.astype('f')

def perron_position(v):
	return np.argmax(abs(v))

nodes, B = nb(G)

nodes_dict = {}
for i, p in enumerate(nodes):
	nodes_dict[p] = i

# print(nodes)
# print(nodes_dict)

# np.random.seed(0)
# val, vec = eigs(B, k=1, which='LM')
# val, vec = eigs(B, k=1, which='LM')
vals, vecs = np.linalg.eig(B.toarray())
max_idx = np.argmax(np.abs(vals))
val, vec = vals[max_idx], vecs[:, max_idx]
# print(val, vec)

print(val, vec)
for x in vec:
	if x < 0:
		vec = -vec
		break

# print(core_verts)
# print(petal_vert_lists)


# core
fig, axs = plt.subplots(1+petal_counts,2)
fig.suptitle('nb Perron eigenvector entries per cycle')
## forwards
axs[0, 0].set_title('Core, ccw')
axs[0, 0].plot([vec[nodes_dict[(core_verts[i],core_verts[i+1])]] for i in range(len(core_verts)-1)])
## backwards
axs[0, 1].set_title('Core, cw')
axs[0, 1].plot([vec[nodes_dict[(core_verts[i],core_verts[i-1])]] for i in range(len(core_verts)-1,0,-1)])
# petals
for i, petal in enumerate(petal_vert_lists):
	## forwards
	axs[i+1, 0].set_title('Petal ' + str(i) + ', ccw')
	axs[i+1, 0].plot([vec[nodes_dict[(petal[i],petal[i+1])]] for i in range(len(petal)-1)])
	## backwards
	axs[i+1, 1].set_title('Petal ' + str(i) + ', cw')
	axs[i+1, 1].plot([vec[nodes_dict[(petal[i],petal[i-1])]] for i in range(len(petal)-1,0,-1)])

plt.show()
plt.close()
