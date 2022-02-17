import math

import cvxpy
from decimal import Decimal
import numpy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx # graphs

"""graph laplacian and ground-truth positions (centered) """
n = 200
p = 0.125
seed = 896803
np.random.seed(seed=seed)
n_fixed = 50
n_free = n - n_fixed
G = nx.random_geometric_graph(n, p, seed=seed)
# position is stored as node attribute data for random_geometric_graph
pos = nx.get_node_attributes(G, "pos")

dmin = 1
ncenter = 0
for i in pos:
    x, y = pos[i]
    d = (x - 0.5) ** 2 + (y - 0.5) ** 2
    if d < dmin:
        ncenter = i
        dmin = d

p = dict(nx.single_source_shortest_path_length(G, ncenter))

pos2npy = np.array([x for x in pos.values()])
x = pos2npy[:,0] - 0.5
y = pos2npy[:,1] - 0.5
L = nx.laplacian_matrix(G)
print(L.shape)

random_node_indices = np.random.choice(np.arange(n), n_fixed, replace=False)
fixed_mask = np.zeros(n, np.bool)
fixed_mask[random_node_indices] = 1
free_mask = ~fixed_mask

"""replot, color by fixed nodes """
fnc = np.zeros(n)
fnc[fixed_mask] = 1
plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G, pos, alpha=0.4)
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=list(p.keys()),
    node_size=80,
    node_color=list(fnc),
    #cmap=plt.cm.Reds_r,
)

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.axis("off")
plt.show()


c = 10
x_var = cvxpy.Variable(n_free)
y_var = cvxpy.Variable(n_free)
# L_free = L[:n_free, :n_free]
L_free = L[np.ix_(free_mask, free_mask)]
# b = x[fixed_mask]@L[n_free:, :n_free] + x[fixed_mask]@L[:n_free, n_free:].T
# d = y[fixed_mask]@L[n_free:, :n_free] + y[fixed_mask]@L[:n_free, n_free:].T
b = 2*L[np.ix_(free_mask, fixed_mask)]@x[fixed_mask]
d = 2*L[np.ix_(free_mask, fixed_mask)]@y[fixed_mask]
objective = cvxpy.Minimize(cvxpy.quad_form(x_var, L_free) + cvxpy.quad_form(y_var, L_free) + b@x_var + d@y_var)
constraint = [cvxpy.sum_squares(x_var) <= 10, cvxpy.sum_squares(y_var) <= 10]
prob = cvxpy.Problem(objective, constraints=constraint)
result = prob.solve(verbose=True)
print(result)

x_mapped = np.zeros(n)
# x_mapped[fixed_mask+free_mask] = x_var.value
# print(x_mapped)
x_mapped[fixed_mask] = x[fixed_mask]
x_mapped[free_mask] = x_var.value
y_mapped = np.zeros(n)
# y_mapped[fixed_mask+free_mask] = y_var.value
y_mapped[fixed_mask] = y[fixed_mask]
y_mapped[free_mask] = y_var.value

newpos = {}
for i in range(n):
  newpos[i] = [x_mapped[i]+0.5, y_mapped[i]+0.5]

plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G, newpos, alpha=0.4)
nx.draw_networkx_nodes(
    G,
    newpos,
    nodelist=list(p.keys()),
    node_size=80,
    node_color=list(p.values()),
    cmap=plt.cm.Reds_r,
)

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.axis("off")
plt.show()