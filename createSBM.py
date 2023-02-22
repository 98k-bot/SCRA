import networkx as nx
import numpy as np
from scipy import sparse, io


q = 0.3
p = 0.8
n = 10
sizes = [n, n, n]
probs = [[p, q, q], [q, p, q], [q, q, p]]

G = nx.stochastic_block_model( sizes, probs )
S = nx.to_scipy_sparse_matrix( G )

labels = np.zeros((1000,2))
labels[:500,0] = 1
labels[500:,1] = 1

io.savemat('sbm1.mat', {'network':S, 'group':labels})       
