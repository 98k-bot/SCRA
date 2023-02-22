import numpy as np
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec
import gudhi
import scipy
import networkx as nx
from scipy import sparse, io
import sklearn
from sklearn import cluster
from sklearn.decomposition import PCA

import k_simplex2vec as ks2v
from s2_6_complex_to_laplacians import build_boundaries, build_laplacians, build_D, extract_simplices
from scipy.sparse import dia_matrix
## Build the graph
n = 20  ## n nodes in each block
q = 0.1
p = 0.95
probs = [[p, q, q, q], [q, p, q, q], [q, q, p, q], [q,q,q,p]]  ## set probabilities for the blocks
## first (0,n-1) are in the same block, (n,2n-1) in the second block, etc..

SBM = nx.generators.community.stochastic_block_model([n, n, n,n], probs, seed=7)
S = nx.to_scipy_sparse_matrix(SBM)
labels = np.zeros((4*n,4))
labels[:n,0] = 1
labels[n:2*n,1] = 1
labels[2*n:3*n,2] = 1
labels[3*n:4*n,3] = 1
io.savemat('sbm1.mat', {'network':S, 'group':labels})

## Build a simplicial complex from the graph
st = gudhi.SimplexTree()  # Gudhi simplex tree --> structure to store the simplices
for edge in SBM.edges():
    st.insert(list(edge))
st.expansion(3)  ## Fills in the clique complex up to dimension 3
#a = st.get_skeleton(2)
sk_value_1_list = []
sk_value_2_list = []
sk_value_3_list = []
for sk_value_1 in st.get_skeleton(1):
    sk_value_1_list.append(sk_value_1)
for sk_value_2 in st.get_skeleton(2):
    sk_value_2_list.append(sk_value_2)
for sk_value_3 in st.get_skeleton(3):
    sk_value_3_list.append(sk_value_3)

simps_2= len(sk_value_2_list)-len(sk_value_1_list)
simps_3 = len(sk_value_3_list)-len(sk_value_2_list)

print('There are {} nodes, {} edges, {} triangles, and {} 3-simplices'.format(len(SBM.nodes),len(SBM.edges), simps_2,simps_3))
#compute adjacency matrix
simplices = extract_simplices(st)
boundaries = build_boundaries(simplices)
laplacians = build_laplacians(boundaries)
D = build_D(boundaries)
# Plot the graph
adjacencies = []
for i in range(len(boundaries)):
    adjacency = laplacians[i] -D[i]
    with open('adjacency'+str(i)+'.npy', 'wb') as f1:
        np.save(f1, adjacency)
    adjacencies.append(adjacency)

pos = nx.circular_layout(SBM)
nx.draw_networkx_nodes(SBM,pos,
                       nodelist=[i for i in range(n)],
                       node_color='r',
                       node_size=10,
                   alpha=0.8)
nx.draw_networkx_nodes(SBM,pos,
                       nodelist=[i for i in range(n,2*n)],
                       node_color='b',
                       node_size=10,
                   alpha=0.8)

nx.draw_networkx_nodes(SBM,pos,
                       nodelist=[i for i in range(2*n,3*n)],
                       node_color='g',
                       node_size=10,
                   alpha=0.8)
#edges
nx.draw_networkx_edges(SBM,pos,width=0.3,alpha=0.2)
#plt.show()
## build transition matrix for the edges
p1 = ks2v.assemble(cplx =st, k= 1, scheme = "uniform", laziness =None)
P1 = p1.toarray()

Simplices = list()
for simplex in st.get_filtration():
    if simplex[1]!= np.inf:
        Simplices.append(simplex[0])
    else:
        break


## Perform random walks on the edges
L = 20
N = 40
Walks = ks2v.RandomWalks(walk_length=L, number_walks=N, P=P1,seed = 3)
# to save the walks in a text file
ks2v.save_random_walks(Walks,'RandomWalks_Tri.txt')

## Embed the edges
dim = len(SBM.edges)
Emb = ks2v.Embedding(Walks = Walks, emb_dim = dim , epochs =5 ,filename ='k-simplex2vec_Tri_embedding.model')

Emb = Word2Vec.load('k-simplex2vec_Tri_embedding.model')
Y=Emb.wv.vectors
with open('embedding.npy', 'wb') as f2:
    np.save(f2, Y)
pca=PCA(n_components=dim)
principalComponents = pca.fit_transform(Y)

Xax=principalComponents[:,0]
Yax=principalComponents[:,1]
Zax =  principalComponents[:,2]

fig,ax=plt.subplots(figsize=(10,7))
plt.scatter(Xax,Yax, s = 10)
plt.title('Projection of the Embedding of the edges in two dimensions')
ax.set_xlabel('PC1', fontsize = 15 )
ax.set_ylabel('PC2',fontsize = 15 )
plt.show()