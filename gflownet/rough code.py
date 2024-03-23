
n=10
import numpy as np
import random
matrix=np.zeros((n,n))

for i in range(n):
    for j in range(i):
        weight=random.sample([-1,0,1],k=1)[0]
        # weight=random.sample([0,1],k=1)[0]
        # print(weight)
        matrix[i,j]=weight
        matrix[j,i]=weight


# print(matrix)
import networkx as nx
# from dgl import DGLGraph
g=nx.from_numpy_array(matrix)
g=g.to_directed()

import dgl
# # _g=DGLGraph.from_networkx( g, node_attrs=None, edge_attrs=['weight'])
_g = dgl.from_networkx(g, edge_attrs=['weight'])

print(_g.edata)
# # Print edge attributes
# print("Edge Information:")
# for u, v, attr in g.edges(data=True):
#     print(f"Edge ({u}, {v}): Weight = {attr['weight']}")


