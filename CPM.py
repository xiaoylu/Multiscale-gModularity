import igraph as ig
import louvain

G = ig.Graph.Famous('Zachary')
partition = louvain.find_partition(G, louvain.ModularityVertexPartition)
print(partition)
#ig.plot(partition)

partition = louvain.find_partition(G, louvain.CPMVertexPartition, resolution_parameter = 0.05)
print(partition)