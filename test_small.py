import matplotlib
from matplotlib import pyplot as plt
from sklearn import metrics
from pylouvain import PyLouvain
from sklearn import metrics
import numpy as np

nodes, edges = PyLouvain.from_file("data/karate.txt")
gamma0 = 0.78

#nodes, edges = PyLouvain.from_gml_file("data/lesmis.gml")
#nodes, edges = PyLouvain.from_gml_file("data/polbooks.gml")

def cmap(nodes, partition):
    m = {n:i for i, _ in enumerate(partition) for n in _}
    return [m[i] for i in nodes]

def test_small_networks(nodes, edges, gamma0):
    pyl = PyLouvain(nodes, edges)
    partition0, q0 = pyl.apply_method(gamma0)
    c0 = cmap(nodes, partition0)
    NMI = []

    gamma_list = np.linspace(0.2, 3.5, num=200)
    for gamma in gamma_list:
        partition, q = PyLouvain(nodes, edges).apply_method(gamma)
        c = cmap(nodes, partition)

        NMI.append( metrics.normalized_mutual_info_score(c0, c) )
    
    plt.plot(gamma_list, NMI, 'b-*', markersize=10)
    plt.show()

test_small_networks(nodes, edges, gamma0)