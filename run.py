#!/usr/bin/env python3

import math
import pickle
from collections import defaultdict
import numpy as np
import scipy
from sklearn import metrics
from scipy.stats import entropy
from pylouvain import PyLouvain, in_order
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


def bayes_model_selection(nodes, edges, partition):
    N, E = len(nodes), len(edges)
    comm = {n:i for i, ns in enumerate(partition) for n in ns}
    k_r, k_r_in = defaultdict(int), defaultdict(int)
    for e in edges:
       u, v = e[0][0], e[0][1]
       k_r[comm[u]] += 1
       k_r[comm[v]] += 1
       if comm[u] == comm[v]:
          k_r_in[comm[u]] += 2 
    a = sum(k_r_in.values())
    b = sum([v * v for v in k_r.values()]) / (2. * E)
    c, d = 2. * E - a, 2. * E - b

    ent = scipy.stats.entropy([float(len(c))/N for c in partition])
    return (a * (np.log(a) - np.log(b)) + c * (np.log(c) - np.log(d))) - N * ent

def multiscale(nodes, edges, gamma, depth = 1, verbose = False):
    '''
    Multi-scale community detection.
    Recursively split sub-graph by maximizing generalized modularity.
    Terminate at each level of recursion by bayes model selection.

    Args:
        nodes: a list of nodes
        edges: a list of edges ((src, dst), weight)
        gamma: the resolution parameter for the generalized modularity
    Return:
        a list of lists, each contains the nodes in a community
    '''

    nodes.sort()
    d = {n:i for i, n in enumerate(nodes)}
    rd = {i:n for n, i in d.items()}
    nodes = list(range(len(d))) 
    edges = [((d[e[0][0]], d[e[0][1]]), e[1]) for e in edges]

    pyl = PyLouvain(nodes, edges)

    partition, q = pyl.apply_method(gamma)
    verbose and print("  " * depth, "depth=", depth, "N=", len(nodes), "Comm=", len(partition))

    if len(partition) < 2: return [list(map(rd.get, nodes))]
    odds = bayes_model_selection(pyl.nodes, pyl.edges, partition)
    verbose and print("  " * depth, "odds=", odds)
    if odds < 0 or math.isnan(odds): return [list(map(rd.get, nodes))]

    comm = {n:i for i, ns in enumerate(partition) for n in ns}
    edge_list = [[] for _ in range(len(partition))]
    for e in edges:
        u, v = e[0][0], e[0][1] 
        if comm[u] == comm[v]: 
            edge_list[comm[u]].append(e)

    R = []
    for nodes_, edges_ in zip(partition, edge_list):
        for group in multiscale(nodes_, edges_, gamma, depth + 1):
            R.append([rd[n] for n in group])

    return R

#########################################################################################
def hist(sizes_distri, figname):
  plt.clf()
  marker = ['b*-', 'rx-', 'ko-.']
  mybins = [0] + list(np.logspace(np.log10(30), np.log10(1000), 8)) 
  print(mybins)

  for i, label in enumerate(sorted(sizes_distri.keys())):
    a = sizes_distri[label]
    hist, bin_edges = np.histogram(a, bins = mybins)
    print(bin_edges)
    plt.plot(bin_edges[:-1], hist, marker[i], label=label, linewidth = 2, markersize = 10) 

  ax = plt.gca()
  ax.set_yscale('log')

  plt.tick_params(axis='both', which='major', labelsize=25)
  plt.legend(loc = "upper right", fontsize = 20)
  plt.tight_layout()
  plt.savefig("fig/hist_sizes_%s.png" % figname)
  print("Save figure named \"fig/hist_sizes_%s.png\"" % figname)


##########################################################################################

def test(graphname):
    pyl = PyLouvain.from_file("data/%s.txt" % graphname)
    partition, q = pyl.apply_method()

    partition2 = multiscale(pyl.nodes, pyl.edges, 0.5)

    sizes_distri = {"Modularity": [len(p) for p in partition], "MultiScale": [len(p) for p in partition2]}
    pickle.dump(sizes_distri, open('fig/save_%s_%d.p' % (graphname, len(pyl.nodes)), 'wb'))
    hist(sizes_distri, graphname)

#test("arxiv")
#test("hep-th-citations")

def test_citations():
    pyl = PyLouvain.from_file("data/hep-th-citations")
    partition, q = pyl.apply_method()
    print(partition, q)

def test_karate_club():
    pyl = PyLouvain.from_file("data/karate.txt")
    partition, q = pyl.apply_method(gamma = 0.7)
    odds = bayes_model_selection(pyl, partition)

    print(partition, q, odds)

#test_karate_club()

def test_lesmis():
    pyl = PyLouvain.from_gml_file("data/lesmis.gml")
    partition, q = pyl.apply_method()
    print(partition, q)

def test_polbooks():
    pyl = PyLouvain.from_gml_file("data/polbooks.gml")
    partition, q = pyl.apply_method()
    print(partition, q)

def test_football():

    # load GNC ground truth by txt file (as defined by conference)
    fconf = open("data/football.gnc.txt", "r")
    gnc = {str(i):int(line.strip()) for i, line in enumerate(fconf)}
    order_ = {i:stri for i, stri in enumerate(sorted(gnc.keys()))}

    x, y, z = [], [], []
    for gamma in np.linspace(0.5, 8.0, num = 35): 
        pyl = PyLouvain.from_file("data/football.txt")
        partition, q = pyl.apply_method(gamma)
        odds = bayes_model_selection(pyl.nodes, pyl.edges, partition)

        print(len(partition), odds)
        x.append(gamma)
        y.append(odds)
        z.append(len(partition))

        comm = {n:i for i, ns in enumerate(partition) for n in ns}
        a = [comm[i] for i in pyl.nodes]
        b = [gnc[order_[i]] for i in pyl.nodes]
        print("NMI=", metrics.adjusted_mutual_info_score(a, b))
   
    plt.plot(x, y, 'r-*', markersize=10)
    ax1 = plt.gca()
    ax1.tick_params(axis='y', labelcolor='r', labelsize=15)
    ax2 = ax1.twinx()
    ax2.plot(x, z, 'b-^', markersize=10)
    ax2.tick_params(axis='y', labelcolor='b', labelsize=15)
    plt.tight_layout()

    plt.savefig("fig/football.png")

#test_football()

def test_football2():

    for gamma in np.linspace(0.4, 0.9, num=10):
        print("gamma=", gamma)
        pyl = PyLouvain.from_file("data/football.txt")
        partition = multiscale(pyl.nodes, pyl.edges, gamma)

        # load GNC ground truth by txt file (as defined by conference)
        fconf = open("data/football.gnc.txt", "r")
        gnc = {str(i):int(line.strip()) for i, line in enumerate(fconf)}
        order_ = {i:stri for i, stri in enumerate(sorted(gnc.keys()))}
        comm = {n:i for i, ns in enumerate(partition) for n in ns}
        a = [comm[i] for i in pyl.nodes]
        b = [gnc[order_[i]] for i in pyl.nodes]

        print("NMI=", metrics.adjusted_mutual_info_score(a, b))

test_football2()
