import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pickle
import time
import numpy as np
from sklearn import metrics

from pylouvain import PyLouvain
from run import multiscale, bayes_model_selection
import os.path

def hist(sizes_distri, figname):
  plt.clf()
  marker = ['b*-', 'rx-', 'ko-.']
  mybins = [0] + list(np.logspace(np.log10(15), np.log10(1000), 8)) 
  print(mybins)

  for i, label in enumerate(sorted(sizes_distri.keys())):
    a = sizes_distri[label]
    hist, bin_edges = np.histogram(a, bins = mybins)
    print(bin_edges)
    x, y = zip(*[(a, b) for a, b in zip(bin_edges[:-1], hist) if b > 0])

    print("bin", bin_edges)
    print("count", hist)

    plt.plot(x, y, marker[i], label=label, linewidth = 2, markersize = 10) 

  ax = plt.gca()
  ax.set_yscale('log')

  plt.tick_params(axis='both', which='major', labelsize=25)
  plt.legend(loc = "upper right", fontsize = 20)
  plt.tight_layout()
  plt.savefig("fig/hist_sizes_%s.png" % figname)
  print("Save figure named \"fig/hist_sizes_%s.png\"" % figname)

##########################################################################################

def test(graphname, gnc = None):
    pyl = PyLouvain.from_file("data/%s.txt" % graphname)

    name_pickle = 'fig/save_%s_%d.p' % (graphname, len(pyl.nodes))
    if not os.path.isfile(name_pickle):

      start = time.time()
      partition, q = pyl.apply_method()
      print("Modularity Time", time.time() - start)

      start = time.time()
      partition2 = multiscale(pyl.nodes, pyl.edges, 0.5)
      print("Multiscale Time", time.time() - start)

      print(name_pickle, "missing")

      sizes_distri = {"Modularity": [len(p) for p in partition], "MultiScale": [len(p) for p in partition2]}

      pickle.dump(sizes_distri, open(name_pickle, 'wb'))
      print("Pickle save", name_pickle)

    sizes_distri = pickle.load(open(name_pickle, "rb"))

    if gnc:
      gnc_fp = open(gnc, "r+")
      sizes_distri["Ground Truth"] = [len(line.split()) for line in gnc_fp] 

    hist(sizes_distri, graphname)

#test("hep-th-citations")
#test("com-amazon.ungraph", "data/com-amazon.all.dedup.cmty.txt")


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

    x, y, z, r = [], [], [], []
    for gamma in np.linspace(0.5, 3.5, num = 35): 
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
        #print("NMI=", metrics.adjusted_mutual_info_score(a, b))

        r.append(metrics.adjusted_mutual_info_score(a, b))
        #r.append(metrics.adjusted_rand_score(a, b))
   

    plt.plot(x, y, 'r-*', markersize=10)
    ax1 = plt.gca()
    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelcolor='r', labelsize=15)
    ax2 = ax1.twinx()
    ax2.plot(x, z, 'm-^', markersize=10)
    ax2.tick_params(axis='y', labelcolor='m', labelsize=15)
    plt.tight_layout()

    plt.savefig("fig/football2.png")


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

#test_football2()
