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
  mybins = [0] + list(np.logspace(np.log10(5), np.log10(10000), 15)) 
  print(mybins)

  for i, label in enumerate(sorted(sizes_distri.keys())):
    a = sizes_distri[label]
    print(min(a), max(a), sum(a))
    hist, bin_edges = np.histogram(a, bins = mybins)
    x, y = zip(*[(a, b) for a, b in zip(bin_edges[1:], hist) if b > 0])
    print(label, hist)
    
    y_norm = y / sum(y)

    plt.plot(x, y_norm, marker[i], label=label, linewidth = 2, markersize = 10) 

  ax = plt.gca()
  ax.set_xscale('log')
  #ax.set_yscale('log')

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
      print("pickle file", name_pickle, "is missing. Recompute.")

      start = time.time()
      partition, q = pyl.apply_method()
      print("Modularity Time", time.time() - start)

      start = time.time()
      partition2 = multiscale(pyl.nodes, pyl.edges, 0.5)
      print("Multiscale Time", time.time() - start)

      results = {"LV": partition, "MS": partition2}
      sizes_distri = {"Modularity": [len(p) for p in partition], "MultiScale": [len(p) for p in partition2]}

      pickle.dump(results, open(name_pickle, 'wb'))
      print("Pickle save", name_pickle)
    else:
      print("pickle file", name_pickle, "is found.")
      
      results = pickle.load(open(name_pickle, "rb"))
      sizes_distri = {"Modularity": [len(p) for p in results["LV"]], "MultiScale": [len(p) for p in results["MS"]]}

    if gnc:
      gnc_fp = open(gnc, "r+")
      gnc_map = {}
      sizes_distri["Ground Truth"] = []
      for i, line in enumerate(gnc_fp):
        x = line.split()
        sizes_distri["Ground Truth"].append(len(x)) 
        for j in x:
          gnc_map[int(j)] = i
      
      gnc_list = [gnc_map[k] for k in pyl.nodes]
      
      lv_map = {v:i for i, c in enumerate(partition) for v in c}
      lv_list = [lv_map[k] for k in pyl.nodes]

      ms_map = {v:i for i, c in enumerate(partition2) for v in c}
      ms_list = [ms_map[k] for k in pyl.nodes]

      print("Louvain NMI=", normalized_mutual_info_score(lv_list, gnc_list) )
      print("Multi-scale NMI=", normalized_mutual_info_score(ms_list, gnc_list) )

    hist(sizes_distri, graphname)

#test("hep-th-citations")
#test("com-amazon.ungraph", "data/com-amazon.all.dedup.cmty.txt")
#test("com-dblp.ungraph", "data/com-dblp.all.cmty.txt")

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
    for gamma in np.linspace(0.5, 8.5, num = 35): 
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
