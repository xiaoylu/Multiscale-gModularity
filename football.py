import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pickle
import time
import numpy as np
from sklearn import metrics

from pylouvain import PyLouvain
from run import multiscale, bayes_model_selection

x, y, z = [], [], []
for gamma in np.linspace(0.2, 0.9, num=20):
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

    x.append(gamma)
    y.append(len(partition))
    z.append(metrics.adjusted_mutual_info_score(a, b))

    print("#comm=", len(partition), "NMI=", metrics.adjusted_mutual_info_score(a, b))
    print()

plt.plot(x, y, 'r-*', markersize=10)
ax1 = plt.gca()
ax1.tick_params(axis='x', labelsize=18)
ax1.tick_params(axis='y', labelcolor='r', labelsize=15)
ax2 = ax1.twinx()
ax2.plot(x, z, 'm-^', markersize=10)
ax2.tick_params(axis='y', labelcolor='m', labelsize=15)
plt.tight_layout()

plt.savefig("fig/football_multi_scale.png")