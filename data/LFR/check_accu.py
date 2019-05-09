# chech NMI
from sklearn import metrics
import networkx as nx
import pickle

for i in range(10):
  path = "mu0.3/instance%d" % i 
  labels_pred = pickle.load(open(path+"/graphtool_comm.p", "rb"))
  labels_pred = labels_pred[1:]
  
  G = nx.Graph()
  with open(path+"/network.dat", "rb") as txt:
    for line in txt:
     if len(line) > 1 and line[0]!='#':
       e = line.split()
       G.add_edge(int(e[0]), int(e[1]))
  
  comm = {}
  with open(path+"/community.dat", "rb") as txt:
    for line in txt:
     if len(line) > 1 and line[0]!='#':
       e = line.split()
       comm[int(e[0])] = int(e[1])
    
  labels_true = [e1 for e0, e1 in sorted(comm.items(), key=lambda x:x[0])]
  print len(labels_true)
  print "AMI=", metrics.adjusted_mutual_info_score(labels_true, labels_pred) 
