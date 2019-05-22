#!/usr/python
# -*- coding: utf-8 -*-
"""

This module visualizes the network edges and the community structure
by a heatmap. It serves as a utility for the execution code. 

"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import itertools
import scipy
import collections
import pickle


def heatmap(G, comm): 
    '''
    Output a PDF heatmap
    
    Args:
        G: input networkx graph (unweighted)
        comm: the partition of the graph, in the format of list of lists
       
    '''
    N = int(G.number_of_nodes())
    index = {i:newi for newi, i in enumerate(itertools.chain(*comm))}
    map_comm = {j:i for i, c in enumerate(comm) for j in c}

    plt.clf()

    fig, ax = plt.subplots(1)

    points = collections.defaultdict(lambda : {'x':[], 'y':[]})
    for i, j in G.edges():
       newi, newj = index[i], index[j]
       if map_comm[i] == map_comm[j]:
         points[1 + map_comm[i]]['x'].append(newi)
         points[1 + map_comm[i]]['x'].append(newj)
         points[1 + map_comm[i]]['y'].append(newj)
         points[1 + map_comm[i]]['y'].append(newi)
       else:
         points[0]['x'].append(newi)
         points[0]['x'].append(newj)
         points[0]['y'].append(newj)
         points[0]['y'].append(newi)

    colormap = ['silver', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'violet', 'purple', 'steelblue', 'hotpink', 'darkorchid', 'plum']
    for k in points:
      plt.scatter(points[k]['x'], points[k]['y'], color = colormap[k % len(colormap)], marker='s', edgecolor='none', s = 4)

    # Hide the right and top spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.invert_yaxis()

    plt.xticks(list(range(0, 120, 10)))
    plt.yticks(list(range(0, 120, 10)))

    plt.savefig("heatmap.pdf")

def draw(G, colormap):
    '''
    Visualize a graph using networkx default API
    
    Args:
        G: input networkx graph (unweighted)
        comm: the partition of the graph, in the format of list of lists
       
    '''

    plt.clf()
    pos=nx.spring_layout(G)
    nx.draw_networkx_nodes(G,pos,node_size=[30*G.degree(k) for k,v in pos.items()],node_shape='o',node_color=list(map(colormap.get, G.nodes())))
    labels=nx.draw_networkx_labels(G,\
        pos={k:v+np.array([0.05,0.05]) for k,v in pos.items()},\
        labels={k:"%d"%k for k,v in pos.items()}, font_size=14)
    nx.draw_networkx_edges(G,pos,width=1,edge_color='black')
    plt.axis('off')
    plt.savefig("network.png", bbox_inches="tight")
    print("Save figure to", "network.png") 

def hist(sizes_distri, figname):
  plt.clf()
  marker = ['b*-', 'rx-', 'ko-.']
  #mybins = np.linspace(0, 1000, num = 20) 
  mybins = [0] + list(np.logspace(np.log10(30), np.log10(1000), 8))
  print(mybins)

  for i, label in enumerate(sorted(sizes_distri.keys())):
    a = sizes_distri[label]
    hist, bin_edges = np.histogram(a, bins = mybins)
    print("bin", bin_edges)
    print("count", hist)

    plt.plot(bin_edges[:-1], hist, marker[i], label=label, linewidth = 2, markersize = 10)

  #ax = plt.gca()
  #ax.set_xscale('log')

  plt.tick_params(axis='both', which='major', labelsize=25)
  plt.legend(loc = "upper right", fontsize = 20)
  plt.tight_layout()
  plt.savefig("hist_sizes_%d.png" % figname)
  print("Save figure named \"hist_sizes.png\"")


if __name__ == "__main__":
  #sizes_distri = {"Ground Truth": gnc_sizes, "Modularity": comms0_sizes, "Multiscale": comms1_sizes}
  arg = pickle.load(open( "save5000.p", "rb" ) )
  hist(arg, 5000)

  arg = pickle.load(open( "save7000.p", "rb" ) )
  hist(arg, 7000)

  arg = pickle.load(open( "save9000.p", "rb" ) )
  hist(arg, 9000)

  arg = pickle.load(open( "save11000.p", "rb" ) )
  hist(arg, 11000)
