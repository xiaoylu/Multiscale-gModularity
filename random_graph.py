# -*- coding: utf-8 -*-
"""

This module test the results on random networks

Example:
    Execute the code to test on LFR network::

        $ python random_graph.py

Research article is available at:
   http://...

"""

import networkx as nx
import os
import time
import random as rd
from run import multiscale

def generator(N, M):
    edges = [(rd.randint(1, N), rd.randint(1, N)) for _ in range(M)]
    edges = list(set([((i, j), 1) for i, j in edges if i != j]))
    nodes = sorted(list(set([x for (i,j),_ in edges for x in [i, j] ])))

    return nodes, edges

def test():
    nodes, edges = generator(1000, 20000)
    commsMS = multiscale(nodes, edges, 0.8, verbose = True)
    
test()