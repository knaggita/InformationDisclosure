import os
import string
import itertools
import numpy as np
import pandas as pd 
from collections import defaultdict

from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder, StandardScaler






def graphgen_thresh_topk(XLHS, XRHS, yRHS, sLHS=None, sRHS=None, 
                         k_min=1, k_max=10, 
                         threshold=1, topk=True, thresh=False):
    
    distances = cdist(XLHS, XRHS)
    edges = {}

    for i in range(len(XLHS)):
        
        if topk:
            sorted_idx = np.argsort(distances[i])
            neighbors = sorted_idx[:k_max]  
            if len(neighbors) < k_min:
                neighbors = sorted_idx[:k_min] 
            edges[i] = neighbors.tolist()
            
        if thresh:
            neighbors = np.where(distances[i] <= threshold)[0]
            edges[i] = neighbors.tolist()
    

    active_indices = set(j for neigh in edges.values() for j in neigh)
    labels = {jt: yRHS[jt] for jt in active_indices}
    target_protectAttrs = {t: sRHS[t] for t in active_indices}
    
    agent_indices = set(edges.keys())
    agent_protectAttrs = {a: sLHS[a] for a in agent_indices}

    graph = {
        'n': len(edges),
        'm': len(labels),
        'edges': edges,
        'labels': labels,
        'target_protectatts': target_protectAttrs,
        'agent_protectatts': agent_protectAttrs
    }
    

    return edges, labels, graph






def getconnectivity_info(edgesx, labelsx):
    
    L = list(edgesx.keys())
    R = set()
    adjL = {}
    
    for u, nbrs in edgesx.items():
        adj = set(nbrs)
        adjL[u] = adj
        R |= adj

    m = len(L)
    n = len(R)
    M = sum(len(adj) for adj in adjL.values())

    avg_left_deg = M / m if m > 0 else 0
    avg_right_deg = M / n if n > 0 else 0

    pos_counts, neg_counts = 0, 0
    only_pos, only_neg, empty_adj = 0, 0, 0

    mixed_nodes = []

    for u, nbrs in adjL.items():
        pos_nbrs = [v for v in nbrs if labelsx.get(v, 0) == 1]
        neg_nbrs = [v for v in nbrs if labelsx.get(v, 0) == -1]

        pos_counts += len(pos_nbrs)
        neg_counts += len(neg_nbrs)

        if not nbrs:
            empty_adj += 1
        elif len(pos_nbrs) > 0 and len(neg_nbrs) == 0:
            only_pos += 1
        elif len(neg_nbrs) > 0 and len(pos_nbrs) == 0:
            only_neg += 1
        else:
            mixed_nodes.append(u)

    avg_left_pos_deg = pos_counts / m if m > 0 else 0
    avg_left_neg_deg = neg_counts / m if m > 0 else 0


    intersections, unions, count = 0, 0, 0
    pos_intersections, neg_intersections = 0, 0

    for u, v in itertools.combinations(L, 2):
        A, B = adjL[u], adjL[v]
        if A or B:
            intersections += len(A & B)
            unions += len(A | B)
            count += 1

            A_pos = {x for x in A if labelsx.get(x, 0) == 1}
            B_pos = {x for x in B if labelsx.get(x, 0) == 1}
            pos_intersections += len(A_pos & B_pos)

            A_neg = {x for x in A if labelsx.get(x, 0) == -1}
            B_neg = {x for x in B if labelsx.get(x, 0) == -1}
            neg_intersections += len(A_neg & B_neg)

    avg_overlap = intersections / count if count > 0 else 0
    avg_pos_overlap = pos_intersections / count if count > 0 else 0
    avg_neg_overlap = neg_intersections / count if count > 0 else 0


    if mixed_nodes:
        cmn_pos_neighs = None
        for u in mixed_nodes:
            pos_neighbors = {v for v in adjL[u] if labelsx.get(v, 0) == 1}
            if cmn_pos_neighs is None:
                cmn_pos_neighs = pos_neighbors
            else:
                cmn_pos_neighs &= pos_neighbors
        numuneighs = len(cmn_pos_neighs) if cmn_pos_neighs else 0
    else:
        numuneighs = 0

    return (
        avg_left_deg,
        avg_left_pos_deg,
        avg_left_neg_deg,
        avg_right_deg,
        avg_overlap,
        avg_pos_overlap,
        avg_neg_overlap,
        only_pos,
        only_neg,
        empty_adj,
        numuneighs
    )


