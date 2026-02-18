import re
import time

import random
import itertools
import numpy as np
import pandas as pd
from functools import lru_cache

from scipy.spatial import distance_matrix
from sklearn.model_selection import train_test_split




def F_S(edges, labels, revealed):

    if not hasattr(F_S, "_cache") or F_S._cache_edges is not edges or F_S._cache_labels is not labels:
        F_S._cache_edges = edges
        F_S._cache_labels = labels
        F_S._memo = {}

    key = tuple(sorted(revealed))  

    if key in F_S._memo:
        return F_S._memo[key]

    utility = 0
    revealed_set = set(revealed) 

    for x, neighbors in edges.items():

        px_plus = sum(1 for t in neighbors if t in revealed_set and labels[t] == 1)
        ux_total = sum(1 for t in neighbors if t not in revealed_set)
        ux_plus = sum(1 for t in neighbors if t not in revealed_set and labels[t] == 1)
        
        if px_plus > 0:
            Qx = 1  
        elif ux_total > 0:
            Qx = ux_plus / ux_total
        else:
            Qx = 0

        utility += Qx

    F_S._memo[key] = utility
    
    return utility





  
def random_label_reveal(edges, labels, budget=None, label_value=None, 
                        non_dict=None, initialS=None, seed=42):
    
    if seed is not None:
        random.seed(seed)

    if non_dict:
        targets = list(non_dict)
    elif label_value is None:
        targets = list(labels.keys())
    else:
        targets = [t for t, l in labels.items() if l == label_value]

    if initialS:
        S = set(initialS)
    else:
        S = set()

    if budget is None:
        size = len(targets)
    else:
        size = budget

    choices = list(set(targets) - S)
    if len(choices) <= (size - len(S)):
        reveal_set = choices
    else:
        reveal_set = random.sample(choices, size - len(S))

    S.update(reveal_set)
    utility = F_S(edges, labels, S)

    return S, utility





 
def bruteforce_label_reveal(edges, labels, budget=None, label_value=None, non_dict=None, initialS=None):
    
    if non_dict:
        targets = non_dict
    elif label_value==None:
        targets = set(labels.keys())
    else:
        targets = {t for t, l in labels.items() if l == label_value}
        
    
    if initialS:
        best_S = initialS
    else:
        best_S = set()
    best_F = F_S(edges, labels, best_S)
    
    if budget==None:
        size = len(targets)
    else:
        size = budget
 
    for r in range(0, size + 1):
        for subset in itertools.combinations(targets, r):
            subset_set = set(subset)
            
            f_val = F_S(edges, labels, subset_set)
            
            if f_val > best_F or (f_val == best_F and len(subset_set) < len(best_S)):
                best_S = subset_set
                best_F = f_val

                
    return best_S, best_F






def greedy_label_reveal(edges, labels, budget=None, label_value=None, non_dict=None, initialS=None):
    
    if non_dict:
        targets = non_dict
    elif label_value==None:
        targets = set(labels.keys())
    else:
        targets = {t for t, l in labels.items() if l == label_value}
        
        
    if initialS:
        S = initialS
    else:
        S = set()
    
    current_utility = F_S(edges, labels, S)

    if budget==None:
        size = len(targets)
    else:
        size = budget
      
    while len(S) < size:
        
        best_gain = 0
        best_t = None

        for t in targets - S:
            gain = F_S(edges, labels, S | {t}) - current_utility
            if gain > best_gain:
                best_gain = gain
                best_t = t

        if best_gain <= 0:
            break

        S.add(best_t)
        current_utility += best_gain

    return S, current_utility






def num_neg_pos_nbrs(edges, labels):

    only_neg_adj = 0
    only_pos_adj = 0
    total_agents = 0
    neg_edge_count = 0
    pos_edge_count = 0
    empty_adj = 0

    for a, nbrs in edges.items():

        total_agents += 1

        neg_nbrs = [t for t in nbrs if labels.get(t, 0) == -1]
        pos_nbrs = [t for t in nbrs if labels.get(t, 0) == 1]

        neg_edge_count += len(neg_nbrs)
        pos_edge_count += len(pos_nbrs)
        
        if not nbrs:
            empty_adj += 1

        if len(neg_nbrs) > 0 and len(pos_nbrs) == 0:
            only_neg_adj += 1
            
        if len(pos_nbrs) > 0 and len(neg_nbrs) == 0:
            only_pos_adj += 1

    return {
        "only_neg_adj": only_neg_adj,
        "only_pos_adj": only_pos_adj,
        "total_agents": total_agents,
        "neg_edges": neg_edge_count,
        "pos_edges": pos_edge_count,
        "empty_adj": empty_adj,
        "ratio_only_neg": only_neg_adj / total_agents if total_agents > 0 else 0
    }






def learn_greedy(datagraphs, thresh, savedname, dim, nruns=100):
    
    if thresh:
        gval = "threshold"
        dfval = "r"
    else:
        gval = "k_max"
        dfval = "kmax"
           
    
    summary_resdf = pd.DataFrame()
    N_RUNS = nruns
    budgetlist = [1, 2, 3, 4, 5]

    for lx in range(len(datagraphs)):

        r_edges_randomx  = datagraphs[lx]["edges"]
        r_labels_randomx = datagraphs[lx]["labels"]
        agentsx = list(r_edges_randomx.keys())

        results_by_K = {Ki: [] for Ki in budgetlist}

        for run in range(N_RUNS):

            np.random.seed(run)
            random.seed(run)

            train_agentsx, test_agentsx = train_test_split(agentsx, test_size=0.3, random_state=run)

            train_r_edges_randomx = {a: r_edges_randomx[a] for a in train_agentsx}
            test_r_edges_randomx  = {a: r_edges_randomx[a] for a in test_agentsx}

            tr_statsx = num_neg_pos_nbrs(train_r_edges_randomx, r_labels_randomx)
            ts_statsx  = num_neg_pos_nbrs(test_r_edges_randomx,  r_labels_randomx)

            for Kx in budgetlist:

                startBF = time.time()
                S, tr_sw = greedy_label_reveal(train_r_edges_randomx, r_labels_randomx, budget=Kx)
                endBF   = time.time()

                ts_sw = F_S(test_r_edges_randomx, r_labels_randomx, S)

                results_by_K[Kx].append({
                    "tr_utility": tr_sw,
                    "ts_utility": ts_sw,
                    "tr_size": tr_statsx["total_agents"],
                    "ts_size": ts_statsx["total_agents"],
                    "tr_only-Ns": tr_statsx["only_neg_adj"],
                    "ts_only-Ns": ts_statsx["only_neg_adj"],
                    "tr_empty_adj": tr_statsx["empty_adj"],
                    "ts_empty_adj": ts_statsx["empty_adj"],

                    ######### exclude: no neighbors and exclusively neg neighbors
                    "tr_perf1": round((tr_sw / max(1, tr_statsx["total_agents"] - 
                                            (tr_statsx["empty_adj"]+tr_statsx["only_neg_adj"])))* 100, 3),
                    "ts_perf1": round((ts_sw / max(1, ts_statsx["total_agents"] - 
                                            (ts_statsx["empty_adj"]+ts_statsx["only_neg_adj"])))* 100, 3),
                    ######### exclude: none
                    "tr_perf2": round((tr_sw / max(1, tr_statsx["total_agents"]))*100, 3),
                    "ts_perf2": round((ts_sw / max(1, ts_statsx["total_agents"]))*100, 3),

                    ######### exclude: no neighbors and exclusively neg/pos neighbors
                    "tr_perf3": round(((tr_sw - tr_statsx["only_pos_adj"]) / max(1, tr_statsx["total_agents"] - 
                                (tr_statsx["empty_adj"]+tr_statsx["only_neg_adj"]+tr_statsx["only_pos_adj"])))*100, 3),
                    "ts_perf3": round(((ts_sw - ts_statsx["only_pos_adj"]) / max(1, ts_statsx["total_agents"] - 
                                (ts_statsx["empty_adj"]+ts_statsx["only_neg_adj"]+ts_statsx["only_pos_adj"])))*100, 3),


                    "greedyTime": endBF - startBF
                })


        for Kz in budgetlist:

            avg = pd.DataFrame(results_by_K[Kz]).mean().to_dict()

            data_dictx = {
                "K": Kz,
                "dataset": savedname + f" ({dim})",
                "graphid": lx,
                dfval: datagraphs[lx][gval],
                "n": datagraphs[lx]["n"],
                "m": datagraphs[lx]["m"]
            }
            data_dictx.update(avg)

            summary_resdf = pd.concat(
                [summary_resdf, pd.DataFrame([data_dictx])],
                ignore_index=True
            )
     
    return summary_resdf






def radius_greedy(agents, targets, R):

    D = distance_matrix(targets, agents)  
    m, n = D.shape
    radii   = np.zeros(m)
    covered = np.zeros(n, dtype=bool)
    
    remaining = R

    sorted_dist = np.sort(D, axis=1)
    sorted_idx  = np.argsort(D, axis=1)
    
    ptr = np.zeros(m, dtype=int)

    while remaining > 0:

        next_gain = np.full(m, np.inf)
        for i in range(m):
            while ptr[i] < n and covered[sorted_idx[i, ptr[i]]]:
                ptr[i] += 1
            if ptr[i] < n:
                next_gain[i] = sorted_dist[i, ptr[i]] - radii[i]

        t = np.argmin(next_gain)
        cost = next_gain[t]
        if not np.isfinite(cost) or cost > remaining:
            break

        radii[t] += cost
        remaining -= cost
        a = sorted_idx[t, ptr[t]]
        covered[a] = True
        ptr[t] += 1

    return radii, covered






def compute_Qx(edgez, labelz, revealed_setz):
    
    Qx_dictz = {}
    
    for xz, neighborz in edgez.items():
        px_plusz = sum(1 for tz in neighborz if tz in revealed_setz and labelz[tz] == 1)
        ux_totalz = sum(1 for tz in neighborz if tz not in revealed_setz)
        ux_plusz = sum(1 for tz in neighborz if tz not in revealed_setz and labelz[tz] == 1)

        if px_plusz > 0:
            Qxz = 1
        elif ux_totalz > 0:
            Qxz = ux_plusz / ux_totalz
        else:
            Qxz = 0
            
        Qx_dictz[xz] = Qxz
        
    return Qx_dictz






def greedy_boost_label_reveal(edgesx, labelsx, revealBgt, boostBgt, label_value=None, 
                              non_dict=None, initialS=None):
    S, greedy_utility = greedy_label_reveal(edges=edgesx, 
                                            labels=labelsx, 
                                            budget=revealBgt,
                                            label_value=label_value,
                                            non_dict=non_dict,
                                            initialS=initialS)


    revealed_set = S
        
    Qx_dict = compute_Qx(edgez=edgesx, labelz=labelsx, revealed_setz=revealed_set)
        
    numhighrisk = sum(1 for p in Qx_dict.values() if p < 1)

    if any(l == 1 for l in labelsx.values()):    
        if numhighrisk <= boostBgt:
            updatedboostBgt = numhighrisk
        else:
            updatedboostBgt = boostBgt
    else:
        updatedboostBgt = 0
        
    highrisk_agents = sorted(Qx_dict, key=Qx_dict.get)[:updatedboostBgt]

    boosted_gain = sum(1 - Qx_dict[a] for a in highrisk_agents)
    boosted_utility = greedy_utility + boosted_gain

    return {
        "revealed_nodes": S,
        "revealed_nodes_labels": {t: labelsx[t] for t in S},
        "utility_before_boost": greedy_utility,
        "utility_after_boost": boosted_utility,
        "boosted_agents": highrisk_agents,
        "usableBoostBudget": updatedboostBgt
    }






def boost_greedy_label_reveal(edgesx, labelsx, revealBgt, boostBgt, label_value=None, 
                              non_dict=None, initialS=None):
    
    if initialS:
        revealed_set = set(initialS)
    else:
        revealed_set = set()

    Qx_dict = compute_Qx(edgez=edgesx, labelz=labelsx, revealed_setz=revealed_set)
        
    numhighrisk = sum(1 for p in Qx_dict.values() if p < 1)
    
    if any(l == 1 for l in labelsx.values()):
        if numhighrisk <=  boostBgt:
            updatedboostBgt = numhighrisk
        else:
            updatedboostBgt = boostBgt
    else:
        updatedboostBgt = 0
        
    highrisk_agents = sorted(Qx_dict, key=Qx_dict.get)[:updatedboostBgt]
    base_utility_before = sum(Qx_dict.values())
    reduced_edges = {x: nbrs for x, nbrs in edgesx.items() if x not in highrisk_agents}

    final_S, final_utility = greedy_label_reveal(edges=reduced_edges, 
                                           labels=labelsx,
                                           budget=revealBgt, 
                                           label_value=label_value,
                                           non_dict=non_dict, 
                                           initialS=initialS)

    return {
        "removed_agents": highrisk_agents,
        "usableBoostBudget": updatedboostBgt,
        "utility_before_removal": base_utility_before,
        "revealed_nodes": final_S,
        "revealed_nodes_labels": {t: labelsx[t] for t in final_S},
        "final_utility-B": final_utility,
        "final_utility+B": final_utility+updatedboostBgt
    }






def groupspecific_greedy_label_reveal(edges, labels, budget=None, label_value=None, 
                                      non_dict=None, initialS=None, favorgroup=None,
                                      gZeroEdges=None, gOneEdges=None):

    
    if non_dict:
        targets = non_dict
    elif label_value==None:
        targets = set(labels.keys())
    else:
        targets = {t for t, l in labels.items() if l == label_value}
        
        
    if initialS:
        S = initialS
    else:
        S = set()
    
    current_utility = F_S(edges, labels, S)
    
    
    if budget==None:
        size = len(targets)
    else:
        size = budget
        

    while len(S) < size:

        best_gain = 0
        best_t = None
        best_gain_group0 = 0
        best_gain_group1 = 0
        
        for t in targets - S:
            tiebreak = False
            newS = S | {t}
            
            gain = F_S(edges, labels, newS) - current_utility
            gain_group0 = F_S(gZeroEdges, labels, newS) - current_utility
            gain_group1 = F_S(gOneEdges, labels, newS) - current_utility

            if favorgroup == 0:
                tiebreak = gain_group0 > best_gain_group0
            else:
                tiebreak = gain_group1 > best_gain_group1

            if (gain > best_gain or (gain == best_gain and tiebreak == True)):
                best_gain = gain
                best_t = t
                best_gain_group0 = gain_group0
                best_gain_group1 = gain_group1

        if best_gain <= 0:
            break

        S.add(best_t)
        current_utility += best_gain

    return S, current_utility


