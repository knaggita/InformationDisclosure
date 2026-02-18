import re
import math
import time

import numpy as np
import pandas as pd
from itertools import cycle

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from matplotlib.transforms import Affine2D

from scripts.algorithms import *





 
def budget_opts(edges, labels, budget):
    opt_res = []
    for b in range(budget + 1):
        startBF = time.time()
        Sopt, Fopt = bruteforce_label_reveal(edges, labels, b)
        endBF = time.time()
        opt_res.append({
                        "b": b,
                        "F(S*)": Fopt,
                        "S*": Sopt,
                         "BFtime": endBF - startBF
                    })
    
    return opt_res





 
def heuristic_greedy_bruteforce(edgesx, labelsx, budgetx, optx):
    
    results_greedy = []
    
    for b in range(budgetx + 1):
        
        posS_greedy, posF_greedy = greedy_label_reveal(edges=edgesx, 
                                                       labels=labelsx, budget=b, label_value=1)
        negS_greedy, negF_greedy = greedy_label_reveal(edges=edgesx, 
                                                       labels=labelsx, budget=budgetx-b,
                                                       label_value=-1)
            
        F_opt_pos = optx[b]['F(S*)']
        F_opt_neg = optx[budgetx-b]['F(S*)']
        
        ratio_f_pos = posF_greedy / F_opt_pos if F_opt_pos > 0 else 0
        ratio_f_neg = negF_greedy / F_opt_neg if F_opt_neg > 0 else 0
        
        combined_S = posS_greedy | negS_greedy
        combined_F = F_S(edgesx, labelsx, combined_S) 
        
        ratio_f_comb = combined_F / optx[budgetx]['F(S*)'] if optx[budgetx]['F(S*)'] > 0 else 0

        results_greedy.append({
            "[k](T+)": b,
            "F(S+)": posF_greedy,
            "F(S+)/F(Sb)": ratio_f_pos,
            "S+": posS_greedy,
            "[K-k](T-)": budgetx - b,
            "F(S-)": negF_greedy,
            "F(S-)/F(Sb)": ratio_f_neg,
            "S-": negS_greedy,
            "S+ U S-": combined_S,
            "F(S+ U S-)": combined_F,
            "F(S+ U S-)/F(Sb)": ratio_f_comb,
            "[K]": b,
            "Sb": optx[b]['S*'],
            "Fb":optx[b]['F(S*)'],
            "BFtime": optx[b]["BFtime"]
        })

    dframe = pd.DataFrame(results_greedy)
    dframe.reset_index(drop=True, inplace=True)
   
    return dframe
    


    
def heuristic_greedy_random(edgesx, labelsx, budgetx):
     
    results_greedy = []
    results_random = []
    results = []
    
    startBF = time.time()
    
    for b in range(budgetx + 1):
        
        posS_greedy, posF_greedy = greedy_label_reveal(edges=edgesx, 
                                                       labels=labelsx, budget=b, label_value=1)
        negS_greedy, negF_greedy = greedy_label_reveal(edges=edgesx, 
                                                       labels=labelsx, budget=budgetx-b,
                                                       label_value=-1)
        combined_S_greedy = posS_greedy | negS_greedy
        combined_F_greedy = F_S(edgesx, labelsx, combined_S_greedy) 
        
        posS_random, posF_random = random_label_reveal(edges=edgesx, 
                                                       labels=labelsx, budget=b, label_value=1)
        negS_random, negF_random = random_label_reveal(edges=edgesx, 
                                                       labels=labelsx, budget=budgetx-b,
                                                       label_value=-1)
        combined_S_random = posS_random | negS_random
        combined_F_random = F_S(edgesx, labelsx, combined_S_random) 

        results_greedy.append({
            "[k](T+)": b,
            "F(S+)": posF_greedy,
            "S+": posS_greedy,
            "[K-k](T-)": budgetx - b,
            "F(S-)": negF_greedy,
            "S-": negS_greedy,
            "S+ U S-": combined_S_greedy,
            "F(S+ U S-)": combined_F_greedy
        })
        
        results_random.append({
            "[k](T+)": b,
            "F(S+)": posF_random,
            "S+": posS_random,
            "[K-k](T-)": budgetx - b,
            "F(S-)": negF_random,
            "S-": negS_random,
            "S+ U S-": combined_S_random,
            "F(S+ U S-)": combined_F_random
        })
        
    endBF = time.time()

    ########
    dframe_greedy = pd.DataFrame(results_greedy)
    dframe_greedy.reset_index(drop=True, inplace=True)
    ###
    max_val_greedy = dframe_greedy['F(S+ U S-)'].max()
    max_rows_greedy = dframe_greedy[dframe_greedy['F(S+ U S-)'] == max_val_greedy]
    min_set_row_greedy = max_rows_greedy.loc[max_rows_greedy['S+ U S-'].apply(len).idxmin()]
    f_union_greedy = min_set_row_greedy['F(S+ U S-)']
    s_union_greedy = min_set_row_greedy['S+ U S-']

    ########
    dframe_random = pd.DataFrame(results_random)
    dframe_random.reset_index(drop=True, inplace=True)
    ###
    max_val_random = dframe_random['F(S+ U S-)'].max()
    max_rows_random = dframe_random[dframe_random['F(S+ U S-)'] == max_val_random]
    min_set_row_random = max_rows_random.loc[max_rows_random['S+ U S-'].apply(len).idxmin()]
    f_union_rdm = min_set_row_random['F(S+ U S-)']
    s_union_rdm = min_set_row_random['S+ U S-']
    
    ########
    startBF_greedy = time.time()
    S_greedy, F_greedy = greedy_label_reveal(edges=edgesx, labels=labelsx, budget=budgetx)
    endBF_greedy = time.time()
    ###
    startBF_rdm = time.time()
    S_random, F_random = random_label_reveal(edges=edgesx, labels=labelsx, budget=budgetx)
    endBF_rdm = time.time()
    
    ########
    Sfull_greedy, Ffull_greedy = greedy_label_reveal(edges=edgesx, labels=labelsx)
    ###
    Sfull_random, Ffull_random = random_label_reveal(edges=edgesx, labels=labelsx)
    
    
    #######
    results.append({
                "K": budgetx,
                "F(Sg+ U Sg-)*": f_union_greedy,
                "(Sg+ U Sg-)*": s_union_greedy,
                "F(Sr+ U Sr-)*": f_union_rdm,
                "(Sr+ U Sr-)*": s_union_rdm,
                ######
                "F(Sg)": F_greedy,
                "Sg": S_greedy,
                "F(Sr)": F_random,
                "Sr": S_random,
                ######
                "F(Sgfull)": Ffull_greedy,
                "Sgfull": Sfull_greedy,
                "F(Srfull)": Ffull_random,
                "Srfull": Sfull_random,
                ######
                "F(Sg+ U Sg-)*/F(Sr+ U Sr-)*": f_union_greedy/f_union_rdm if f_union_rdm>0 else 0,
                "F(Sg+ U Sg-)*/F(Sgfull)": f_union_greedy/Ffull_greedy if Ffull_greedy>0 else 0,
                "F(Sr+ U Sr-)*/F(Srfull)": f_union_rdm/Ffull_random if Ffull_random>0 else 0,
                ######
                "F(Sg)/F(Sr)": F_greedy/F_random if F_random>0 else 0,
                "F(Sg)/F(Sgfull)": F_greedy/Ffull_greedy if Ffull_greedy>0 else 0,
                "F(Sr)/F(Srfull)": F_random/Ffull_random if Ffull_random>0 else 0,
                ######
                "budgetGreedyTime": endBF_greedy-startBF_greedy,
                "budgetRdmTime": endBF_rdm-startBF_rdm,       
        
            })
    

    dframex = pd.DataFrame(results)
    dframex.reset_index(drop=True, inplace=True)
    
   
    return dframe_greedy, dframe_random, dframex






def format_value(x):
    if isinstance(x, int):
        return str(x)
    elif isinstance(x, float):
        if round(x,1) == x:
            return f"{x:.1f}"
        else:
            return f"{x:.2f}"
    else:
        return str(x)






def df_to_latex_preserve(df: pd.DataFrame) -> str:
    lines = df.to_string(index=False).splitlines()
    columns = lines[0].split()
    data_lines = lines[1:]
    latex = "\\begin{tabular}{%s}\n" % ("l" * len(columns))
    latex += " & ".join(columns) + " \\\\\n\\hline\n"
    latex += "\n".join([" & ".join(line.split()) + " \\\\" for line in data_lines])
    latex += "\n\\end{tabular}"
    return latex






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






def sm_results_plot(das, x_col, ycol1=None, ycol2=None, ycol3=None, ycol4=None, ycol1_name=None, 
                   ycol2_name=None, ycol3_name=None, ycol4_name=None, ycol5=None, ycol5_name=None,
                   save_as=None, figsize=(9, 7), showfig=False): 
    

    fig, ax = plt.subplots(figsize=figsize)


    ax.plot(das[x_col].unique(),
            das.groupby(x_col)[ycol3].mean(),
            color="black", linewidth=5, markersize=12, marker="s", label=ycol3_name)

    ax.plot(das[x_col].unique(),
            das.groupby(x_col)[ycol4].mean(),
            color="black", linestyle=":", linewidth=5, markersize=12, marker="s", label=ycol4_name)


    colors = plt.cm.tab10.colors

    for i, k in enumerate(sorted(das["K"].unique())):
        
        subset = das[das["K"] == k]
        
        ax.plot(subset[x_col], subset[ycol1],
                marker="o",
                color=colors[i % len(colors)],
                linewidth=5, markersize=12,
                alpha=0.8,
                label = f"{ycol1_name}K={k}")
        
        ax.plot(subset[x_col], subset[ycol2],
                marker="X",
                linestyle=":",
                linewidth=5, markersize=12,
                alpha=1,
                color=colors[i % len(colors)],
                label = f"{ycol2_name}K={k}")

    lgd = ax.legend(loc="upper center", 
                      frameon=True,
                      bbox_to_anchor=(0.45, 1.33),
                      prop={"family": "monospace", "weight": 530, "size": 21},
                      handlelength=1, handletextpad=0.4, columnspacing=0.5, 
                      labelspacing=0.1, ncol=4)
    

    if x_col == "kmax":
        xlbl = "Maximum #targets in agents' neighborhood"
    else:
        xlbl = "Threshold for agents' neighborhood"

    ax.set_xlabel(xlbl, labelpad=10, fontsize=28, fontweight=530)
    ax.set_ylabel("Social welfare", labelpad=10, fontsize=28, fontweight=530)

    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    ymin = das[ycol4].min()
    ymax = das[ycol3].max()

    margin = 0.25 * (ymax - ymin) if ymax > ymin else 1

    ax = plt.gca()
    ax.set_ylim(ymin - margin, ymax + margin) 
    ax.autoscale(False) 
    
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=25, fontweight=530)
    plt.yticks(fontsize=25, fontweight=530)
    sns.despine(top=True)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches='tight')

    if showfig:
        plt.show()
        
    plt.close()






def learnsetting_plot(summary_res, thresh, perfv, save_as):    
    
    sns.set(style="whitegrid")
    palette = sns.color_palette("tab10", n_colors=len(summary_res["K"].unique()))

    train_markers = ['o', 'o', 'o', 'o', 'o']
    test_markers  = ['X', 'X', 'X', 'X', 'X']
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if perfv == "perf1":
        ylb_name = r"Avg. $\mathrm{Perf}_{1}$ (%)"
        trdata = "tr_perf1"
        tsdata = "ts_perf1"
    elif perfv == "perf2":
        ylb_name = r"Avg. $\mathrm{Perf}_{2}$ (%)"
        trdata = "tr_perf2"
        tsdata = "ts_perf2"
    else:   
        ylb_name = r"Avg. $\mathrm{Perf}_{3}$ (%)"
        trdata = "tr_perf3"
        tsdata = "ts_perf3"  
        
    if thresh:
        xlb_name = "Threshold for agents' neighborhood"
        xval = "r"
    else:
        xlb_name = "Maximum #targets in agents' neighborhood"
        xval = "kmax"

    for i, k in enumerate(sorted(summary_res["K"].unique())):
        sub = summary_res[summary_res["K"] == k]
        color = palette[i % len(palette)]
        ax.plot(sub[xval], sub[trdata], 
                 marker=train_markers[i % len(train_markers)], 
                 alpha=0.8,
                 linestyle="-", color=color, label=f"tr,K={k}", linewidth=5, markersize=12)
        ax.plot(sub[xval], sub[tsdata], 
                 marker=test_markers[i % len(test_markers)], 
                 alpha=1,            
                 linestyle=":", color=color, label=f"ts,K={k}", linewidth=5, markersize=12)


    ax.set_xlabel(xlb_name, labelpad=5, fontsize=30, fontweight=530)
    ax.set_ylabel(ylb_name, labelpad=5, fontsize=34, fontweight=530)

        
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", 
               prop={'family':'monospace', "weight": 530, "size": 25},  
               handlelength=1, handletextpad=0.3, columnspacing=0.5, labelspacing=0.15,
               bbox_to_anchor=(0.5, 1.25), ncol=5, frameon=True)


    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(bottom=True, left=True, top=False, right=False, direction='out', 
                   length=6, width=1.2, labelsize=25)

    ymin = summary_res[tsdata].min()
    ymax = summary_res[trdata].max()

    if perfv != "perf3":
        margin = 0.07 *(ymax - ymin) if ymax > ymin else 1
        ax = plt.gca()
        bottom, top_limit = ymin - margin, 102
        ax.set_ylim(bottom, top_limit)

        span, step = 100 - bottom, (100 - bottom)/5
        power = 10**np.floor(np.log10(step))
        step = (np.array([1,2,5])*power)[np.argmin(np.abs(np.array([1,2,5])*power - step))]

        ticks = np.arange(bottom - bottom%step, 100 + step, step)
        ticks = ticks[ticks >= bottom]; ticks[-1] = 100
        if len(ticks) > 7: step*=2; ticks = np.arange(bottom - bottom%step, 100 + step, step); ticks[-1]=100

        if ticks[0] <= bottom:
            bottom_space = 0.05 * (ticks[-1] - ticks[0])
            ax.set_ylim(ticks[0] - bottom_space, ax.get_ylim()[1])

        ax.set_yticks(ticks)
        ax.autoscale(False) 

        
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=25, fontweight=530)
    plt.yticks(fontsize=25, fontweight=530)
    sns.despine(top=True)
    plt.tight_layout()

    plt.savefig(save_as, dpi=700, bbox_inches='tight')
    plt.show()






def compute_itm_results(datasgraphs, thresh, dsetname, dsetdim):
   
    summary_res = pd.DataFrame()
        
    if thresh:
        gval = "threshold"
        dfval = "r"
    else:
        gval = "k_max"
        dfval = "kmax"

    for lx in range(len(datasgraphs)):

        r_edges_randomx  = datasgraphs[lx]["edges"]
        r_labels_randomx  = datasgraphs[lx]["labels"]

        for Kx in [1, 2, 3, 4, 5]:

            for Bx in [4, 3, 2, 1, 0]:

                dfx = pd.DataFrame()

                #####   
                startBF_greedy = time.time()
                Sgdx, Fgdx = greedy_label_reveal(edges=r_edges_randomx, 
                                               labels=r_labels_randomx, 
                                               budget=Kx)
                endBF_greedy = time.time()

                #####
                startBF_greedyboost = time.time()
                dictax = greedy_boost_label_reveal(edgesx=r_edges_randomx, 
                                                  labelsx=r_labels_randomx, 
                                                  revealBgt=Kx, 
                                                  boostBgt=Bx)
                endBF_greedyboost = time.time()


                #####
                startBF_boostgreedy = time.time()
                dictbx = boost_greedy_label_reveal(edgesx=r_edges_randomx, 
                                                  labelsx=r_labels_randomx, 
                                                  revealBgt=Kx, 
                                                  boostBgt=Bx)
                endBF_boostgreedy = time.time()


                data_dictx = {
                    "K": Kx,
                    "B": Bx,
                    "Sg": Sgdx,
                    "Sg_lbls": {t: r_labels_randomx[t] for t in Sgdx},
                    "F(Sg)": Fgdx,

                    "Sgb": dictax['revealed_nodes'],
                    "Sgb_lbls": dictax['revealed_nodes_labels'],
                    "F(Sgb)b4B": dictax['utility_before_boost'],
                    "F(Sgb)": dictax['utility_after_boost'],
                    "Boostedgb": dictax['boosted_agents'],
                    "usedBbudgetgb": dictax['usableBoostBudget'],

                    "Sbg": dictbx['revealed_nodes'],
                    "Sbg_lbls": dictbx['revealed_nodes_labels'],
                    "F(Sbg)b4B": dictbx['final_utility-B'],
                    "F(Sbg)": dictbx['final_utility+B'],
                    "Boostedbg": dictbx['removed_agents'],
                    "usedBbudgetbg": dictbx['usableBoostBudget'],

                    dfval: datasgraphs[lx][gval],
                    "n": datasgraphs[lx]['n'],
                    "m": datasgraphs[lx]['m'],

                    "greedyTime": endBF_greedy-startBF_greedy,
                    "greedyBoostTime": endBF_greedyboost-startBF_greedyboost,
                    "boostGreedyTime": endBF_boostgreedy-startBF_boostgreedy,

                    "dataset": dsetname + f" ({dsetdim})",
                    "graphid": lx
                }

                dfx = pd.DataFrame([data_dictx])

                #####
                summary_res = pd.concat([summary_res, dfx], ignore_index=True)
                

    return summary_res





 
def plot_intm_results(resdf, thresh, save_as):
    
    colors = plt.cm.tab10.colors
    markers_bg = ['o', 'o', 'o', 'o']
    markers_gb = ['X', 'X', 'X', 'X']
    
    if thresh:
        xlb_name = "Threshold for agents' neighborhood"
        xval = "r"
    else:
        xlb_name = "Maximum #targets in agents' neighborhood"
        xval = "kmax"

    for k, group_k in resdf.groupby("K"):
        fig, ax = plt.subplots(figsize=(12, 6))
        group_k = group_k.copy()

        group_k["d(Sbg−Sg)"] = group_k["F(Sbg)"] - group_k["F(Sg)"]
        group_k["d(Sgb−Sg)"] = group_k["F(Sgb)"] - group_k["F(Sg)"]

        for i, (b, group_b) in enumerate(group_k.groupby("B")):
            color = colors[i % len(colors)]
            marker_bg = markers_bg[i % len(markers_bg)]
            marker_gb = markers_gb[i % len(markers_gb)]

            ax.plot(group_b[xval], group_b["d(Sbg−Sg)"],
                    linestyle="-", color=color, marker=marker_bg, markersize=12,
                    linewidth=5, alpha=0.8,
                    label=fr"$\Delta_F(\mathrm{{ig,g}})$, B={b}")

            ax.plot(group_b[xval], group_b["d(Sgb−Sg)"],
                    linestyle=":", color=color, marker=marker_gb, markersize=12,
                    linewidth=5, alpha=1,
                    label=fr"$\Delta_F(\mathrm{{gi,g}})$, B={b}")
            

        ax.set_xlabel(xlb_name, labelpad=5, fontsize=30, fontweight=530)
        ax.set_ylabel("Gain from intervention", labelpad=5, fontsize=30, fontweight=530)

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        handles, labels = ax.get_legend_handles_labels()
        title_handle = mlines.Line2D([], [], linestyle='none')

        handles.append(title_handle)
        labels.append(r"$\mathbf{K = %d}$" % k)

        legend = fig.legend(handles, labels, loc="upper center",
                            prop={"weight": 530, "size": 24},
                            handlelength=1, handletextpad=0.4, columnspacing=0.8,
                            labelspacing=0.15, bbox_to_anchor=(0.5, 1.31), ncol=4, frameon=True)
        
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(bottom=True, left=True, top=False, right=False, 
                       direction='out', length=6, width=1.2, labelsize=14)

        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        plt.setp(legend.get_title(), fontsize=18)

        ax = plt.gca()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(fontsize=25, fontweight=530)
        plt.yticks(fontsize=25, fontweight=530)
        sns.despine(top=True)
        plt.tight_layout()


        plt.savefig(save_as+str(k)+".pdf", dpi=300, bbox_inches='tight')
        plt.show()
    


    

def add_meta(dff, agraph, agid, thresh):
    dff = dff.copy()
    if thresh:
        dff["r"] = agraph["threshold"]
    else:
        dff["kmax"] = agraph["k_max"]
    dff["n"] = agraph["n"]
    dff["m"] = agraph["m"]
    dff["graphid"] = agid
    return dff
    


    

def safe_ratio(num, den):
    return num / den if num > 0 else 0
    


    

def fairness_greedy_random(bgraphs, dname, budgets, thresh=False):
    
    summary, summary_g0, summary_g1 = [], [], []
    greedy_all, random_all = [], []
    group_results = []

    for gid, gph in enumerate(bgraphs):
        
        edges, labels = gph["edges"], gph["labels"]

        g0edges = {u: e for u, e in edges.items() if gph["agent_protectatts"].get(u) == 0}
        g1edges = {u: e for u, e in edges.items() if gph["agent_protectatts"].get(u) == 1}

        g0targets = {t: labels[t] for t in labels if gph["target_protectatts"].get(t) == 0}
        g1targets = {t: labels[t] for t in labels if gph["target_protectatts"].get(t) == 1}

        for K in budgets:

            df_g, df_r, df = heuristic_greedy_random(edgesx=edges, 
                                                     labelsx=labels,
                                                     budgetx=K)
            df = add_meta(df, gph, gid, thresh)
            df["dataset"] = dname
            df["group0edges"], df["group1edges"] = g0edges, g1edges
            df["group0targets"], df["group1targets"] = g0targets, g1targets
            df_g["graphid"] = gid
            df_r["graphid"] = gid

            summary.append(df)
            greedy_all.append(df_g)
            random_all.append(df_r)

            #############
            df_g0_g, df_g0_r, df_g0 = heuristic_greedy_random(edgesx=g0edges, 
                                                              labelsx=labels, 
                                                              budgetx=math.floor(K/2))
            df_g0 = add_meta(df_g0, gph, gid, thresh)
            summary_g0.append(df_g0)

            ############
            df_g1_g, df_g1_r, df_g1 = heuristic_greedy_random(edgesx=g1edges, 
                                                              labelsx=labels, 
                                                              budgetx=math.floor(K/2))
            df_g1 = add_meta(df_g1, gph, gid, thresh)
            summary_g1.append(df_g1)


            def SW(edges_subset, revealed_subset):
                return F_S(edges=edges_subset, labels=labels, revealed=revealed_subset)
            
            if thresh:
                valg = "r"
                valname = "threshold"
            else:
                valg = "kmax"
                valname = "k_max"


            row = {
                "K": K,
                valg: gph[valname],
                "n": gph["n"],
                "n_g0": len(g0edges),
                "n_g1": len(g1edges),
                "m": gph["m"],
                "dataset": dname,
                "graphid": gid,
            }
            

            for tag in ["(Sg+ U Sg-)*", "(Sr+ U Sr-)*", "Sg", "Sr"]:
                
                if tag == "Sg" or tag == "Sr":
                    row[f"g0F({tag})"] = safe_ratio(SW(g0edges, df[tag].item()), len(g0edges))
                    row[f"g1F({tag})"] = safe_ratio(SW(g1edges, df[tag].item()), len(g1edges))
                    ####
                    row[f"g0-F({tag})"] = safe_ratio(df_g0[f"F({tag})"].item(), len(g0edges))
                    row[f"g1-F({tag})"] = safe_ratio(df_g1[f"F({tag})"].item(), len(g1edges))
                    ####
                    row[f"g0Qx({tag})"] = compute_Qx(edgez=g0edges, labelz=labels, revealed_setz=df[tag].item())
                    row[f"g1Qx({tag})"] = compute_Qx(edgez=g1edges, labelz=labels, revealed_setz=df[tag].item())
                    
                else:
                    row[f"g0F{tag}"] = safe_ratio(SW(g0edges, df[tag].item()), len(g0edges))
                    row[f"g1F{tag}"] = safe_ratio(SW(g1edges, df[tag].item()), len(g1edges))
                    ####
                    row[f"g0-F{tag}"] = safe_ratio(df_g0[f"F{tag}"].item(), len(g0edges))
                    row[f"g1-F{tag}"] = safe_ratio(df_g1[f"F{tag}"].item(), len(g1edges))
                    ####
                    row[f"g0Qx{tag}"] = compute_Qx(edgez=g0edges, labelz=labels, revealed_setz=df[tag].item())
                    row[f"g1Qx{tag}"] = compute_Qx(edgez=g1edges, labelz=labels, revealed_setz=df[tag].item())                

            group_results.append(row)
            

    return {
        "summary": pd.concat(summary, ignore_index=True),
        "summary_g0": pd.concat(summary_g0, ignore_index=True),
        "summary_g1": pd.concat(summary_g1, ignore_index=True),
        "greedy": pd.concat(greedy_all, ignore_index=True),
        "random": pd.concat(random_all, ignore_index=True),
        "group_results": pd.DataFrame(group_results),
    }
    


    

def plot_fairness(df, g1val, g0val, save_as=False, thresh=False):

    Ks = sorted(df["K"].unique())
    colors = ["#7f7f7f",  "#E69F00", "#56B4E9", "#009E73", 
              "#F0E442", "#0072B2",  "#D55E00", "#CC79A7"]
    
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    if thresh:
        valg = "r"
        valname = "threshold"
        xlabel = "Threshold for agents' neighborhood"
    else:
        valg = "kmax"
        valname = "kmax"
        xlabel = "Maximum #targets in agents' neighborhood"


    for K, c in zip(Ks, colors):
        sub = df[df["K"] == K].sort_values(valg)


        ## male 1, female 0
        plt.plot(sub[valg], sub[g1val], color=c, linestyle="-",
                 linewidth=5, marker="o", markersize=12, alpha=1, label=f"K={K}")

        plt.plot(sub[valg], sub[g0val], color=c, linestyle=":", 
                 linewidth=5, marker="o", markersize=12, alpha=1)

    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel("Avg. Group Social Welfare", fontsize=25)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)

    leg = plt.legend(
                title="Line style: Solid: Male, Dashed: Female",
                title_fontsize=20,
                fontsize=23,
                ncol=4,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.3),
                frameon=True,
            )

    leg.get_title().set_fontweight("bold")


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    if save_as:
        plt.savefig(save_as, dpi=300, bbox_inches='tight')
        
    plt.show()
    


    

def group_graph_info(bgraph, edges, namex, d, graph_id, thresh=False):
    
    avg_left_deg, avg_left_pos_deg, avg_left_neg_deg, avg_right_deg,\
        avg_overlap, avg_pos_overlap, avg_neg_overlap,\
        only_pos, only_neg, empty_adj, unipos = getconnectivity_info(edges, bgraph["labels"])
    
    if thresh:
        valg = "r"
        valname = "threshold"
    else:
        valg = "kmax"
        valname = "k_max"

    
    labels = bgraph["labels"]

    return {
        "Dataset (d)": f"{namex} ({d})",
        valg: bgraph[valname],
        "n": len(edges),
        "m": bgraph["m"],
        "#+ves": sum(l == 1 for l in labels.values()),
        "#-ves": sum(l == -1 for l in labels.values()),
        "avg LHSd": round(avg_left_deg, 2),
        "avg LHS+d": round(avg_left_pos_deg, 2),
        "avg LHS-d": round(avg_left_neg_deg, 2),
        "avg RHSd": round(avg_right_deg, 2),
        "avg overlap": round(avg_overlap, 2),
        "avg overlap+": round(avg_pos_overlap, 2),
        "avg overlap-": round(avg_neg_overlap, 2),
        "only+Ns": only_pos,
        "only-Ns": only_neg,
        "emptyNs": empty_adj,
        "uni+": unipos,
        "graphID": graph_id,
    }
    


    

def alt_fairness_greedy(bgraphs, dname, thresh=False):

    summary_dfw = pd.DataFrame()
    
    if thresh:
        valg = "r"
        valname = "threshold"
    else:
        valg = "kmax"
        valname = "k_max"

        
    for lw, bgh in enumerate(bgraphs):

        dfw = pd.DataFrame()

        for Kw in [0, 2, 4, 6]:

            group0edges = {g: bgh["edges"][g] for g in bgh["edges"] if bgh["agent_protectatts"].get(g) == 0}
            group1edges = {g: bgh["edges"][g] for g in bgh["edges"] if bgh["agent_protectatts"].get(g) == 1}

            group0targets = {t: bgh["labels"][t] for t in bgh["labels"] if bgh['target_protectatts'].get(t) == 0}
            group1targets = {t: bgh["labels"][t] for t in bgh["labels"] if bgh['target_protectatts'].get(t) == 1}

            ########
            startBF_greedyw = time.time()
            S_greedyw, F_greedyw = greedy_label_reveal(edges=bgh["edges"], 
                                                       labels=bgh["labels"], 
                                                       budget=Kw)
            endBF_greedyw = time.time()

            ###
            startBF_groupgreedyw0 = time.time()
            S_greedyw0, F_greedyw0 = groupspecific_greedy_label_reveal(edges=bgh["edges"], 
                                                                       labels=bgh["labels"], 
                                                                       budget=Kw,
                                                                       favorgroup=0, 
                                                                       gZeroEdges=group0edges,
                                                                       gOneEdges=group1edges)
            endBF_groupgreedyw0 = time.time()

            ###
            startBF_groupgreedyw1 = time.time()
            S_greedyw1, F_greedyw1 = groupspecific_greedy_label_reveal(edges=bgh["edges"], 
                                                                       labels=bgh["labels"], 
                                                                       budget=Kw,
                                                                       favorgroup=1, 
                                                                       gZeroEdges=group0edges,
                                                                       gOneEdges=group1edges)
            endBF_groupgreedyw1 = time.time()


            resultdfw = {
                'K': Kw,            
                valg:  bgh[valname],
                "n": bgh['n'],
                "m":  bgh['m'],
                'Sg': S_greedyw,
                'F(Sg)': F_greedyw,
                'g0F(Sg)': safe_ratio(F_S(group0edges, bgh["labels"], S_greedyw), len(group0edges)),
                'g1F(Sg)': safe_ratio(F_S(group1edges, bgh["labels"], S_greedyw), len(group1edges)),

                'Sg0': S_greedyw0,
                'F(Sg0)': F_greedyw0,
                'g0F(Sg0)': safe_ratio(F_S(group0edges, bgh["labels"], S_greedyw0), len(group0edges)),
                'g1F(Sg0)': safe_ratio(F_S(group1edges, bgh["labels"], S_greedyw0), len(group1edges)),

                'Sg1': S_greedyw1,
                'F(Sg1)': F_greedyw1,
                'g0F(Sg1)': safe_ratio(F_S(group0edges, bgh["labels"], S_greedyw1), len(group0edges)),
                'g1F(Sg1)': safe_ratio(F_S(group1edges, bgh["labels"], S_greedyw1), len(group1edges)),

                'greedyTime': endBF_greedyw-startBF_greedyw,
                'greedyFav0Time': endBF_groupgreedyw0-startBF_groupgreedyw0,
                'greedyFav1Time': endBF_groupgreedyw1-startBF_groupgreedyw1,

                'group0edges': group0edges,
                'group1edges': group1edges,
                "dataset": dname,
                "graphid": lw
            }

            summary_dfw = pd.concat([summary_dfw, pd.DataFrame.from_records([resultdfw])], ignore_index=True)
            
    return summary_dfw
