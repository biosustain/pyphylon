"""
Functions for running Mash analysis.
"""

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hc
import scipy.spatial as sp


def cluster_corr_dist(df_mash_corr_dist, thresh=0.1, method="ward", metric="euclidean"):
    """
    Hierarchically Mash-based pairwise-pearson-distance matrix
    """
    link = hc.linkage(sp.distance.squareform(df_mash_corr_dist), method=method, metric=metric)
    dist = sp.distance.squareform(df_mash_corr_dist)

    clst = pd.DataFrame(index=df_mash_corr_dist.index)
    clst["cluster"] = hc.fcluster(link, thresh * dist.max(), "distance")

    return link, dist, clst


# Sensitivity analysis to pick the threshold (for E. coli we use 0.1)
# We pick the threshold where the curve just starts to bottom out
def sensitivity_analysis(df_mash_corr_dist_complete):
    from kneebow.rotor import Rotor

    x = list(np.logspace(-3, -1, 10)) + list(np.linspace(0.1, 1, 19))

    def num_uniq_clusters(thresh):
        link = hc.linkage(sp.distance.squareform(df_mash_corr_dist_complete), method="ward", metric="euclidean")
        dist = sp.distance.squareform(df_mash_corr_dist_complete)

        clst = pd.DataFrame(index=df_mash_corr_dist_complete.index)
        clst["cluster"] = hc.fcluster(link, thresh * dist.max(), "distance")

        return len(clst.cluster.unique())

    thresh_v_n_clusters_df = pd.DataFrame()
    thresh_v_n_clusters_df["threshold"] = pd.Series(x)
    thresh_v_n_clusters_df["num_clusters"] = pd.Series(x).apply(num_uniq_clusters)

    # Find which value the elbow corresponds to
    thresh_v_n_clusters_sorted_df = thresh_v_n_clusters_df.sort_values(by="num_clusters", ascending=True).reset_index(
        drop=True
    )

    # transform input into form necessary for package
    results_itr = zip(list(thresh_v_n_clusters_sorted_df.index), list(thresh_v_n_clusters_sorted_df.num_clusters))
    data = list(results_itr)

    rotor = Rotor()
    rotor.fit_rotate(data)
    elbow_idx = rotor.get_elbow_index()
    thresh_v_n_clusters_sorted_df["num_clusters"][elbow_idx]

    # Grab elbow threshold
    cond = thresh_v_n_clusters_df["num_clusters"] == thresh_v_n_clusters_sorted_df["num_clusters"][elbow_idx]
    elbow_threshold = thresh_v_n_clusters_df[cond]["threshold"].iloc[0]

    return thresh_v_n_clusters_sorted_df, elbow_idx, elbow_threshold
