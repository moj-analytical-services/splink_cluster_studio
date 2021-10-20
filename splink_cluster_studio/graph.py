import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms.centrality import (
    eigenvector_centrality,
    edge_betweenness_centrality,
)
from networkx.algorithms.distance_measures import diameter
from networkx.algorithms.cluster import transitivity
from networkx.algorithms.bridges import bridges
from networkx.algorithms.community.centrality import girvan_newman
import networkx.algorithms.community as nx_comm


def weight_from_prob(df, prob_colname, weight_colname="match_weight"):

    f1 = df[prob_colname] == 1
    df.loc[f1, weight_colname] = 40

    f2 = df[prob_colname] == 0
    df.loc[f2, weight_colname] = -40

    f3 = ~(f1 | f2)
    df.loc[f3, weight_colname] = np.log2(df[prob_colname] / (1 - df[prob_colname]))
    return df


def _apply_edge_betweenness(pdf, unique_id_colname, weight_colname):

    nxGraph = nx.Graph()
    nxGraph = nx.from_pandas_edgelist(
        pdf, f"{unique_id_colname}_l", f"{unique_id_colname}_r", weight_colname
    )
    eb = edge_betweenness_centrality(nxGraph, normalized=True, weight=weight_colname)
    unpacked = [
        (min(r[0][0], r[0][1]), max(r[0][0], r[0][1]), r[1]) for r in eb.items()
    ]
    eb = pd.DataFrame(
        unpacked,
        columns=[
            f"{unique_id_colname}_l",
            f"{unique_id_colname}_r",
            "edge_betweenness",
        ],
    )
    eb.index.name = "groupby_index"
    return eb


def edge_betweenness(
    df_edges, cluster_id_colname, weight_colname, unique_id_colname="unique_id"
):

    eb_df = df_edges.groupby(f"{cluster_id_colname}_l").apply(
        _apply_edge_betweenness, unique_id_colname, weight_colname
    )

    eb_df = eb_df.reset_index().drop(
        [f"{cluster_id_colname}_l", "groupby_index"], axis=1
    )

    join_cols = [f"{unique_id_colname}_l", f"{unique_id_colname}_r"]
    df_edges_eb = df_edges.merge(
        eb_df, left_on=join_cols, right_on=join_cols, how="left"
    )
    return df_edges_eb


def _apply_is_bridge(pdf, unique_id_colname, weight_colname):

    nxGraph = nx.Graph()
    nxGraph = nx.from_pandas_edgelist(
        pdf, f"{unique_id_colname}_l", f"{unique_id_colname}_r", weight_colname
    )
    list_of_bridges = bridges(nxGraph)

    list_of_bridges = [(min(r[0], r[1]), max(r[0], r[1])) for r in list_of_bridges]

    is_bridge = pd.DataFrame(
        list_of_bridges,
        columns=[
            f"{unique_id_colname}_l",
            f"{unique_id_colname}_r",
        ],
    )
    is_bridge.index.name = "groupby_index"
    is_bridge["is_bridge"] = True

    return is_bridge


def is_bridge(
    df_edges, cluster_id_colname, weight_colname, unique_id_colname="unique_id"
):

    bridge_df = df_edges.groupby(f"{cluster_id_colname}_l").apply(
        _apply_is_bridge, unique_id_colname, weight_colname
    )

    bridge_df = bridge_df.reset_index().drop(
        [f"{cluster_id_colname}_l", "groupby_index"], axis=1
    )

    join_cols = [f"{unique_id_colname}_l", f"{unique_id_colname}_r"]
    df_edges_eb = df_edges.merge(
        bridge_df, left_on=join_cols, right_on=join_cols, how="left"
    )
    df_edges_eb["is_bridge"] = df_edges_eb["is_bridge"].fillna(False)
    return df_edges_eb


def _apply_eigen_centrality(df_edges, unique_id_colname, weight_colname):
    nxGraph = nx.Graph()
    nxGraph = nx.from_pandas_edgelist(
        df_edges, f"{unique_id_colname}_l", f"{unique_id_colname}_r", weight_colname
    )
    ec = eigenvector_centrality(nxGraph, tol=1e-03)
    out_df = pd.DataFrame.from_dict(ec, orient="index", columns=["eigen_centrality"])
    out_df.index.name = "unique_id"
    return out_df


def eigen_centrality(
    df_nodes,
    df_edges,
    cluster_id_colname,
    weight_colname,
    unique_id_colname="unique_id",
):
    df_ec = df_edges.groupby(f"{cluster_id_colname}_l").apply(
        _apply_eigen_centrality, unique_id_colname, weight_colname
    )
    df_ec = df_ec.reset_index()
    df_ec = df_ec.drop(f"{cluster_id_colname}_l", axis=1)

    df_nodes_ec = df_nodes.merge(
        df_ec, left_on=unique_id_colname, right_on=unique_id_colname, how="left"
    )

    return df_nodes_ec


def _apply_cluster_simple_stats(df_edges, unique_id_colname):

    unique_nodes = set(df_edges[f"{unique_id_colname}_l"]).union(
        set(df_edges[f"{unique_id_colname}_r"])
    )
    num_nodes = len(unique_nodes)
    max_possible_edges = (num_nodes * (num_nodes - 1)) / 2
    num_edges = len(df_edges)
    density = num_edges / max_possible_edges

    df = pd.DataFrame(
        [{"num_nodes": num_nodes, "num_edges": num_edges, "density": density}]
    )
    df.index.name = "to_drop"
    return df


def cluster_simple_stats(df_edges, cluster_id_colname, unique_id_colname="unique_id"):
    df_c = df_edges.groupby(f"{cluster_id_colname}_l").apply(
        _apply_cluster_simple_stats, unique_id_colname
    )

    df_c = df_c.reset_index()
    df_c = df_c.rename({f"{cluster_id_colname}_l": cluster_id_colname}, axis=1)
    df_c = df_c.drop("to_drop", axis=1)
    return df_c


def _apply_cluster_networkx_stats(df_edges, unique_id_colname, weight_colname):
    nxGraph = nx.Graph()
    nxGraph = nx.from_pandas_edgelist(
        df_edges, f"{unique_id_colname}_l", f"{unique_id_colname}_r", weight_colname
    )
    d = diameter(nxGraph)
    t = transitivity(nxGraph)
    tric = nx.average_clustering(nxGraph)
    sq = nx.square_clustering(nxGraph)
    sqc = sum(sq.values()) / len(sq.values())

    nc = nx.algorithms.node_connectivity(nxGraph)
    ec = nx.algorithms.edge_connectivity(nxGraph)

    # EB modularity
    def largest_edge_betweenness(G):
        centrality = edge_betweenness_centrality(
            G, weight=weight_colname, normalized=True
        )
        return max(centrality, key=centrality.get)

    comp = girvan_newman(nxGraph, most_valuable_edge=largest_edge_betweenness)
    gn = tuple(sorted(c) for c in next(comp))

    nc = nx.number_of_nodes(nxGraph)

    if nc > 2:
        try:
            co_eb_mod = nx_comm.modularity(nxGraph, gn)
        except ZeroDivisionError:
            raise Exception(
                f"ZeroDivisionError on component"
                "This can occur if one of the weights (distances) is zero."
            )
    else:
        co_eb_mod = -1.0

    df = pd.DataFrame(
        [
            {
                "diameter": d,
                "transitivity": t,
                "tri_clustcoeff": tric,
                "sq_clustcoeff": sqc,
                "node_conn": nc,
                "edge_conn": ec,
                "cluster_eb_modularity": co_eb_mod,
            }
        ]
    )
    df.index.name = "to_drop"
    return df


def cluster_networkx_stats(
    df_edges, cluster_id_colname, weight_colname, unique_id_colname="unique_id"
):
    df_c = df_edges.groupby(f"{cluster_id_colname}_l").apply(
        _apply_cluster_networkx_stats, unique_id_colname, weight_colname
    )

    df_c = df_c.reset_index()
    df_c = df_c.rename({f"{cluster_id_colname}_l": cluster_id_colname}, axis=1)
    df_c = df_c.drop("to_drop", axis=1)
    return df_c


def _add_match_weight_if_not_exists(edges_with_clusters_pd, prob_colname=None):
    cols = list(edges_with_clusters_pd.columns)
    if "match_weight" not in cols:
        if prob_colname is None:

            if "tf_adjusted_match_prob" in cols:
                prob_colname = "tf_adjusted_match_prob"
            else:
                prob_colname = "match_probability"

        edges_with_clusters_pd = weight_from_prob(edges_with_clusters_pd, prob_colname)
    return edges_with_clusters_pd


def compute_node_metrics(
    nodes_with_clusters_pd: pd.DataFrame,
    edges_with_clusters_pd: pd.DataFrame,
    cluster_colname: str,
    prob_colname: str = None,
    ground_truth_cluster_colname: str = None,
):
    """Compute node metrics - specifically eigen_centrality

    Args:
        nodes_with_clusters_pd (pd.DataFrame): Example: https://gist.github.com/RobinL/73f7101cb2e1145ce8820bafdd15e988
        edges_with_clusters_pd (pd.DataFrame): Example: https://gist.github.com/RobinL/73f7101cb2e1145ce8820bafdd15e988
        cluster_colname (str): Name of cluster column name e.g. cluster_low
        prob_colname (bool, optional): Allows user to explicitly select the column containing match probabilities, otherwie
            it is autodetected. Defaults to None.
        ground_truth_cluster_colname (str, optional): Column containing ground truth cluster labels.
            If provided, false positive links can be detected. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing node metrics
    """

    edges_with_clusters_pd = _add_match_weight_if_not_exists(
        edges_with_clusters_pd, prob_colname
    )

    nodes_for_vis_pd = eigen_centrality(
        nodes_with_clusters_pd,
        edges_with_clusters_pd,
        cluster_colname,
        "match_weight",
    )

    if ground_truth_cluster_colname:
        nodes_for_vis_pd = nodes_for_vis_pd.rename(
            columns={ground_truth_cluster_colname: "ground_truth_cluster"}
        )

    return nodes_for_vis_pd


def compute_edge_metrics(
    edges_with_clusters_pd: pd.DataFrame,
    cluster_colname: str,
    prob_colname: str = None,
    ground_truth_cluster_colname: str = None,
):
    """Compute edge metrics - specifically edge_betweenness and is_bridge

    Args:
        edges_with_clusters_pd (pd.DataFrame): Example: https://gist.github.com/RobinL/73f7101cb2e1145ce8820bafdd15e988
        cluster_colname (str): Name of cluster column name e.g. cluster_low
        prob_colname (bool, optional): Allows user to explicitly select the column containing match probabilities, otherwie
            it is autodetected. Defaults to None.
        ground_truth_cluster_colname (str, optional): Column containing ground truth cluster labels.
            If provided, false positive links can be detected. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing edge metrics
    """

    edges_with_clusters_pd = _add_match_weight_if_not_exists(
        edges_with_clusters_pd, prob_colname
    )

    edges_with_clusters_pd = edge_betweenness(
        edges_with_clusters_pd, cluster_colname, "match_weight"
    )
    edges_with_clusters_pd = is_bridge(
        edges_with_clusters_pd, cluster_colname, "match_weight"
    )
    if ground_truth_cluster_colname:
        edges_with_clusters_pd["is_false_positive"] = (
            edges_with_clusters_pd[f"{ground_truth_cluster_colname}_l"]
            != edges_with_clusters_pd[f"{ground_truth_cluster_colname}_r"]
        )

    return edges_with_clusters_pd


def compute_cluster_metrics(
    edges_with_clusters_pd: pd.DataFrame,
    cluster_colname: str,
    simple_only: bool = False,
    prob_colname: bool = None,
):
    """Compute a table of cluster metrics e.g. density corresponding to the edges provided

    Args:
        edges_with_clusters_pd (pd.DataFrame): Example: https://gist.github.com/RobinL/73f7101cb2e1145ce8820bafdd15e988
        cluster_colname (str): Name of cluster column name e.g. cluster_low
        simple_only (bool, optional): Compute only low-computational-intensity metrics. Defaults to False.
        prob_colname (bool, optional): Allows user to explicitly select the column containing match probabilities, otherwie
            it is autodetected. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing cluster metrics
    """

    edges_with_clusters_pd = _add_match_weight_if_not_exists(
        edges_with_clusters_pd, prob_colname
    )

    df_cluster_stats = cluster_simple_stats(edges_with_clusters_pd, cluster_colname)
    if simple_only:
        return df_cluster_stats

    df_cluster_stats_nx = cluster_networkx_stats(
        edges_with_clusters_pd, cluster_colname, "match_weight"
    )
    df_cluster_stats = df_cluster_stats.merge(
        df_cluster_stats_nx,
        left_on=cluster_colname,
        right_on=cluster_colname,
        how="left",
    )
    return df_cluster_stats
