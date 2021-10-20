import pandas as pd

pyspark_installed = True
try:
    import pyspark
except ImportError:
    pyspark_installed = False
    pyspark = None


def sql_in_values_expr(cluster_ids_list):
    elem = cluster_ids_list[0]
    elements_are_strings = type(elem) == str
    if elements_are_strings:
        cluster_ids_list = [f'"{e}"' for e in cluster_ids_list]
    values_comma_separated = ", ".join(map(str, cluster_ids_list))
    expr = f" IN ({values_comma_separated})"
    return expr


def get_nodes_corresponding_to_clusters_sql(
    nodes_table_name: str,
    cluster_colname: str,
    cluster_ids_list: list,
) -> str:
    """Generate a sql statement to get the nodes corresponding to a list of clusters
    i.e. filter nodes down to only those corresponding to cluster_ids_list

    Args:
        nodes_table_name (str): Nodes table name, can be qualified with db name e.g. nodes_with_clusters or my_db.nodes_with_clusters.
        cluster_colname (str):
        cluster_ids_list (list):

    Returns:
        str: sql statement
    """

    in_expr = sql_in_values_expr(cluster_ids_list)

    sql = f"""
    select * from {nodes_table_name}
    where {cluster_colname} {in_expr}
    """

    return sql


def get_edges_corresponding_to_clusters_sql(
    nodes_with_clusters_table_name: str,
    edges_table_name: str,
    cluster_colname: str,
    cluster_ids_list: list,
    use_source_dataset_to_join: bool = True,
    threshold_filter_expr: str = None,
    unique_id_colname: str = "unique_id",
):
    """Generate a sql statement to get the edges corresponding to a list of clusters
    i.e. filter edges down to only those corresponding to cluster_ids_list

    Args:
        nodes_with_clusters_table_name (str): Nodes table name, can be qualified with db name e.g. nodes_with_clusters or my_db.nodes_with_clusters.
        edges_table_name (str): Edges table name, can be qualified with db name
        cluster_colname (str):  e.g. cluster_medium
        cluster_ids_list (list):
        use_source_dataset_to_join (bool, optional): . Defaults to True.
        threshold_filter_expr (str, optional): . Defaults to None.
        unique_id_colname (str, optional): . Defaults to "unique_id".

    Returns:
        str: SQL statement
    """

    if threshold_filter_expr:
        thres_filter = f" AND {threshold_filter_expr}"
    else:
        thres_filter = ""

    source_l_expr = ""
    source_r_expr = ""
    if use_source_dataset_to_join:
        source_l_expr = f"AND {edges_table_name}.source_dataset_l = nc1.source_dataset"
        source_r_expr = f"AND {edges_table_name}.source_dataset_r = nc2.source_dataset"

    in_expr = sql_in_values_expr(cluster_ids_list)

    sql = f"""
    SELECT
        {edges_table_name}.*,
        nc1.{cluster_colname} AS {cluster_colname}_l,
        nc2.{cluster_colname} AS {cluster_colname}_r
    FROM
        {edges_table_name}
        LEFT JOIN {nodes_with_clusters_table_name} AS nc1
            ON {edges_table_name}.{unique_id_colname}_l = nc1.{unique_id_colname}
        {source_l_expr}
        LEFT JOIN {nodes_with_clusters_table_name} AS nc2
            ON {edges_table_name}.{unique_id_colname}_r = nc2.{unique_id_colname}
        {source_r_expr}
    WHERE
        nc1.{cluster_colname} = nc2.{cluster_colname}
        AND nc1.{cluster_colname} {in_expr}
        {thres_filter}

    """

    return sql


def get_nodes_corresponding_to_clusters_from_spark(
    df_nodes_with_clusters: pyspark.sql.DataFrame,
    cluster_colname: str,
    cluster_ids_list: list,
) -> pd.DataFrame:
    """Get a pandas dataframe of all the nodes associated with a
    list of clusters using Spark

    Args:
        df_nodes_with_clusters (DataFrame):
        cluster_colname (str):
        cluster_ids_list (list):

    Returns:
        pd.DataFrame:
    """

    spark = df_nodes_with_clusters.sql_ctx.sparkSession

    df_nodes_with_clusters.createOrReplaceTempView("df_nodes_with_clusters")

    sql = get_nodes_corresponding_to_clusters_sql(
        "df_nodes_with_clusters",
        cluster_colname,
        cluster_ids_list,
    )

    return spark.sql(sql).toPandas()


def get_edges_corresponding_to_clusters_from_spark(
    df_nodes_with_clusters: pyspark.sql.DataFrame,
    df_edges: pyspark.sql.DataFrame,
    cluster_colname: str,
    cluster_ids_list: list,
    threshold_filter_expr=None,
    unique_id_colname="unique_id",
) -> pd.DataFrame:
    """Get a pandas dataframe of the edges corresponding to
    a list of clusters using Spark.

    Args:
        df_nodes_with_clusters (DataFrame): DataFrame of nodes and clusters.
        edges (DataFrame): DataFrame of edges.
        cluster_colname (str):
        cluster_ids_list (str):
        threshold_filter_expr ([type], optional): If you want to filter out edges below
            a certain threshold. Defaults to None.
        unique_id_colname (str, optional): Defaults to "unique_id".

    The list of edges may include edges belonging to a cluster, but where the match
    probability did not exceed the clustering threshold.
    This would happen in the case of a transitive match  (e.g. A->B, B->C implies
    A->C even though A->C was below the threshold)
    Do we want to show these edges in the vis?  If so, we need to know what threshold
    condition was used to generate the clusters

    Returns:
        pd.DataFrame
    """
    spark = df_nodes_with_clusters.sql_ctx.sparkSession
    df_edges.createOrReplaceTempView("df_edges")
    df_nodes_with_clusters.createOrReplaceTempView("df_nodes_with_clusters")

    use_source_dataset_to_join = "source_dataset" in df_nodes_with_clusters.columns

    sql = get_edges_corresponding_to_clusters_sql(
        "df_nodes_with_clusters",
        "df_edges",
        cluster_colname,
        cluster_ids_list,
        threshold_filter_expr=threshold_filter_expr,
        unique_id_colname=unique_id_colname,
        use_source_dataset_to_join=use_source_dataset_to_join,
    )

    return spark.sql(sql).toPandas()
