from jinja2 import Template
import json
import os
import pkgutil
import pandas as pd


def render_html_vis(
    nodes_with_clusters_pd: pd.DataFrame,
    edges_corresponding_to_clusters_pd: pd.DataFrame,
    splink_settings: dict,
    out_path: str,
    cluster_colname: str,
    df_cluster_metrics: pd.DataFrame = None,
    named_clusters: dict = None,
    overwrite: bool = False,
):
    """Render the visualisation to a self-contained html page

    The page bundles all javascript so works offline

    Example inputs: https://gist.github.com/RobinL/73f7101cb2e1145ce8820bafdd15e988
    Args:
        nodes_with_clusters_pd (pd.DataFrame): A pandas dataframe of nodes with associated clusters.
            Optionally contains the ground truth cluster.
        edges_corresponding_to_clusters_pd (pd.DataFrame): A pandas dataframe of edges associated with the nodes,
            including the cluster
        splink_settings (dict): Splink settings as a dictionary e.g. from linker.model.current_settings_obj.settings_dict
        out_path (str): The path to which the output html file is written
        cluster_colname (str): The name of the cluster column, e.g. cluster_medium
        df_cluster_metrics (pd.DataFrame, optional): A dataframe of cluster metrics, output
            from splink_cluster_studio.graph.compute_cluster_metrics. Defaults to None.
        named_clusters (dict, optional): A dictionary that allows you to name clusters in the selection box in the html vis.
            e.g. {10: "John Smith low density", 15: "This cluster contains FP"}.  This replaces the values 10 and 15 with the strings
            for ease of use.
            Defaults to None.
        overwrite (bool, optional): Whether to overwrite the html file if it already exists. Defaults to False.
    """

    # When developing the package, it can be easier to point
    # ar the script live on observable using <script src=>
    # rather than bundling the whole thing into the html
    bundle_observable_notebook = True

    cols = list(edges_corresponding_to_clusters_pd.columns)
    if "tf_adjusted_match_prob" in cols:
        prob_col = "tf_adjusted_match_prob"
    else:
        prob_col = "match_probability"
    svu_options = {
        "cluster_colname": cluster_colname,
        "prob_colname": prob_col,
    }

    template_path = "jinja/cluster_template.j2"
    template = pkgutil.get_data(__name__, template_path).decode("utf-8")
    template = Template(template)

    template_data = {
        "raw_edge_data": edges_corresponding_to_clusters_pd.to_json(orient="records"),
        "raw_node_data": nodes_with_clusters_pd.to_json(orient="records"),
        "splink_settings": json.dumps(splink_settings),
        "svu_options": json.dumps(svu_options),
    }

    if df_cluster_metrics is not None:
        template_data["raw_clusters_data"] = df_cluster_metrics.to_json(
            orient="records"
        )

    if named_clusters is not None:
        template_data["named_clusters"] = json.dumps(named_clusters)

    files = {
        "embed": "vega-embed@6",
        "vega": "vega@5",
        "vegalite": "vega-lite@5",
        "svu_text": "splink_vis_utils.js",
    }
    for k, v in files.items():
        f = pkgutil.get_data(__name__, f"js_lib/{v}")
        f = f.decode("utf-8")
        template_data[k] = f

    files = {"custom_css": "custom.css"}
    for k, v in files.items():
        f = pkgutil.get_data(__name__, f"css/{v}")
        f = f.decode("utf-8")
        template_data[k] = f

    template_data["bundle_observable_notebook"] = bundle_observable_notebook

    rendered = template.render(**template_data)

    if os.path.isfile(out_path) and not overwrite:
        raise ValueError(
            f"The path {out_path} already exists. Please provide a different path."
        )
    else:
        with open(out_path, "w") as html_file:
            html_file.write(rendered)
