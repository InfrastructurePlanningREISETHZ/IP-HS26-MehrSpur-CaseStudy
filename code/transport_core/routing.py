from __future__ import annotations

import numpy as np
import pandas as pd
from pandana import Network
from pandana.cyaccess import cyaccess
from sklearn.neighbors import KDTree


class RoutingNetwork(Network):
    """Pandana network with explicit 32-bit internal node indices."""

    def __init__(self, node_x, node_y, edge_from, edge_to, edge_weights, twoway=True):
        nodes_df = pd.DataFrame({"x": node_x, "y": node_y})
        edges_df = pd.DataFrame({"from": edge_from, "to": edge_to}).join(edge_weights)

        self.nodes_df = nodes_df
        self.edges_df = edges_df
        self.impedance_names = list(edge_weights.columns)
        self.variable_names = set()
        self.poi_category_names = []
        self.poi_category_indexes = {}
        self.node_idx = pd.Series(
            np.arange(len(nodes_df), dtype=np.int32),
            index=nodes_df.index,
        )
        edges = pd.concat(
            [self._node_indexes(edges_df["from"]), self._node_indexes(edges_df["to"])],
            axis=1,
        ).astype(np.int32)
        self.net = cyaccess(
            self.node_idx.to_numpy(dtype=np.int32),
            nodes_df.astype("double").values,
            edges.to_numpy(dtype=np.int32),
            edges_df[edge_weights.columns].transpose().astype("double").values,
            twoway,
        )
        self._twoway = twoway
        self.kdtree = KDTree(nodes_df.values)

