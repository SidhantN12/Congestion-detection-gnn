"""Small heterogeneous GNN: type-specific input encoders into a shared
latent space, N layers of relation-specific message passing (one
GraphSAGE operator per edge type, via PyG's HeteroConv), MLP head on
gcell nodes predicting 2 values (horizontal, vertical congestion ratio).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class CongestionGNN(nn.Module):
    def __init__(self, in_dims, edge_types, hidden_dim=64, num_layers=3):
        """
        in_dims: dict {node_type: input_feature_dim}
        edge_types: list of (src_type, relation, dst_type) tuples
        """
        super().__init__()
        self.node_types = list(in_dims.keys())

        self.encoders = nn.ModuleDict({
            ntype: nn.Linear(dim, hidden_dim) for ntype, dim in in_dims.items()
        })

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                et: SAGEConv(hidden_dim, hidden_dim) for et in edge_types
            }, aggr="sum")
            self.convs.append(conv)
            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(hidden_dim) for nt in self.node_types
            }))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data):
        x_dict = {
            ntype: F.relu(self.encoders[ntype](data[ntype].x))
            for ntype in self.node_types
        }
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

        for conv, norm in zip(self.convs, self.norms):
            update = conv(x_dict, edge_index_dict)
            x_dict = {
                nt: F.relu(norm[nt](update[nt] + x_dict[nt]))
                for nt in self.node_types
            }

        return self.head(x_dict["gcell"])
