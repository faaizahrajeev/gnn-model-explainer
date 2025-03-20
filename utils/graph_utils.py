"""graph_utils.py

   Utility for sampling graphs from a dataset.
"""
import networkx as nx
import numpy as np
import torch
import torch.utils.data


class GraphSampler(torch.utils.data.Dataset):
    """Samples graphs and nodes in a dataset for GNN training."""

    def __init__(self, G_list, features="default", normalize=True, assign_feat="default", max_num_nodes=0):
        self.adj_all = []
        self.len_all = []
        self.feature_all = []
        self.label_all = []
        self.assign_feat_all = []

        self.max_num_nodes = max_num_nodes or max(G.number_of_nodes() for G in G_list)

        existing_node = list(G_list[0].nodes())[0]
        self.feat_dim = len(G_list[0].nodes[existing_node]["feature"])  # Fix feature extraction

        for G in G_list:
            adj = np.array(nx.to_numpy_matrix(G))
            if normalize:
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
            self.adj_all.append(adj)
            self.len_all.append(G.number_of_nodes())
            self.label_all.append(G.graph.get("label", -1))  # Get label safely

            # Feature handling
            f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
            for i, u in enumerate(G.nodes()):
                f[i, :] = G.nodes[u]["feature"]
            self.feature_all.append(f)

            # Assign feature processing
            if assign_feat == "id":
                self.assign_feat_all.append(np.hstack((np.identity(self.max_num_nodes), self.feature_all[-1])))
            else:
                self.assign_feat_all.append(self.feature_all[-1])

        self.feat_dim = self.feature_all[0].shape[1]
        self.assign_feat_dim = self.assign_feat_all[0].shape[1]

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        return {
            "adj": adj_padded,
            "feats": self.feature_all[idx].copy(),
            "label": self.label_all[idx],
            "num_nodes": num_nodes,
            "assign_feats": self.assign_feat_all[idx].copy(),
        }


def neighborhoods(adj, n_hops, use_cuda):
    """Returns the n_hops degree adjacency matrix adj."""
    adj = torch.tensor(adj, dtype=torch.float)
    if use_cuda:
        adj = adj.cuda()
    hop_adj = power_adj = adj
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        prev_hop_adj = hop_adj
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
    return hop_adj.cpu().numpy().astype(int)