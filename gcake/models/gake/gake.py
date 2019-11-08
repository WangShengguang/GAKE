import numpy as np
import torch
from torch import nn

from config import TorchConfig as Config
from gcake.model.modules import Graph


class GakeModel(nn.Module):
    def __init__(self, all_triples, entity_num, relation_num, dim=128):
        super().__init__()
        self.graph = Graph(all_triples)
        self.gake = GAKE(entity_num, relation_num, dim)

    def split_pos_neg(self, all_encoded, y_labels=None):
        positives = all_encoded
        negatives = None
        if y_labels is not None:
            positives = all_encoded[torch.where(y_labels > 0.999)[0]]
            negatives = all_encoded[torch.where(y_labels < 0.0001)[0]]
        return positives, negatives

    def forward(self, hrts, y_labels=None):
        preds = []
        loss = torch.zeros([1, 1]).to(Config.device)
        positives, negatives = self.split_pos_neg(hrts)
        for h, r, t in list(positives):
            _preds = []
            _losses = []
            for node_id in [h, r, t]:
                node = self.graph.get_node(node_id)
                neighbour_nodes = self.graph.get_neighbor_context(node)
                path_nodes = self.graph.get_path_context(node)
                edge_nodes = self.graph.get_edge_context(node)
                _loss = self.gake(node, neighbour_nodes, path_nodes, edge_nodes)
                _preds.append(_loss.item())
                loss += _loss  # loss越小越好
            preds.append(np.mean(_preds))
        if y_labels is None:
            return 10 / np.asarray(preds)
        else:
            return 10 / np.asarray(preds), loss / len(hrts)


class GAKE(nn.Module):
    """
    https://www.aclweb.org/anthology/C16-1062.pdf
    """

    def __init__(self, entity_num, relation_num, dim=128):
        super().__init__()
        self.entity_num = entity_num
        self.ent_embeddings = nn.Embedding(entity_num, dim)
        self.rel_embeddings = nn.Embedding(relation_num, dim)

    def set_ent_embeddings(self, ent_embeddings: nn.Embedding, rel_embeddings):  # TODO: only pass weight
        self.ent_embeddings = ent_embeddings
        self.rel_embeddings = rel_embeddings

    def get_embedding(self, si_node, si_context_nodes):
        si_id = torch.tensor([si_node.id]).to(Config.device)
        if si_node.dtype == 'relation':
            si_emb = self.rel_embeddings(si_id)
        else:
            si_emb = self.ent_embeddings(si_id)
        #
        ent_ids, rel_ids = [], []
        for node in si_context_nodes:
            if node.dtype == 'relation':
                rel_ids.append(node.id)
            else:
                ent_ids.append(node.id)
        if rel_ids:
            _rel_ids = torch.tensor(rel_ids).to(Config.device)
            rel_ids_emb = self.rel_embeddings(_rel_ids)
        else:
            rel_ids_emb = torch.zeros([1, 1]).to(Config.device)

        if ent_ids:
            _ent_ids = torch.tensor(ent_ids).to(Config.device)
            ent_ids_emb = self.ent_embeddings(_ent_ids)
        else:
            ent_ids_emb = torch.zeros([1, 1]).to(Config.device)

        if ent_ids and rel_ids:
            si_context_emb = torch.cat([ent_ids_emb, rel_ids_emb])
        elif ent_ids:
            si_context_emb = ent_ids_emb
        else:
            si_context_emb = rel_ids_emb
        return si_emb, si_context_emb

    def get_log_p(self, si_node, si_context_nodes, context_accumulate=False):
        if len(si_context_nodes) == 0:
            return torch.zeros([1, 1]).to(Config.device)

        def cacu_exp(emb):
            # exp_res = torch.exp(torch.mm(emb, torch.transpose(context_pie.unsqueeze(0), 0, 1)))
            exp_res = torch.mm(emb, torch.transpose(context_pie.unsqueeze(0), 0, 1))
            # exp_res = torch.softmax(exp_res, dim=0)
            return exp_res

        # s embedding
        si_emb, si_context_emb = self.get_embedding(si_node, si_context_nodes)
        si_context_emb = torch.cat([si_emb, si_context_emb], dim=0)

        context_pie = torch.sum(si_context_emb, dim=0) / torch.norm(si_context_emb, p=2)  # dim

        p = torch.softmax(cacu_exp(si_context_emb), dim=0)  # 有可能 si_context_emb 和为负，则p>1
        log_p = torch.log(p)

        if context_accumulate:
            res = torch.sum(log_p)
        else:
            res = log_p[0]

        return res

    def forward(self, node, neighbor_nodes, path_nodes, edge_nodes):
        """
        :param node:  Node Object
        :param neighbors: List of Node Object
        :param paths:  List of Node Object
        :param edges:  List of Node Object
        :return:
        """
        _neighbors_p = self.get_log_p(node, neighbor_nodes, context_accumulate=True)
        _paths_p = self.get_log_p(node, path_nodes, context_accumulate=True)
        _edge_p = self.get_log_p(node, edge_nodes)
        # λT = 1, λP = 0.1 and λE = 0.1
        score = 1 * _neighbors_p + 0.1 * _paths_p + 0.1 * _edge_p
        loss = - score  # 最小化loss,最大化score

        return loss
