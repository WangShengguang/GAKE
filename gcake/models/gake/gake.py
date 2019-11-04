import torch
from torch import nn

from config import TorchConfig as Config


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
        if si_node.dtype == 'entity':
            si_emb = self.ent_embeddings(si_id)
        else:
            si_emb = self.rel_embedding(si_id)
        #
        ent_ids, rel_ids = [], []
        for node in si_context_nodes:
            if node.dtype == 'entity':
                ent_ids.append(node.id)
            else:
                rel_ids.append(node.id)
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

    def get_p(self, si_node, si_context_nodes):
        if len(si_context_nodes) == 0:
            return torch.zeros([1, 1]).to(Config.device)
        # s embedding
        si_emb, si_context_emb = self.get_embedding(si_node, si_context_nodes)
        context_pie = torch.sum(si_context_emb, dim=0) / torch.norm(si_context_emb, p=2)  # dim

        def cacu_exp(emb):
            # exp_res = torch.exp(torch.mm(emb, torch.transpose(context_pie.unsqueeze(0), 0, 1)))
            exp_res = torch.mm(emb, torch.transpose(context_pie.unsqueeze(0), 0, 1))
            # exp_res = torch.softmax(exp_res, dim=0)
            return exp_res

        p = cacu_exp(si_emb) / torch.sum(cacu_exp(si_context_emb))

        return p

    def forward(self, node, neighbor_nodes, path_nodes, edge_nodes):
        """
        :param node:  Node Object
        :param neighbors: List of Node Object
        :param paths:  List of Node Object
        :param edges:  List of Node Object
        :return:
        """
        _neighbors_p = self.get_p(node, neighbor_nodes)
        _paths_p = self.get_p(node, path_nodes)
        _edge_p = self.get_p(node, edge_nodes)
        # λT = 1, λP = 0.1 and λE = 0.1
        score = 1 * _neighbors_p + 0.1 * _paths_p + 0.1 * _edge_p
        loss = 1.2 - score  # 最小化loss,最大化score
        return score, loss
