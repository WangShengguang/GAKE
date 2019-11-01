import torch
from torch import nn
from config import TorchConfig as Config


class GAKE(nn.Module):
    """
    https://www.aclweb.org/anthology/C16-1062.pdf
    """

    def __init__(self, entity_num, relation_num, dim=128, ent_embeddings: nn.Embedding = None):
        super().__init__()
        self.entity_num = entity_num
        if ent_embeddings is None:
            ent_embeddings = nn.Embedding(entity_num + relation_num, dim)
        self.embeddings = ent_embeddings
        self.linear = nn.Linear(3, 1)

    def get_p(self, si_id, si_context_ids):
        if len(si_context_ids) == 0:
            return torch.zeros([1, 1]).to(Config.device)

        si_context_emb = self.embeddings(si_context_ids)
        context_pie = torch.sum(si_context_emb, dim=0) / torch.norm(si_context_emb, p=2)  # (n,dim)

        def cacu_exp(node_ids):
            nodes_emb = self.embeddings(node_ids)
            exp_res = torch.exp(torch.mm(nodes_emb, torch.transpose(context_pie.unsqueeze(0), 0, 1)))
            return exp_res

        p = cacu_exp(si_id) / torch.sum(cacu_exp(si_context_ids))
        return p

    def forward(self, entity_id, neighbor_ids, path_ids, edge_ids):
        _neighbors_p = self.get_p(entity_id, neighbor_ids)
        _paths_p = self.get_p(entity_id, path_ids)
        _edge_p = self.get_p(entity_id, edge_ids)
        global_weight_p = self.linear(torch.cat([_neighbors_p, _paths_p, _edge_p]).squeeze())  # 全局概率，最大化
        loss = 1 - global_weight_p
        return global_weight_p, loss

    def find_best_threshold(self, valid_datas):
        """
        :param valid_datas:
        :return:
        """
        pass
