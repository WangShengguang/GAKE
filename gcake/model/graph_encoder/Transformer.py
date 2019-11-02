import torch
import torch.nn as nn
import torch.nn.functional as F

from gcake.models.gake import GAKE
from gcake.models.modules import Graph

from config import Config  # TODO: remove device


class GAKEGraphEncoder(nn.Module):
    def __init__(self, triples, num_entity, num_relation, dim):
        super().__init__()
        self.graph = Graph(triples)
        self.gake = GAKE(num_entity, num_relation, dim)

    def set_ent_embeddings(self, ent_embeddings: nn.Embedding):  # TODO: only pass wieght
        self.gake.set_ent_embeddings(ent_embeddings)

    def forward(self, htrs, device=Config.device):  # TODO: remove device
        loss = torch.zeros(1)
        for h, r, t in htrs:
            for entity_id in (int(h), int(t)):
                _neighbor_ids = self.graph.get_neighbor_context(entity_id)
                _path_ids = self.graph.get_path_context(entity_id)
                _edge_ids = self.graph.get_edge_context(entity_id)
                #
                entity_id = torch.tensor(
                    [entity_id], dtype=torch.long).to(device)
                neighbor_ids = torch.tensor(
                    _neighbor_ids, dtype=torch.long).to(device)
                path_ids = torch.tensor(_path_ids, dtype=torch.long).to(device)
                edge_ids = torch.tensor(_edge_ids, dtype=torch.long).to(device)
                global_weight_p, _loss = self.gake(
                    entity_id, neighbor_ids, path_ids, edge_ids)
                loss += _loss
        return loss
