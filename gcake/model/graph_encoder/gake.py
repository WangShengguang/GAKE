import torch
import torch.nn as nn

from config import Config  # TODO: remove device
from gcake.models.gake import GAKE
import numpy as np


class GAKEGraphEncoder(nn.Module):
    def __init__(self, graph, num_entity, num_relation, dim):
        super().__init__()
        self.graph = graph
        self.gake = GAKE(num_entity, num_relation, dim)

    def set_ent_embeddings(self, ent_embeddings: nn.Embedding, rel_embedding):  # TODO: only pass wieght
        self.gake.set_ent_embeddings(ent_embeddings, rel_embedding)

    def forward(self, htrs, is_train=False, device=Config.device):  # TODO: remove device
        loss = torch.zeros([1, 1]).to(device)
        preds = []
        for h, r, t in htrs:
            for entity_id in (int(h), int(t)):
                _neighbor_nodes = self.graph.get_neighbor_context(entity_id)
                _path_nodes = self.graph.get_path_context(entity_id)
                _edge_nodes = self.graph.get_edge_context(entity_id)
                #
                pred, _loss = self.gake(self.graph.entities[entity_id],
                                        _neighbor_nodes, _path_nodes, _edge_nodes)
                preds.append(pred)
                loss += _loss
            if device == 'gpu':
                torch.cuda.empty_cache()
        if is_train:
            return preds, loss
        else:
            return np.asarray(preds)
