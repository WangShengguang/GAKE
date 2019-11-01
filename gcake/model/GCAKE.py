from typing import List

import torch
from torch import nn
from torch.nn import TransformerEncoder

from config import Config
from gcake.models.gake import GAKE
from gcake.models.modules import Graph


class SentenceEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, sent_len, dropout=0.1, activation="relu"):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation=activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, norm=encoder_norm)
        self.linear = nn.Linear(sent_len, 3)

    def forward(self, sents_embed):
        sents_feature = self.encoder(sents_embed)
        sents_feature = self.linear(sents_feature.permute(0, 2, 1)).permute(0, 2, 1)
        return sents_feature


class TriplesEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1, activation="relu"):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)
        # self.linear = nn.Linear()

    def forward(self, triples_embedding, sents_encoded):
        triples_feature = self.encoder(triples_embedding, sents_encoded)
        return triples_feature


class GraphEncoder(nn.Module):
    def __init__(self, triples, num_entity, num_relation, dim, ent_embeddings: nn.Embedding):
        super().__init__()
        self.graph = Graph(triples)
        self.gake = GAKE(num_entity, num_relation, dim, ent_embeddings)

    def forward(self, htrs, device=Config.device):
        loss = torch.zeros(1)
        for h, r, t in htrs:
            for entity_id in (int(h), int(t)):
                _neighbor_ids = self.graph.get_neighbor_context(entity_id)
                _path_ids = self.graph.get_path_context(entity_id)
                _edge_ids = self.graph.get_edge_context(entity_id)
                #
                entity_id = torch.tensor([entity_id], dtype=torch.long).to(device)
                neighbor_ids = torch.tensor(_neighbor_ids, dtype=torch.long).to(device)
                path_ids = torch.tensor(_path_ids, dtype=torch.long).to(device)
                edge_ids = torch.tensor(_edge_ids, dtype=torch.long).to(device)
                global_weight_p, _loss = self.gake(entity_id, neighbor_ids, path_ids, edge_ids)
                loss += _loss
        return loss


class GCAKE(nn.Module):
    def __init__(self, num_entity: int, num_relation: int, total_word: int,
                 dim: int, sent_len: int = Config.max_len,
                 triples: List = ()):
        super(GCAKE, self).__init__()

        self.ent_embedding = nn.Embedding(num_entity, dim)
        self.rel_embedding = nn.Embedding(num_relation, dim)
        self.word_embedding = nn.Embedding(total_word, dim)
        # embed_dim must be divisible by num_heads
        self.triples_encoder = TriplesEncoder(d_model=dim, nhead=4, dim_feedforward=2048, num_layers=3)
        self.sent_encoder = SentenceEncoder(d_model=dim, nhead=4, dim_feedforward=2048, num_layers=3, sent_len=sent_len)
        self.second_triple_encoder = nn.TransformerDecoderLayer(d_model=dim, nhead=4, dim_feedforward=2048)
        self.graph_encoder = GraphEncoder(triples, num_entity, num_relation, dim, ent_embeddings=self.ent_embedding)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=3 * dim, out_features=1),
            # nn.ReLU(),
            # nn.Linear(in_features=dim, out_features=1),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(dim)

    def _lookup(self, hrts, sentences=None):
        """ look up embedding """
        h, r, t = hrts[:, 0], hrts[:, 1], hrts[:, 2]
        h_embedding = self.ent_embedding(h)
        r_embedding = self.rel_embedding(r)
        t_embedding = self.ent_embedding(t)
        sent_embedding = self.word_embedding(sentences) if sentences is not None else None
        return h_embedding, r_embedding, t_embedding, sent_embedding

    def forward(self, hrts, sentences, y_labels=None):
        """ train """
        h_embed, r_embed, t_embed, sentences_embed = self._lookup(hrts, sentences)
        triples_embed = torch.stack((h_embed, r_embed, t_embed), dim=1)
        sent_encoded = self.sent_encoder(sentences_embed)
        triples_encoded = self.triples_encoder(triples_embed, sent_encoded)
        pred = self.classifier(triples_encoded.view(triples_encoded.shape[0], -1))
        if y_labels is None:
            return pred
        else:
            graph_loss = self.graph_encoder(hrts)  # graph structure is prepared in data helper
            loss = torch.sum(y_labels * pred) + graph_loss
            return loss

    def predict(self, h, r, t, sent):
        """ test """
        pass
