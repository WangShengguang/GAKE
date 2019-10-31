import torch
import torch.nn as nn
import torch.nn.functional as F


class GCAKE(nn.Module):
    def __init__(self, graph_encoder: nn.Module,
                 sent_encoder: nn.Module,
                 total_ent: int, total_rel: int, total_word: int, dim: int, sent_len: int,
                 nhead=6):
        super(GCAKE, self).__init__()

        self.ent_embedding = nn.Embedding(total_ent, dim)
        self.rel_embedding = nn.Embedding(total_rel, dim)
        self.word_embedding = nn.Embedding(total_word, dim)

        self.first_triple_encoder = nn.TransformerDecoderLayer(
            d_module=dim, nhead=nhead, dim_feedforward=2048)
        self.sent_encoder = sent_encoder
        self.second_triple_encoder = nn.TransformerDecoderLayer(
            d_module=dim, nhead=nhead, dim_feedforward=2048)
        self.graph_encoder = graph_encoder

        self.classifier = nn.Sequential([
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid()
        ])

    def _lookup(self, h, r, t, sent=None):
        """ look up embedding """
        h = self.ent_embedding(h)
        r = self.rel_embedding(h)
        t = self.ent_embedding(h)
        if sent:
            sent = self.word_embedding(sent)
            return h, r, t, sent
        return h, r, t

    def forward(self, h, r, t, sent, graph):
        """ train """
        h, r, t, sent = self._lookup(h, r, t, sent)

        # TODO: Batchnorm, Dropout?!

        triple_embedding = torch.cat((h, r, t))
        sent_embedding = self.sent_encoder(sent)
        triple_sent_embedding = self.first_triple_encoder(
            triple_embedding, sent_embedding)
        graph_embedding = self.graph_encoder(graph)  # graph structure is prepared in data helper
        triple_sent_graph_embedding = self.first_triple_encoder(
            triple_sent_embedding, graph_embedding)

    def predict(self, h, r, t, sent):
        """ test """
        pass
