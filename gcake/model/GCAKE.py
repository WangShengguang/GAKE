import torch
from torch import nn

from config import Config
from .models import GraphAttention
from .models import TriplesEncoder, TriplesAttention, TransformerSentenceEncoder


class GCAKE(nn.Module):
    def __init__(self, all_triples,
                 num_entity: int, num_relation: int, total_word: int,
                 dim: int, sent_len: int, use_graph_encoder=True):
        super(GCAKE, self).__init__()
        self.ent_embedding = nn.Embedding(num_entity, dim)
        self.rel_embedding = nn.Embedding(num_relation, dim)
        self.word_embedding = nn.Embedding(total_word, dim)
        # embed_dim must be divisible by num_heads
        self.triples_encoder = TriplesEncoder(
            d_model=dim, nhead=4, dim_feedforward=2048, num_layers=3)
        self.triples_attn = TriplesAttention(
            d_model=dim, nhead=4, dim_feedforward=2048, num_layers=3)
        self.sent_encoder = TransformerSentenceEncoder(
            d_model=dim, nhead=4, dim_feedforward=2048, num_layers=3, sent_len=sent_len)

        # self.criterion = nn.BCELoss()
        margin = 1
        self.criterion = nn.MarginRankingLoss(margin, False)
        self.neg_y_label = torch.tensor([-1]).to(Config.device)

        # self.criterion = nn.Softplus()
        # torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()  # Binary Cross Entropy
        self.soft_plus = nn.Softplus()
        self.l2_reg_lambda = 0.001
        self.use_graph = use_graph_encoder
        self.parms_init(all_triples, dim, use_graph=use_graph_encoder)

    def parms_init(self, all_triples, dim, use_graph=True):
        if use_graph:
            self.graph_encoder = GraphAttention(all_triples, dim, self.ent_embedding, self.rel_embedding)
            dim *= 2  # torch.cat([triples,graph],dim=0)
        self.classifier = nn.Linear(in_features=3 * dim, out_features=1)
        self.layer_norm = nn.LayerNorm(dim)

    def _lookup(self, hrts, sentences=None):
        """ look up embedding """
        if isinstance(hrts, list):
            hrts = torch.tensor(hrts, dtype=torch.long).to(Config.device)
        h, r, t = hrts[:, 0], hrts[:, 1], hrts[:, 2]
        h_embedding = self.ent_embedding(h)
        r_embedding = self.rel_embedding(r)
        t_embedding = self.ent_embedding(t)
        sent_embedding = self.word_embedding(
            sentences) if sentences is not None else None
        return h_embedding, r_embedding, t_embedding, sent_embedding

    def split_pos_neg(self, all_encoded, y_labels=None):
        positives = all_encoded
        negatives = None
        if y_labels is not None:
            positives = all_encoded[torch.where(y_labels > 0.999)[0]]
            negatives = all_encoded[torch.where(y_labels < 0.0001)[0]]
        return positives, negatives

    def _forward_without_graph(self, hrts, sentences=None, y_labels=None):
        """ train """
        h_embed, r_embed, t_embed, sentences_embed = self._lookup(
            hrts, sentences)
        triples_embed = torch.stack((h_embed, r_embed, t_embed), dim=1)

        triples_encoded = self.triples_encoder(triples_embed)

        if sentences is not None:
            sent_encoded = self.sent_encoder(sentences_embed)
            triples_encoded = self.triples_attn(triples_encoded, sent_encoded)

        positives_encoded, negatives_encoded = self.split_pos_neg(triples_encoded, y_labels)

        p_score = self.classifier(positives_encoded.view(positives_encoded.shape[0], -1))

        if y_labels is None:
            return p_score
        else:
            # loss = torch.sum(y_labels * pred)
            n_score = self.classifier(negatives_encoded.view(negatives_encoded.shape[0], -1))
            loss = self.criterion(p_score, n_score, self.neg_y_label)
            return p_score, loss

    def _forward_bce(self, hrts, sentences=None, y_labels=None):
        """ train """
        h_embed, r_embed, t_embed, sentences_embed = self._lookup(
            hrts, sentences)
        triples_embed = torch.stack((h_embed, r_embed, t_embed), dim=1)
        triples_encoded = self.triples_encoder(triples_embed)
        if sentences is not None:
            sent_encoded = self.sent_encoder(sentences_embed)
            triples_encoded = self.triples_encoder(triples_encoded, sent_encoded)
        score = self.classifier(triples_encoded.view(triples_encoded.shape[0], -1))
        pred = self.sigmoid(score)

        if y_labels is None:
            return pred
        else:
            # loss = torch.sum(y_labels * pred)
            y_labels[torch.where(y_labels < 0)] = 0
            loss = self.bce_loss(pred, y_labels)
            return pred, loss

    def _forward(self, hrts, sentences=None, y_labels=None):
        """ train """
        h_embed, r_embed, t_embed, sentences_embed = self._lookup(
            hrts, sentences)
        triples_embed = torch.stack((h_embed, r_embed, t_embed), dim=1)
        triples_encoded = self.triples_encoder(triples_embed)
        if sentences is not None:
            sent_encoded = self.sent_encoder(sentences_embed)
            triples_encoded = self.triples_attn(triples_encoded, sent_encoded)

        graph_encoded = self.graph_encoder(hrts)
        triples_graph_encoded = self.triples_attn(triples_encoded, graph_encoded)

        all_encoded = torch.cat([triples_encoded, triples_graph_encoded], dim=-1)

        # all_encoded = self.layer_norm(all_encoded)
        score = self.classifier(all_encoded.view(all_encoded.shape[0], -1))
        pred = self.sigmoid(score)
        if y_labels is None:
            return pred
        else:
            losses = self.soft_plus(score * y_labels)
            loss = (torch.mean(losses) +
                    self.l2_reg_lambda * (torch.norm(self.classifier.weight) + torch.norm(self.classifier.bias)))
            return pred, loss

    def forward(self, hrts, sentences=None, y_labels=None):
        if self.use_graph:
            return self._forward(hrts, sentences, y_labels=y_labels)
        else:
            return self._forward_without_graph(hrts, sentences, y_labels=y_labels)
