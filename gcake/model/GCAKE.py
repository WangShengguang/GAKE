import torch
from torch import nn


class TriplesEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1, activation="relu"):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerDecoder(
            decoder_layer, num_layers, decoder_norm)

    def forward(self, triples_embedding, sents_encoded):
        triples_feature = self.encoder(triples_embedding, sents_encoded)
        return triples_feature


class GCAKE(nn.Module):
    def __init__(self, sent_encoder, graph_encoder,
                 num_entity: int, num_relation: int, total_word: int,
                 dim: int, sent_len: int):
        super(GCAKE, self).__init__()

        self.ent_embedding = nn.Embedding(num_entity, dim)
        self.rel_embedding = nn.Embedding(num_relation, dim)
        self.word_embedding = nn.Embedding(total_word, dim)
        # embed_dim must be divisible by num_heads
        self.triples_encoder = TriplesEncoder(
            d_model=dim, nhead=4, dim_feedforward=2048, num_layers=3)
        self.sent_encoder = sent_encoder
        self.second_triple_encoder = nn.TransformerDecoderLayer(
            d_model=dim, nhead=4, dim_feedforward=2048)  # TODO: connect graph eoncoder to it
        self.graph_encoder = graph_encoder
        self.graph_encoder.set_ent_embeddings(self.ent_embedding, self.rel_embedding)

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
        sent_embedding = self.word_embedding(
            sentences) if sentences is not None else None
        return h_embedding, r_embedding, t_embedding, sent_embedding

    def split_hrts(self, hrts, y_labels=None):
        positive_hrts = hrts
        negative_hrts = []
        if y_labels is not None:
            positive_hrts = hrts[torch.where(y_labels > 0.999)[0]]
            negative_hrts = hrts[torch.where(y_labels < 0.0001)[0]]
        return positive_hrts, negative_hrts

    def forward(self, hrts, sentences, y_labels=None):
        """ train """
        h_embed, r_embed, t_embed, sentences_embed = self._lookup(
            hrts, sentences)
        triples_embed = torch.stack((h_embed, r_embed, t_embed), dim=1)

        sent_encoded = self.sent_encoder(sentences_embed)
        triples_encoded = self.triples_encoder(triples_embed, sent_encoded)

        # TODO: didn't connect graph embedding?!

        pred = self.classifier(triples_encoded.view(
            triples_encoded.shape[0], -1))

        if y_labels is None:
            return pred
        else:
            # graph structure is prepared in data helper (NOT NOW...)
            positive_hrts, negative_hrts = self.split_hrts(hrts, y_labels)
            preds, graph_loss = self.graph_encoder(positive_hrts, is_train=True)
            loss = torch.sum(y_labels * pred) + graph_loss
            return loss

    def predict(self, h, r, t, sent):
        """ test """
        pass
