import torch
import torch.nn as nn

from config import Config
from .modules import Graph


class TriplesAttention(nn.Module):
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


class TriplesEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1, activation="relu"):
        super(TriplesEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation=activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers, norm=encoder_norm)

    def forward(self, triples_embed):
        sents_feature = self.encoder(triples_embed)
        return sents_feature


class TransformerSentenceEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, sent_len, dropout=0.1, activation="relu"):
        super(TransformerSentenceEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation=activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers, norm=encoder_norm)
        self.linear = nn.Linear(sent_len, 3)

    def forward(self, sents_embed):
        sents_feature = self.encoder(sents_embed)
        sents_feature = self.linear(
            sents_feature.permute(0, 2, 1)).permute(0, 2, 1)
        return sents_feature


class GraphAttention(nn.Module):
    """
    https://www.aclweb.org/anthology/C16-1062.pdf
    """

    def __init__(self, all_triples, d_model, ent_embeddings, rel_embeddings):
        """
        :param all_triples:
        :param ent_embeddings: nn.Embedding
        :param rel_embeddings: nn.Embedding
        """
        super().__init__()
        self.graph = Graph(all_triples)
        self.ent_embeddings = ent_embeddings
        self.rel_embeddings = rel_embeddings
        self.layer_norm = nn.LayerNorm(d_model)

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

    def attention(self, node_emed, neighbours_embed):
        att = torch.mm(neighbours_embed, torch.transpose(node_emed, 0, 1))
        attention_out = torch.mm(torch.transpose(att / 15, 0, 1), neighbours_embed)
        return self.layer_norm(node_emed + attention_out)

    def forward(self, hrts):
        hrts_li = []
        for h, r, t in hrts:
            triple_att = []
            for node_id in [h, r, t]:
                node = self.graph.get_node(node_id)
                neighbour_nodes = self.graph.get_context(node, path_len=5)
                node_embed, neighbours_embed = self.get_embedding(node, neighbour_nodes)
                attention_out = self.attention(node_embed, neighbours_embed)
                triple_att.append(attention_out.squeeze())
            hrts_li.append(torch.stack(triple_att))
        return torch.stack(hrts_li)
