import torch.nn as nn


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
