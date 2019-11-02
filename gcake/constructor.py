# Model constructor

import logging
from gcake.data_helper import DataHelper

logger = logging.getLogger('constructor')


class Constructor(object):
    def __init__(self, args, configs):
        self.model_name = args.model
        self.sentence_encoder_name = configs.sentence_encoder
        self.graph_encoder_name = configs.graph_encoder

        # TODO: should not depend on this (i.e. construct graph outside the model)
        self.data_helper = DataHelper(args.dataset)

        self.args = args
        self.configs = configs

    def _get_sentence_encoder(self, sent_len):
        """ get setence encoder model """
        assert self.sentence_encoder_name in self.configs.all_sentence_encoder

        if self.sentence_encoder_name == 'Transformer':
            from gcake.model.sentence_encoder.Transformer import TransformerSentenceEncoder
            return TransformerSentenceEncoder(d_model=self.configs.embedding_dim, nhead=4, dim_feedforward=2048, num_layers=3, sent_len=sent_len)

    def _get_graph_encoder(self, triples, num_entity, num_relation, embedding_dim):
        """ get graph encoder model """
        assert self.graph_encoder_name in self.configs.all_graph_encoder

        if self.graph_encoder_name == 'GAKE':
            from gcake.model.graph_encoder.Transformer import GAKEGraphEncoder
            return GAKEGraphEncoder(triples, num_entity, num_relation, embedding_dim)

    def _warp_gpu(self, model):
        """ send the model to device and warp the model if use multiple GPU """
        # TODO: device should be passed through args
        model.to(self.configs.device)
        if self.configs.gpu_nums > 1 and self.configs.multi_gpu:
            model = nn.DataParallel(model)

    def get_model(self):
        """ construct model """
        logger.info('Constructing model...')

        # TODO: not sure what's the purpos of this, isn't it different kind of model?!
        from gcake.models.gake import GAKE
        from gcake.model.GCAKE import GCAKE

        # TODO: this should be pass from outside
        num_entity = len(self.data_helper.entity2id)  # TODO  len() != max()
        num_relation = len(self.data_helper.relation2id)
        num_word = len(self.data_helper.word2id)

        sent_len = self.configs.max_len
        embedding_dim = self.configs.embedding_dim

        if self.model_name == "GCAKE":
            # TODO: should not depend on this (i.e. construct graph outside the model)
            triples, sentences = self.data_helper.get_data("train")

            sentence_encoder = self._get_sentence_encoder(sent_len)
            graph_encoder = self._get_graph_encoder(
                triples, num_entity, num_relation, embedding_dim)

            model = GCAKE(sentence_encoder, graph_encoder,
                          num_entity, num_relation,
                          total_word=num_word, sent_len=self.configs.max_len,
                          dim=embedding_dim)

        else:  # TODO: not sure what's the purpos of this, isn't it different kind of model?!
            Model = {"GAKE": GAKE}[self.model_name]
            model = Model(num_entity, num_relation, dim=embedding_dim)
        self._warp_gpu(model)
        return model
