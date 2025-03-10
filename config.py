import os

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = cur_dir
data_dir = os.path.join(root_dir, "data")

output_dir = os.path.join(root_dir, "output")
ckpt_dir = os.path.join(output_dir, "ckpt")


# torch.cuda.manual_seed(rand_seed) #gpu
# tf.random.set_random_seed(rand_seed)

class SubModelConfig(object):
    """ configuration of sub-model to construct """
    all_sentence_encoder = ['Transformer']
    sentence_encoder = 'Transformer'
    all_graph_encoder = ['GAKE']
    graph_encoder = 'GAKE'


class TorchConfig(object):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    gpu_nums = torch.cuda.device_count()
    multi_gpu = True
    gradient_accumulation_steps = 1
    clip_grad = 2


class Config(TorchConfig, SubModelConfig):
    load_pretrain = True
    rand_seed = 1234
    load_model_mode = "max_step"  # max_mrr,min_loss,max_step
    #
    learning_rate = 0.001
    #
    min_epoch_nums = 2
    max_epoch_nums = 10
    patience_num = 3
    #
    embedding_dim = 128  # entity enbedding dim, relation enbedding dim , word enbedding dim
    max_len = 50  # max sentence length
    batch_size = 16

    subtask = 'general'
    test_batch_size = 128
    lr = 0.0001


class DevConfig(Config):
    """ only used for development purpose """
    subtask = 'dev'
    batch_size = 64
    test_batch_size = 128
