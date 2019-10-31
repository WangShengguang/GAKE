import os
import random

import numpy as np
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = cur_dir
data_dir = os.path.join(root_dir, "data")

output_dir = os.path.join(root_dir, "output")
ckpt_dir = os.path.join(output_dir, "ckpt")

rand_seed = 1234
random.seed(rand_seed)
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)  # cpu


# torch.cuda.manual_seed(rand_seed) #gpu
# tf.random.set_random_seed(rand_seed)


class TorchConfig(object):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    gpu_nums = torch.cuda.device_count()
    multi_gpu = True
    gradient_accumulation_steps = 1
    clip_grad = 2


class Config(TorchConfig):
    load_pretrain = True
    load_model_mode = "max_step"  # max_mrr,min_loss,max_step
    #
    learning_rate = 0.001
    #
    min_epoch_nums = 2
    max_epoch_nums = 10
    patience_num = 3
    #
    max_len = 502
    batch_size = 16
