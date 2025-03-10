import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import trange

from config import Config, ckpt_dir
from gcake.data_helper import DataHelper
from gcake.evaluator import Evaluator
from gcake.models.modules import Graph

logger = logging.getLogger('trainer')


class Saver(object):
    def __init__(self, dataset, model_name):
        self.dataset = dataset
        self.model_name = model_name
        # TODO: ckpt_dir should be managed by ckpt_manager.py rather than hard coded in config
        self.model_dir = os.path.join(ckpt_dir, dataset, model_name)

    def load_model(self, model, mode="max_step"):
        model_path = os.path.join(
            self.model_dir, mode, f"{self.model_name}.bin")
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt["net"])  # 断点续训
        step = ckpt["step"]
        return model_path, step

    def save(self, model, step, mode="max_step"):
        model_path = os.path.join(
            self.model_dir, mode, f"{self.model_name}.bin")
        state = {"net": model.state_dict(), "step": step}
        torch.save(state, model_path)
        return model_path


class BaseTrainer(object):
    def __init__(self):
        self.global_step = 0

    def get_optimizer(self, model):
        self.optimizer = optim.Adam(
            model.parameters(), lr=Config.learning_rate)
        # model, self.optimizer = amp.initialize(model, self.optimizer, opt_level="O1")  # 这里是“欧一”，不是“零一”
        return model

    def backward(self, loss, model):
        if Config.gpu_nums > 1 and Config.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu
        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps
        # https://zhuanlan.zhihu.com/p/79887894
        loss.backward()
        if self.global_step % Config.gradient_accumulation_steps == 0:
            # gradient clipping
            nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=Config.clip_grad)
            # performs updates using calculated gradients
            self.optimizer.step()
            # clear previous gradients
            self.optimizer.zero_grad()
        return loss


class Trainer(BaseTrainer):
    def __init__(self, model, args, configs):
        super().__init__()
        self.model = model
        self.get_optimizer(model)

        self.model_name = args.model
        self.dataset = args.dataset
        self.min_num_epoch = configs.min_epoch_nums

        self.data_helper = DataHelper(self.dataset)
        self.saver = Saver(self.dataset, self.model_name)
        self.evaluator = Evaluator(
            self.dataset, self.model_name, load_model=False)

        self.min_loss = 100.0
        self.best_val_mrr = 0.0
        self.patience_counter = 0

    def check_loss_save(self, model, global_step, loss):
        if loss <= self.min_loss:
            self.saver.save(model, global_step, mode="min_loss")
            self.min_loss = loss
        else:
            self.patience_counter += 1

    def check_save_mrr(self, model, global_step):
        self.evaluator.set_model(model=model)
        mr, mrr, hit_10, hit_3, hit_1 = self.evaluator.test_link_prediction(
            data_type="valid", _tqdm=len(self.data_helper.entity2id) > 1000)
        # rank_metrics = "\n*model:{} {} valid, mrr:{:.3f}, mr:{:.3f}, hit_10:{:.3f}, hit_3:{:.3f}, hit_1:{:.3f}\n".format(
        #     self.model_name, self.data_set, mrr, mr, hit_10, hit_3, hit_1)
        if mrr >= self.best_val_mrr:
            ckpt_path = self.saver.save(model, global_step, mode="max_mrr")
            # logger.info(rank_metrics)
            # print(rank_metrics)
            logger.info(
                "get best mrr: {:.3f}, save to : {}".format(mrr, ckpt_path))
            self.best_val_mrr = mrr
        else:
            self.patience_counter += 1
        return mr, mrr, hit_10, hit_3, hit_1

    def run(self, mode):
        logger.info("{} {} start train ...".format(
            self.model_name, self.dataset))

        model = self.model

        if Config.load_pretrain and False:  # 断点续训
            model_path, global_step = self.saver.load_model(
                model, mode=Config.load_model_mode)
            print("* Model load from file: {}".format(model_path))
        else:
            global_step = 0
        per_epoch_step = len(self.data_helper.get_data("train")[
                             0]) // Config.batch_size // 2  # 正负样本
        start_epoch_num = global_step // per_epoch_step  # 已经训练过多少epoch
        print("start_epoch_num: {}".format(start_epoch_num)) # TODO: use logger
        for epoch_num in trange(1, min(self.min_num_epoch, Config.max_epoch_nums) + 1,
                                desc="{} {} train epoch ".format(self.model_name, self.dataset)):
            if epoch_num <= start_epoch_num:
                continue
            losses = []
            for triples, sentences, y_labels in self.data_helper.batch_iter(data_type="train",
                                                                            batch_size=Config.batch_size,
                                                                            _shuffle=True):
                loss = model(triples, sentences, y_labels)
                # if global_step % Config.check_step == 0: # train step metrics
                #     self.evaluator.set_model(sess, model)
                #     metrics = self.evaluator.evaluate_metrics(x_batch.tolist(), _tqdm=False)
                #     mr, mrr = metrics["ave"]["MR"], metrics["ave"]["MRR"]
                #     hit_10, hit_3, hit_1 = metrics["ave"]["Hit@10"], metrics["ave"]["Hit@3"], metrics["ave"][
                #         "Hit@1"]
                #     logger.info("{} {} train epoch_num: {}, global_step: {}, loss: {:.3f}, "
                #                  "mr: {:.3f}, mrr: {:.3f}, Hit@10: {:.3f}, Hit@3: {:.3f}, Hit@1: {:.3f}".format(
                #         self.data_set, self.model_name, epoch_num, global_step, loss,
                #         mr, mrr, hit_10, hit_3, hit_1))
                # logger.info(" step:{}, loss: {:.3f}".format(global_step, loss))
                # predict = sess.run(model.predict, feed_dict={model.input_x: x_batch, model.input_y: y_batch})
                losses.append(loss.item())
                self.backward(loss, model)
                global_step += 1
                print(f"global_step: {global_step}, loss:{loss.item():.4f}")  # TODO: use logger
            self.check_loss_save(model, global_step, loss)
            # if epoch_num > self.min_num_epoch:
            mr, mrr, hit_10, hit_3, hit_1 = self.check_save_mrr(
                model, global_step)
            logger.info("valid epoch_num: {}, global_step: {}, loss: {:.3f}, "
                         "mr: {:.3f}, mrr: {:.3f}, hit_10: {:.3f}, hit_3: {:.3f}, hit_1: {:.3f}".format(
                             epoch_num, global_step, np.mean(losses),
                             mr, mrr, hit_10, hit_3, hit_1))
            self.saver.save(model, global_step, mode="max_step")
            logger.info(
                "epoch {} end  ...\n------------------------------\n\n".format(epoch_num))
            # Early stopping and logger best f1
            if self.patience_counter >= Config.patience_num and epoch_num > self.min_num_epoch:
                logger.info("Best val f1: {:.3f} best loss:{:.3f}".format(
                             self.best_val_mrr, self.min_loss))
                break


class GraphTrainer(Trainer):
    """ only used for the graph sub-model test purpose """

    def __init__(self, dataset, model_name, min_num_epoch=Config.min_epoch_nums):
        super().__init__(dataset, model_name, min_num_epoch)
        self.dataset = dataset

    def run(self, mode, device=Config.device):
        model = self.get_model()
        triples, sentences = self.data_helper.get_all_datas()
        graph = Graph(triples=triples)

        for i, entity_id in enumerate(graph.entities):
            # data
            _neighbor_ids = graph.get_neighbor_context(entity_id)
            _path_ids = graph.get_path_context(entity_id)
            _edge_ids = graph.get_edge_context(entity_id)
            #
            entity_id = torch.tensor([entity_id], dtype=torch.long).to(device)
            neighbor_ids = torch.tensor(
                _neighbor_ids, dtype=torch.long).to(device)
            path_ids = torch.tensor(_path_ids, dtype=torch.long).to(device)
            edge_ids = torch.tensor(_edge_ids, dtype=torch.long).to(device)

            global_weight_p, loss = model(
                entity_id, neighbor_ids, path_ids, edge_ids)
            self.backward(loss, model)
            print(
                f"* i:{i},\tentity:{entity_id.item()},\tloss:{loss.item():.4f}")
