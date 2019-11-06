import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import trange, tqdm

from config import Config, ckpt_dir
from gcake.data_helper import DataHelper
from gcake.evaluator import Evaluator
from gcake.models.modules import Graph


class Saver(object):
    def __init__(self, dataset, model_name):
        self.dataset = dataset
        self.model_name = model_name
        self.model_dir = os.path.join(ckpt_dir, dataset, model_name)

    def load_model(self, model, mode="max_step", fail_ok=False):
        model_path = os.path.join(self.model_dir, mode, f"{self.model_name}.bin")
        if os.path.isfile(model_path):
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt["net"])  # 断点续训
            step = ckpt["step"]
            epoch = ckpt["epoch"]
        else:
            if fail_ok:
                epoch = 0
                step = 0
            else:
                raise ValueError(f'model path :{model_path} is not exist')
        return model_path, epoch, step

    def save(self, model, epoch, step=-1, mode="max_step", dic: dict = None):
        model_path = os.path.join(self.model_dir, mode, f"{self.model_name}.bin")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        state = {"net": model.state_dict(), 'epoch': epoch, "step": step}
        if isinstance(dic, dict):
            state.update(dic)
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
            # logging.info(rank_metrics)
            # print(rank_metrics)
            logging.info(
                "get best mrr: {:.3f}, save to : {}".format(mrr, ckpt_path))
            self.best_val_mrr = mrr
        else:
            self.patience_counter += 1
        return mr, mrr, hit_10, hit_3, hit_1

    def run(self, mode):
        logging.info("{} {} start {} ...".format(mode, self.model_name, self.dataset))

        model = self.model

        if mode != "train":
            self.evaluator.set_model(model=model)
            mr, mrr, hit_10, hit_3, hit_1 = self.evaluator.test_link_prediction(data_type=mode)
            _test_log = ("mr:{:.3f}, mrr:{:.3f}, hit_1:{:.3f}, hit_3:{:.3f}, hit_10:{:.3f}".format(
                mr, mrr, hit_1, hit_3, hit_10))
            logging.info(_test_log)
            print(_test_log)
            return

        if Config.load_pretrain and False:  # 断点续训
            model_path, epoch, global_step = self.saver.load_model(
                model, mode=Config.load_model_mode)
            print("* Model load from file: {}".format(model_path))
        else:
            global_step = 0
        per_epoch_step = len(self.data_helper.get_data("train")[0]) // Config.batch_size // 2  # 正负样本
        start_epoch_num = global_step // per_epoch_step  # 已经训练过多少epoch
        print("start_epoch_num: {}".format(start_epoch_num))

        for epoch_num in trange(1, min(self.min_num_epoch, Config.max_epoch_nums) + 1,
                                desc="{} {} train epoch ".format(self.model_name, self.dataset)):
            if epoch_num <= start_epoch_num:
                continue
            losses = []
            for triples, sentences, y_labels in self.data_helper.batch_iter(data_type="train",
                                                                            batch_size=Config.batch_size,
                                                                            _shuffle=True):
                pred, loss = model(triples, sentences, y_labels)
                # if global_step % Config.check_step == 0: # train step metrics
                #     self.evaluator.set_model(sess, model)
                #     metrics = self.evaluator.evaluate_metrics(x_batch.tolist(), _tqdm=False)
                #     mr, mrr = metrics["ave"]["MR"], metrics["ave"]["MRR"]
                #     hit_10, hit_3, hit_1 = metrics["ave"]["Hit@10"], metrics["ave"]["Hit@3"], metrics["ave"][
                #         "Hit@1"]
                #     logging.info("{} {} train epoch_num: {}, global_step: {}, loss: {:.3f}, "
                #                  "mr: {:.3f}, mrr: {:.3f}, Hit@10: {:.3f}, Hit@3: {:.3f}, Hit@1: {:.3f}".format(
                #         self.data_set, self.model_name, epoch_num, global_step, loss,
                #         mr, mrr, hit_10, hit_3, hit_1))
                # logging.info(" step:{}, loss: {:.3f}".format(global_step, loss))
                # predict = sess.run(model.predict, feed_dict={model.input_x: x_batch, model.input_y: y_batch})
                losses.append(loss.item())
                self.backward(loss, model)
                global_step += 1
                logging.info(f"epoch:{epoch_num}({per_epoch_step}), global_step: {global_step}, "
                             f"loss:{loss.item():.4f}")
            self.check_loss_save(model, global_step, loss)
            # if epoch_num > self.min_num_epoch:
            mr, mrr, hit_10, hit_3, hit_1 = self.check_save_mrr(
                model, global_step)
            logging.info("valid epoch_num: {}, global_step: {}, loss: {:.3f}, "
                         "mr: {:.3f}, mrr: {:.3f}, hit_10: {:.3f}, hit_3: {:.3f}, hit_1: {:.3f}".format(
                epoch_num, global_step, np.mean(losses),
                mr, mrr, hit_10, hit_3, hit_1))
            self.saver.save(model, global_step, mode="max_step")
            logging.info(
                "epoch {} end  ...\n------------------------------\n\n".format(epoch_num))
            # Early stopping and logging best f1
            if self.patience_counter >= Config.patience_num and epoch_num > self.min_num_epoch:
                logging.info("Best val f1: {:.3f} best loss:{:.3f}".format(
                    self.best_val_mrr, self.min_loss))
                break


class GraphTrainer(Trainer):
    """ only used for the graph sub-model test purpose """

    def __init__(self, model, args, configs):
        super().__init__(model, args, configs)
        self.dataset = args.dataset
        self.model = model
        from gcake.models.modules import Graph
        all_triples, sentences = self.data_helper.get_all_datas()
        self.graph = Graph(all_triples)

    def find_best_threshold(self, model, valid_triples, device=Config.device):
        model.eval()
        all_preds = []
        for i, (h, r, t) in enumerate(tqdm(valid_triples, desc="valid best threshold ")):
            for entity_id in (h, t):
                # data
                _neighbor_nodes = self.graph.get_neighbor_context(entity_id)
                _path_nodes = self.graph.get_path_context(entity_id)
                _edge_nodes = self.graph.get_edge_context(entity_id)
                #
                pred, loss = model(self.graph.entities[entity_id], _neighbor_nodes, _path_nodes, _edge_nodes)
                all_preds.append(pred.item())
                if i % 1000 == 0:
                    logging.info(f"* valid step: {i}/{len(valid_triples)},\t"
                                 f"entity:{entity_id},pred:{pred.item():.4f}, loss:{loss.item():.4f}")
                    print(f"* valid step: {i}/{len(valid_triples)},\t"
                          f"entity:{entity_id}, pred:{pred.item():.4f}, loss: {loss.item():.4f}, "
                          f"min/max={min(all_preds):.4f}/{max(all_preds):.4f}")
                    if device == 'gpu':
                        torch.cuda.empty_cache()
        all_preds = np.asarray(all_preds)
        best_threshold = max_correct_count = 0
        min_value, max_value = min(all_preds), max(all_preds)
        logging.info(f"min_value: {min_value:.4f}, max_value: {max_value:.4f}")
        print(f"min_value: {min_value:.4f}, max_value: {max_value:.4f}")
        if min_value < max_value:
            for threshold in tqdm(np.arange(min_value, max_value, (max_value - min_value) / 10),
                                  desc="threshold interval"):
                _correct_count = len(valid_triples) - sum(all_preds > threshold)
                if _correct_count > max_correct_count:
                    max_correct_count = _correct_count
                    best_threshold = threshold
        return max_correct_count, best_threshold

    def run(self, mode, device=Config.device):

        if mode != "train":
            self.evaluator.set_model(model=self.model)
            mr, mrr, hit_10, hit_3, hit_1 = self.evaluator.test_link_prediction(data_type=mode)
            _test_log = ("mr:{:.3f}, mrr:{:.3f}, hit_1:{:.3f}, hit_3:{:.3f}, hit_10:{:.3f}".format(
                mr, mrr, hit_1, hit_3, hit_10))
            logging.info(_test_log)
            print(_test_log)
            return

        triples, sentences = self.data_helper.get_all_datas()
        graph = Graph(triples=triples)
        max_correct_count = 0
        valid_triples, sentences = self.data_helper.get_data('valid')
        # count, threshold = self.find_best_threshold(self.model, valid_triples)
        train_points_cnt = len(graph.entities)
        model_path, saved_epoch, saved_step = self.saver.load_model(self.model, mode='max_step', fail_ok=True)
        print(f"* load model from : {model_path}, epoch: {saved_epoch}, step:{saved_step}")
        step = 0
        entity_ids = sorted(graph.entities)
        for epoch in trange(5, desc="GAKE train epoch "):
            self.model.train()
            for entity_id in tqdm(entity_ids, desc='GAKE train'):
                step += 1
                if step < saved_step or epoch < saved_epoch:
                    continue
                node = graph.entities[entity_id]
                # data
                _neighbor_nodes = graph.get_neighbor_context(entity_id)
                _path_nodes = graph.get_path_context(entity_id)
                _edge_nodes = graph.get_edge_context(entity_id)
                #
                pred, loss = self.model(node, _neighbor_nodes, _path_nodes, _edge_nodes)

                self.backward(loss, self.model)
                if step % 100 == 0:
                    print(f"*train step: {step}/{train_points_cnt},  "
                          f"entity:{entity_id},\tloss:{loss.item():.4f}")
            self.saver.save(self.model, epoch=epoch, step=step, mode="max_step")
            if device == 'gpu':
                torch.cuda.empty_cache()
            count, threshold = self.find_best_threshold(self.model, valid_triples)
            print(f'\n* epoch: {epoch}, count: {count}/{len(valid_triples)}, threshold: {threshold:.4f}')
            if count > max_correct_count:
                max_correct_count = count
                self.saver.save(self.model, epoch=epoch, step=step, mode="max_acc",
                                dic={'count': f'{count}/{len(valid_triples)}', 'best_threshold': threshold})
