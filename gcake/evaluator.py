import logging
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from config import Config
from gcake.data_helper import DataHelper


class Predictor(object):
    def __init__(self, dataset, model_name, load_model=True):
        self.model_name = model_name
        self.dataset = dataset
        self.data_helper = DataHelper(dataset)
        self.entity_nums = len(self.data_helper.entity2id)
        self.relation_nums = len(self.data_helper.relation2id)
        if load_model:  # 方便训练过程中测试
            self.load_model()

    def load_model(self):
        pass

    def set_model(self, model):
        model.eval()
        self.model = model

    def predict(self, batch_h: List, batch_r: List, batch_t: List):
        """ 用于预测
        :param batch_h: [0,6,3,...,h_id]
        :param batch_t: [0,6,3,...,t_id]
        :param batch_r: [0,6,3,...,r_id]
        :return:
        """
        input_size, _batch_size = len(batch_h), Config.batch_size
        x_data = list(zip(batch_h, batch_r, batch_t)) + [(0, 0, 0)] * (_batch_size - input_size % _batch_size)
        prediction = []
        for input_x in [x_data[i:i + _batch_size] for i in range(0, len(x_data), _batch_size)]:
            _pred = self.model(input_x)
            prediction.extend(_pred.flatten().tolist())
        return np.array(prediction[:input_size])

    # link Predict 链接预测，预测头实体or尾实体
    def predict_head_entity(self, t, r):
        r'''This mothod predicts the top k head entities given tail entity and relation.

        Args:
            t (int): tail entity id
            r (int): relation id
            k (int): top k head entities
        Returns:
            list: k possible head entity ids order by score desc
        '''
        test_h = list(self.data_helper.entity2id.values())
        test_r = [r] * self.entity_nums
        test_t = [t] * self.entity_nums
        predictions = self.predict(test_h, test_r, test_t)
        head_ids = predictions.flatten().argsort().tolist()  # dist越小越相似
        # print(head_ids)
        return head_ids

    def predict_tail_entity(self, h, r):
        r'''This mothod predicts the top k tail entities given head entity and relation.

        Args:
            h (int): head entity id
            r (int): relation id
            k (int): top k tail entities
        Returns:
            list: k possible tail entity ids
        '''
        test_h = [h] * self.entity_nums
        test_r = [r] * self.entity_nums
        test_t = list(self.data_helper.entity2id.values())
        predictions = self.predict(test_h, test_r, test_t)
        tail_ids = predictions.flatten().argsort().tolist()  # dist越小越相似
        # print(tail_ids)
        return tail_ids

    def predict_relation(self, h, t):
        r'''This methods predict the relation id given head entity and tail entity.

        Args:
            h (int): head entity id
            t (int): tail entity id
            k (int): top k relations

        Returns:
            list: k possible relation ids
        '''
        test_h = [h] * self.relation_nums
        test_r = list(range(self.relation_nums))
        test_t = [t] * self.relation_nums
        predictions = self.predict(test_h, test_t, test_r)
        relations = predictions.flatten().argsort().tolist()  # dist越小越相似
        # print(relations)
        return relations

    def predict_triple(self, h, t, r, thresh=None):
        r'''This method tells you whether the given triple (h, t, r) is correct of wrong

        Args:
            h (int): head entity id
            t (int): tail entity id
            r (int): relation id
            thresh (fload): threshold for the triple
        '''
        prediction = self.predict([h], [t], [r])
        return prediction


class Evaluator(Predictor):
    def __init__(self, dataset, model_name, load_model=True):
        super().__init__(dataset, model_name, load_model)

    def get_valid_lr(self):
        valid_left, valid_right = {}, {}
        valid_triples, sentences = self.data_helper.get_data("valid")  # [(h,t,r)]
        valid_count = len(valid_triples)
        for i in range(valid_count):
            if (valid_triples[i][-1] != valid_triples[i - 1][-1]):
                valid_right[valid_triples[i - 1][-1]] = i - 1
                valid_left[valid_triples[i][-1]] = i
        valid_left[valid_triples[0][-1]] = 0
        valid_right[valid_triples[valid_count - 1][-1]] = valid_count - 1
        return valid_left, valid_right

    def get_best_threshold(self, positives_score, negatives_score):
        interval = 0.01
        bestAcc, max_score, min_score = 0.0, 0.0, 1000
        bestThresh = 0.0
        rel_threshold = {}
        valid_left, valid_right = self.get_valid_lr()
        for r in self.data_helper.relation2id.values():
            if valid_left[r] == -1:
                continue
            total = (valid_right[r] - valid_left[r] + 1) * 2

            min_score = min(min_score, negatives_score[valid_left[r]])
            max_score = max(max_score, positives_score[valid_left[r]])
            for i in range(valid_left[r] + 1, valid_right[r] + 1):
                min_score = min(min_score, positives_score[i], negatives_score[i])
                max_score = max(max_score, positives_score[i], negatives_score[i])

            n_interval = int((max_score - min_score) / interval)
            for i in range(n_interval + 1):
                tmpThresh = min_score + i * interval
                correct = 0
                for j in range(valid_left[r], valid_right[r] + 1):
                    if (positives_score[j] <= tmpThresh):
                        correct += 1
                    if (negatives_score[j] > tmpThresh):
                        correct += 1

                tmpAcc = 1.0 * correct / total
                if i == 0:
                    bestThresh = tmpThresh
                    bestAcc = tmpAcc
                elif tmpAcc > bestAcc:
                    bestAcc = tmpAcc
                    bestThresh = tmpThresh

            rel_threshold[r] = bestThresh
        return rel_threshold

    def evaluate_metrics(self, triples_li, _tqdm=True):
        """
        :param triples_li: [(h,t,r),(h,t,r),...]
        :param _tqdm: 进度条
        :return:
        """
        # logging.info("*model:{} {}, test start, {}: {} ".format(self.model_name, self.dataset, data_type, total))
        total = len(triples_li)
        if _tqdm:
            triples_li = tqdm(triples_li, desc=f"{self.model_name} {self.dataset} test_link_prediction", total=total)
        ranks = []
        ranks_left = []
        ranks_right = []
        hits_N = [1, 3, 10]
        hits = {k: [] for k in hits_N}
        hits_left = {k: [] for k in hits_N}
        hits_right = {k: [] for k in hits_N}

        def get_hit_num(rank):
            """
            :param rank:
            :return: {1:1,3:1,10:0}
            """
            hit_num = {k: 1 if rank <= k else 0 for k in hits_N}
            return hit_num

        for step, (h, r, t) in enumerate(triples_li):
            # left_rank, right_rank = self.get_rank_score(h, t, r)
            pred_head_ids = self.predict_head_entity(t, r)
            left_rank = pred_head_ids.index(h) + 1
            pred_tail_ids = self.predict_tail_entity(h, r)
            right_rank = pred_tail_ids.index(t) + 1
            ranks.append(left_rank), ranks.append(right_rank)
            ranks_left.append(left_rank)
            ranks_right.append(right_rank)

            for k, _hit_num in get_hit_num(left_rank).items():
                hits_left[k].append(_hit_num)
                hits[k].append(_hit_num)
            for k, _hit_num in get_hit_num(right_rank).items():
                hits_right[k].append(_hit_num)
                hits[k].append(_hit_num)

            if _tqdm:
                logging.info("*model:{} {}, test step: {}/{}".format(self.model_name, self.dataset, step, total))
        # Hit@N
        left_hits = {k: np.mean(hit_nums) for k, hit_nums in hits_left.items()}
        right_hits = {k: np.mean(hit_nums) for k, hit_nums in hits_right.items()}
        ave_hits = {k: np.mean(hit_nums) for k, hit_nums in hits.items()}
        # MR
        mr = np.mean(ranks)
        mr_left = np.mean(ranks_left)
        mr_right = np.mean(ranks_right)
        # MRR
        mrr = np.mean(1. / np.array(ranks))
        mrr_left = np.mean(1. / np.array(ranks_left))
        mrr_right = np.mean(1. / np.array(ranks_right))
        metrics = {
            "left": {"MR": mr_left, "MRR": mrr_left,
                     "Hit@10": left_hits[10], "Hit@3": left_hits[3], "Hit@1": left_hits[1]},
            "right": {"MR": mr_right, "MRR": mrr_right,
                      "Hit@10": right_hits[10], "Hit@3": right_hits[3], "Hit@1": right_hits[1]},
            "ave": {"MR": mr, "MRR": mrr,
                    "Hit@10": ave_hits[10], "Hit@3": ave_hits[3], "Hit@1": ave_hits[1]}
        }
        return metrics

    def test_link_prediction(self, data_type, _tqdm=True):
        """
        链接预测，预测头实体或尾实体
        """
        triples, sentences = self.data_helper.get_data(data_type)
        logging.info("* model:{},{} test_link_prediction start, {}: {} ,Config.test_count：{}".format(
            self.model_name, self.dataset, data_type, len(triples), Config.test_count))
        metrics = self.evaluate_metrics(triples[:Config.test_count], _tqdm=_tqdm)
        mr, mrr = metrics["ave"]["MR"], metrics["ave"]["MRR"]
        hit_10, hit_3, hit_1 = metrics["ave"]["Hit@10"], metrics["ave"]["Hit@3"], metrics["ave"]["Hit@1"]
        # logging.info("mr:{:.3f}, mrr:{:.3f}, hit_1:{:.3f}, hit_3:{:.3f}, hit_10:{:.3f}".format(
        #     mr, mrr, hit_1, hit_3, hit_10))
        return mr, mrr, hit_10, hit_3, hit_1

    def test_triple_classification(self, data_type):
        y_true = []
        y_pred = []
        positive_samples, negative_samples = [], []
        # todo get positive_samples, negative_samples
        _positive_samples, _negative_samples = np.asarray(positive_samples), np.asarray(negative_samples)
        positive_score = self.predict(batch_h=_positive_samples[:, 0], batch_t=_positive_samples[:, 1],
                                      batch_r=_positive_samples[:, 2])
        negative_score = self.predict(batch_h=_negative_samples[:, 0], batch_t=_negative_samples[:, 1],
                                      batch_r=_negative_samples[:, 2])
        rel_threshold = self.get_best_threshold(positive_score, negative_score)
        total = len(positive_samples) + len(negative_samples)
        logging.info("* model:{} {}, test_triple_classification start, {}: {} ".format(
            self.model_name, self.dataset, data_type, total))
        for x_batch, y_batch in tqdm(
                self.data_helper.batch_iter(data_type=data_type, batch_size=Config.batch_size),
                total=total / Config.batch_size,
                desc="{} {} test_triple_classification".format(self.model_name, self.dataset)):
            prediction = self.predict(batch_h=x_batch[:, 0], batch_t=x_batch[:, 1], batch_r=x_batch[:, 2])
            for i, _pred in enumerate(prediction.reshape([-1]).astype(int).tolist()):
                _threshold = rel_threshold[x_batch[i][2]]  # 越小越相似，为正样例
                pred = 1 if _pred < _threshold else 0
                y_pred.append(pred)
            y_batch[y_batch <= 0] = 0
            y_true.extend(y_batch.reshape([-1]).tolist())
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return accuracy, precision, recall, f1


def get_rank_hit_metrics(y_id, pred_ids):
    """
    :type y_id: int
    :type pred_ids:list, k possible ids order by score desc
    """
    ranking = pred_ids.index(y_id) + 1
    _matrics = {
        'MRR': 1.0 / ranking,
        'MR': float(ranking),
        'HITS@1': 1.0 if ranking <= 1 else 0.0,
        'HITS@3': 1.0 if ranking <= 3 else 0.0,
        'HITS@10': 1.0 if ranking <= 10 else 0.0}
    return _matrics
