import json
import os
import random

import numpy as np
import pandas as pd
import torch
from keras.preprocessing import sequence

from config import data_dir, Config
from dataset.common import sentence_to_words


class DataHelper(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.pos_triples_set = set()  # 配合生成负样本
        self.load_data()

    def load_data(self):
        with open(os.path.join(data_dir, self.dataset, "entity2id.json"), "r", encoding="utf-8") as f:
            self.entity2id = json.load(f)
        with open(os.path.join(data_dir, self.dataset, "relation2id.json"), "r", encoding="utf-8") as f:
            self.relation2id = json.load(f)
        with open(os.path.join(data_dir, self.dataset, "word2id.json"), "r", encoding="utf-8") as f:
            self.word2id = json.load(f)

    def get_data(self, data_type):
        data_frame = pd.read_csv(os.path.join(data_dir, self.dataset, f"{data_type}.csv"))
        head, relation, tail, sentence = (data_frame["head"], data_frame["relation"],
                                          data_frame["tail"], data_frame["sentence"])
        heads = [self.entity2id[entity] for entity in head]
        relations = [self.relation2id[rel] for rel in relation]
        tails = [self.entity2id[entity] for entity in tail]
        hrts = [(h, r, t) for h, r, t in zip(heads, relations, tails)]
        sentences = [[self.word2id.get(token, 0) for token in sentence_to_words(s)] for s in sentence]
        assert len(hrts) == len(sentences)
        return hrts, sentences

    def get_all_datas(self):
        all_triples = []
        all_sentences = []
        for data_type in ["train", "valid", "test"]:
            hrts, sentences = self.get_data(data_type)
            all_triples.extend(hrts)
            all_sentences.extend(sentences)
        return all_triples, all_sentences

    def get_batch_negative_samples(self, batch_positive_samples):
        if not self.pos_triples_set:
            all_triples, all_sentences = self.get_all_datas()
            self.pos_triples_set = set(all_triples)
            self.all_sentences = all_sentences
            self.entity_ids = list(self.entity2id.values())

        batch_negative_triples = []
        batch_negative_sentences = []
        for (h, r, t) in batch_positive_samples:
            while (h, r, t) in self.pos_triples_set:
                e = random.choice(self.entity_ids)
                if random.randint(0, 1):
                    h = e
                else:
                    t = e
            batch_negative_triples.append((h, r, t))
            batch_negative_sentences.append(random.sample(self.all_sentences, 1)[0])
        assert len(batch_positive_samples) == len(batch_negative_triples) == len(batch_negative_sentences)
        return batch_negative_triples, batch_negative_sentences

    def batch_iter(self, data_type, batch_size, _shuffle=True, negative_y_label=-1.0):
        positive_triples, positive_sentences = self.get_data(data_type)
        semi_data_size = len(positive_triples)
        order = list(range(semi_data_size))
        if _shuffle:
            np.random.shuffle(order)
        semi_batch_size = batch_size // 2
        epoch_step = semi_data_size // semi_batch_size

        for batch_step in range(epoch_step):
            # print("batch_step： {}".format(batch_step))
            batch_idxs = order[batch_step * semi_batch_size:(batch_step + 1) * semi_batch_size]
            if len(batch_idxs) != semi_batch_size:
                continue
            _positive_triples = [positive_triples[idx] for idx in batch_idxs]
            _positive_sentences = [positive_sentences[idx] for idx in batch_idxs]
            _negative_triples, _negative_sentences = self.get_batch_negative_samples(_positive_triples)
            triples = _positive_triples + _negative_triples
            sentences = _positive_sentences + _negative_sentences
            sentences = sequence.pad_sequences(sentences,
                                               # maxlen=min(Config.max_len, max([len(s) for s in sentences])),
                                               maxlen=Config.max_len,
                                               padding="post", value=self.word2id["[PAD]"])
            y_batch = [[1.0]] * semi_batch_size + [[negative_y_label]] * semi_batch_size
            #
            triples = torch.tensor(triples, dtype=torch.long).to(Config.device)
            sentences = torch.tensor(sentences, dtype=torch.long).to(Config.device)
            y_labels = torch.tensor(y_batch, dtype=torch.float32).to(Config.device)
            yield triples, sentences, y_labels
            # yield np.asarray(triples), np.asarray(sentences), np.asarray(y_batch)
