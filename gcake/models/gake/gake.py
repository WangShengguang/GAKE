import random

import torch
from torch import nn

from config import Config
from gcake.trainer import BaseTrainer


class Node(object):
    def __init__(self, id, dtype):
        """
        :param dtype:  entity, relation
        """
        self.id = id
        self.dtype = dtype
        self.left = set()  # 反向指针
        self.right = set()  # 关系指针


class Graph(object):
    def __init__(self, triples):
        self.entities, self.relations = self.create_graph(triples)

    def create_graph(self, triples):
        entities = {}
        relations = {}
        for h, r, t in triples:
            if h not in entities:
                entities[h] = Node(id=h, dtype="entity")
            if t not in entities:
                entities[t] = Node(id=t, dtype="entity")
            if r not in relations:
                relations[r] = Node(id=r, dtype="relation")

            h_node, t_node = entities[h], entities[t]
            r_node = relations[r]
            # h -> r -> t
            h_node.right.add(r_node)
            r_node.left.add(h_node)
            r_node.right.add(t_node)
            t_node.left.add(r_node)

        return entities, relations

    def get_neighbor_context(self, entity_id, num=10):
        entity_node = self.entities[entity_id]
        neighbor_relations = entity_node.right
        neighbor_entities = set()
        for rel in neighbor_relations:
            neighbor_entities.update(rel.right)
        neighbors = list(neighbor_relations) + list(neighbor_entities)
        return [node.id for node in neighbors]

    def get_path_context(self, entity_id, path_len=5):
        """
        A path in a given knowledge graph reflects both direct and indirect relations
            between entities.
        """
        node = self.entities[entity_id]  # vertex
        visited_edges = set()
        times = 0
        while len(visited_edges) < path_len and times < 2 ** path_len:
            times += 1
            if node.right:
                node = random.sample(node.right, 1)[0]
            elif node.left:
                node = random.sample(node.left, 1)[0]
            if node.dtype == "relation":
                visited_edges.add(node)
        return [node.id for node in visited_edges]

    def get_edge_context(self, entity_id):
        """ All relations connecting a given entity are representative to that entity
        """
        entity_node = self.entities[entity_id]  # edge
        edge_context = list(entity_node.left) + list(entity_node.right)
        return [node.id for node in edge_context]


class GAKE(nn.Module):
    """
    https://www.aclweb.org/anthology/C16-1062.pdf
    """

    def __init__(self, entity_num, relation_num, dim=128):
        super().__init__()
        self.entity_num = entity_num
        self.embeddings = nn.Embedding(entity_num + relation_num, dim)
        self.linear = nn.Linear(3, 1)

    def get_p(self, si_id, si_context_ids):
        si_context_emb = self.embeddings(si_context_ids)
        context_pie = torch.sum(si_context_emb, dim=0) / torch.norm(si_context_emb, p=2)  # (n,dim)

        def cacu_exp(node_ids):
            nodes_emb = self.embeddings(node_ids)
            exp_res = torch.exp(torch.mm(nodes_emb, torch.transpose(context_pie.unsqueeze(0), 0, 1)))
            return exp_res

        p = cacu_exp(si_id) / torch.sum(cacu_exp(si_context_ids))
        return p

    def forward(self, entity_id, neighbor_ids, path_ids, edge_ids):
        _neighbors_p = self.get_p(entity_id, neighbor_ids)
        _paths_p = self.get_p(entity_id, path_ids)
        _edge_p = self.get_p(entity_id, edge_ids)
        global_weight_p = self.linear(torch.cat([_neighbors_p, _paths_p, _edge_p]).squeeze())  # 全局概率，最大化
        loss = 1 - global_weight_p
        return global_weight_p, loss

    def find_best_threshold(self, valid_datas):
        """
        :param valid_datas:
        :return:
        """
        pass


class Trainer(BaseTrainer):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def run(self, mode):
        from gcake.data_helper import DataHelper
        data_helper = DataHelper(self.dataset)
        entity_num = len(data_helper.entity2id)
        relation_num = len(data_helper.relation2id)
        model = GAKE(entity_num, relation_num, dim=128)
        self.init_model(model)

        triples, sentences = data_helper.get_all_datas()
        graph = Graph(triples=triples)

        for i, entity_id in enumerate(graph.entities):
            # data
            _neighbor_ids = graph.get_neighbor_context(entity_id)
            _path_ids = graph.get_path_context(entity_id)
            _edge_ids = graph.get_edge_context(entity_id)
            #
            entity_id = torch.tensor([entity_id], dtype=torch.long).to(Config.device)
            neighbor_ids = torch.tensor(_neighbor_ids, dtype=torch.long).to(Config.device)
            path_ids = torch.tensor(_path_ids, dtype=torch.long).to(Config.device)
            edge_ids = torch.tensor(_edge_ids, dtype=torch.long).to(Config.device)

            global_weight_p, loss = model(entity_id, neighbor_ids, path_ids, edge_ids)
            self.backfoward(loss, model)
            # import ipdb
            # ipdb.set_trace()
            print(f"* i:{i},entity:{entity_id.item()}, loss:{loss.item():.4f}")
