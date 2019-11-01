import random


class Node(object):
    def __init__(self, id, dtype):
        """
        :param dtype:  entity, relation
        """
        self.id = id
        self.dtype = dtype
        self.ins = set()  # 反向指针
        self.outs = set()  # 关系指针


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
            h_node.outs.add(r_node)  # h -> r
            r_node.ins.add(h_node)  # h <- r
            r_node.outs.add(t_node)  # r -> t
            t_node.ins.add(r_node)  # r <- t

        return entities, relations

    def get_neighbor_context(self, entity_id, num=10):
        entity_node = self.entities[entity_id]
        neighbor_relations = entity_node.outs
        neighbor_entities = set()
        for rel in neighbor_relations:
            neighbor_entities.update(rel.outs)
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
            if node.outs:
                node = random.sample(node.outs, 1)[0]
            elif node.ins:
                node = random.sample(node.ins, 1)[0]
            if node.dtype == "relation":
                visited_edges.add(node)
        return [node.id for node in visited_edges]

    def get_edge_context(self, entity_id):
        """ All relations connecting a given entity are representative to that entity
        """
        entity_node = self.entities[entity_id]  # edge
        edge_context = list(entity_node.ins) + list(entity_node.outs)
        return [node.id for node in edge_context]
