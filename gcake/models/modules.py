import random


class _Node(object):
    def __init__(self, id, dtype):
        """
        :param dtype:  entity, relation
        """
        self.id = id
        self.dtype = dtype
        self.ins = set()  # 反向指针
        self.outs = set()  # 关系指针


class _Graph(object):
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
            h_node.outs.add(r_node)  # h  ->  r
            r_node.ins.add(h_node)  # h   <-  r
            r_node.outs.add(t_node)  # r  ->  t
            t_node.ins.add(r_node)  # r   <-  t

        return entities, relations

    def get_neighbor_context(self, entity_id, max_num=100):
        """三元组"""
        entity_node = self.entities[entity_id]
        neighbor_relations = entity_node.outs
        neighbor_triples = set()
        for rel in neighbor_relations:
            for tail in rel.outs:
                neighbor_triples.add((entity_node, rel, tail))
        #
        if len(neighbor_triples) > max_num:
            neighbor_triples = random.sample(neighbor_triples, max_num)
        neighbors = []
        for tri in neighbor_triples:
            neighbors.extend(tri)
        return [entity_node] + neighbors

    def get_path_context(self, entity_id, path_len=5):
        """
        A path in a given knowledge graph reflects both direct and indirect relations
            between entities.
            随机游走
        """
        node = self.entities[entity_id]  # vertex
        edges_context = set()
        times = 0
        while len(edges_context) < path_len * 2 and times < 2 ** path_len:
            times += 1
            if node.outs:
                edge = random.sample(node.outs, 1)[0]
                node = random.sample(edge.outs, 1)[0]
            elif node.ins:
                edge = random.sample(node.ins, 1)[0]
                node = random.sample(edge.ins, 1)[0]
            else:
                break
            edges_context.add(edge)
            edges_context.add(node)
        return [self.entities[entity_id]] + list(edges_context)

    def get_edge_context(self, entity_id, max_num=100):
        """ All relations connecting a given entity are representative to that entity
            所有相邻边
        """
        entity_node = self.entities[entity_id]  # edge
        edge_context = list(entity_node.ins) + list(entity_node.outs)
        if len(edge_context) > max_num:
            edge_context = random.sample(edge_context, max_num)
        return [entity_node] + edge_context
