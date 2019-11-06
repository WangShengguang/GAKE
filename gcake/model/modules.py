import random


class Node(object):
    def __init__(self, id, dtype):
        """
        :param dtype:  entity, relation
        """
        self.id = id
        assert dtype in ['head', 'tail', 'relation']
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
                entities[h] = Node(id=h, dtype="head")
            if t not in entities:
                entities[t] = Node(id=t, dtype="tail")
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

    def get_neighbor_context(self, node: Node):
        """相连的三元组
            h -> r,t
            t -> h,r
            r -> h,t
        """
        neighbor_triples = set()
        if node.dtype == 'head':
            neighbor_relations = node.outs
            for rel in neighbor_relations:
                for tail in rel.outs:
                    neighbor_triples.add((node, rel, tail))
        elif node.dtype == 'tail':
            neighbor_relations = node.ins
            for rel in neighbor_relations:
                for head in rel.ins:
                    neighbor_triples.add((head, rel, node))
        else:  # node.dtype == relation
            heads = node.ins
            tails = node.outs
            for h, t in zip(heads, tails):
                neighbor_triples.add((h, node, t))
        #
        neighbor_nodes = set()
        for tri in neighbor_triples:
            neighbor_nodes.update(tri)
        return neighbor_nodes

    def get_path_context(self, node: Node, path_len=5):
        """
        A path in a given knowledge graph reflects both direct and indirect relations
            between entities.
            随机游走
        """
        path_nodes = set()
        times = 0
        while len(path_nodes) < path_len * 2 and times < 2 ** path_len:
            times += 1
            if node.outs:
                node = random.sample(node.outs, 1)[0]
            elif node.ins:
                node = random.sample(node.ins, 1)[0]
            else:
                break
            path_nodes.add(node)
        return path_nodes

    def get_edge_context(self, node: Node):
        """ All relations connecting a given entity are representative to that entity
            所有相邻边
        """
        edge_nodes = set()
        if node.dtype != 'relation':
            edge_nodes.update(node.ins)
            edge_nodes.update(node.outs)
        return edge_nodes

    def get_context(self, node, path_len=5):
        _neighbor_nodes = self.get_neighbor_context(node)
        _path_nodes = self.get_path_context(node, path_len)
        _edge_nodes = self.get_edge_context(node)
        return list(_neighbor_nodes | _path_nodes | _edge_nodes)
