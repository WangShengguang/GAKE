import random


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
