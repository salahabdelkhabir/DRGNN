import networkx as nx
import matplotlib.pyplot as plt

try:
    from dgl import DGLGraph
except ImportError:
    DGLGraph = None


def create_graph(edge_list):
    graph = nx.Graph()
    graph.add_edges_from(edge_list)
    return graph


def visualize_graph(graph, title="Graph Visualization"):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=1500, font_size=12)
    plt.title(title)
    plt.show()


def convert_to_dgl(graph):
    return DGLGraph(graph)


def load_graph_from_file(file_path):
    graph = nx.read_edgelist(file_path, nodetype=int, data=(('weight', float),))
    return graph


def save_graph_to_file(graph, file_path):
    nx.write_edgelist(graph, file_path, data=["weight"])
