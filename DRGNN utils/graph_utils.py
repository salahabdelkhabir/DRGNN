# utils/graph_utils.py

import networkx as nx
import matplotlib.pyplot as plt
from dgl import DGLGraph

# Function to create a graph from edge list
def create_graph(edge_list):
    graph = nx.Graph()
    graph.add_edges_from(edge_list)
    return graph

# Function to visualize a graph
def visualize_graph(graph, title="Graph Visualization"):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=1500, font_size=12)
    plt.title(title)
    plt.show()

# Function to convert NetworkX graph to DGL graph
def convert_to_dgl(graph):
    return DGLGraph(graph)

# Function to load a graph from file
def load_graph_from_file(file_path):
    graph = nx.read_edgelist(file_path, nodetype=int, data=(('weight', float),))
    return graph

# Function to save a graph to file
def save_graph_to_file(graph, file_path):
    nx.write_edgelist(graph, file_path, data=["weight"])

# Example usage
if __name__ == "__main__":
    edge_list = [(1, 2), (2, 3), (3, 4), (4, 1)]

    # Create and visualize graph
    graph = create_graph(edge_list)
    visualize_graph(graph)

    # Convert to DGL graph
    dgl_graph = convert_to_dgl(graph)

    # Save and load graph
    save_graph_to_file(graph, "example_graph.edgelist")
    loaded_graph = load_graph_from_file("example_graph.edgelist")
    visualize_graph(loaded_graph, title="Loaded Graph")
