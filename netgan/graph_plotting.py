import networkx as nx
import scipy.sparse as sp


def plot_graph_from_sparse_matrix(sparse_matrix, draw_type="networkx"):
    """"
    sparse_matrix (scipy sparse matrix):
     An adjacency matrix representation of a graph
    """
    print(sparse_matrix.shape)
    G = nx.from_scipy_sparse_matrix(sparse_matrix)
    print("graph nodes: ")
    print(list(G.nodes()))
    if draw_type == "networkx":
        nx.draw_networkx(G)
    elif draw_type == "kamada_kawai":
        nx.draw_kamada_kawai(G)
    return


def plot_graph(graph, draw_type="networkx"):
    """"
        plot a graph
    """
    # print(graph.shape)
    sparse_matrix = sp.csr_matrix(graph)
    G = nx.from_scipy_sparse_matrix(sparse_matrix)
    print("graph nodes: ")
    print(list(G.nodes()))
    if draw_type == "networkx":
        nx.draw_networkx(G)
    elif draw_type == "kamada_kawai":
        nx.draw_kamada_kawai(G)
    return
