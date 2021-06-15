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
        plot an input graph
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


def graph_similarity(sparse_matrix1, sparse_matrix2):

    g1 = nx.from_scipy_sparse_matrix(sparse_matrix1)
    sparse_matrix = sp.csr_matrix(sparse_matrix2)
    g2 = nx.from_scipy_sparse_matrix(sparse_matrix)

    isomorphic = nx.could_be_isomorphic(g1, g2)
    edit_dis = nx.graph_edit_distance(g1, g2)
    # for i in g1.edges:
    #     nx.edge_match(g1, g2)
    print("g1 and g2 are isomorphic? ", isomorphic)
    print("Edit distance between g1 and g2: ", edit_dis)

