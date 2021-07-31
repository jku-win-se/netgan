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
    print("graph: ")
    sparse_matrix = sp.csr_matrix(graph)
    G = nx.from_scipy_sparse_matrix(sparse_matrix)
    print("graph nodes: ")
    print(list(G.nodes()))
    if draw_type == "networkx":
        nx.draw_networkx(G)
    elif draw_type == "kamada_kawai":
        nx.draw_kamada_kawai(G)
    return


def graph_similarity(sparse_matrix1, sparse_matrix2, r0, r1, i_timeout):
    g1 = nx.from_scipy_sparse_matrix(sparse_matrix1)
    sparse_matrix = sp.csr_matrix(sparse_matrix2)
    g2 = nx.from_scipy_sparse_matrix(sparse_matrix)

    print("Is generated graph connected? ", nx.is_connected(g2))
    print("Are g1 and g2 isomorphic? ", nx.could_be_isomorphic(g1, g2))

    edit_dis = nx.graph_edit_distance(g1, g2, roots=(r0, r1), timeout=i_timeout)
    print("Edit distance between g1 and g2: ", edit_dis)


if __name__ == "__main__":
    m = [[0, 1, 0], [1, 0, 1], [0, 0, 1]]
    matrix = sp.csr_matrix(m)
    G = nx.from_scipy_sparse_matrix(matrix)
    plot_graph_from_sparse_matrix(matrix)
