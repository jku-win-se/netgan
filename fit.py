from netgan.netgan import *
from netgan import utils

from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib import pyplot as plt
import scipy.sparse as sp
import tensorflow as tf
import numpy as np
import time

def fit(adj):
    '''does the thing
        parameters:
            adj (scipy sparse csr): adjacency matrix for the input graph
        output:
            model (?): the trained model
    '''
    lcc = utils.largest_connected_components(adj)
    adj = adj[lcc,:][:,lcc]
    n = adj.shape[0]

    val_share = 0.1
    test_share = 0.05
    #seed = 481516234

    # split the graph into train/test/validation
    train_ones, val_ones, val_zeros, test_ones, test_zeros = utils.train_val_test_split_adjacency(adj, val_share, test_share, undirected=True, connected=True, asserts=True)

    # generate the training graph and ensure it is symmetric
    #train_graph = sp.coo.matrix((np.ones(len(train_ones)), (train_ones[:, 0], train_ones[:,1]))).tocsr()
    train_graph = sp.coo_matrix((np.ones(len(train_ones)),(train_ones[:,0], train_ones[:,1]))).tocsr()
    assert (train_graph.toarray() == train_graph.toarray().T).all()

    rw_len = 16
    batch_size = 128

    walker = utils.RandomWalker(train_graph, rw_len, p=1, q=1, batch_size=batch_size)

    # define the model
    model = NetGAN(n, rw_len, walk_generator=walker.walk, \
            gpu_id=0, use_gumbel=True, disc_iters=3, \
            W_down_generator_size=128, W_down_discriminator_size=128, \
            l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5, \
            generator_layers=[40], discriminator_layers=[30], \
            temp_start=5, learning_rate=0.0003)

    # stopping criterion can be one of 'val' or 'eo'
    stopping_criterion = 'val'
    if stopping_criterion == 'eo':
        stopping = 0.5
    else:
        stopping = None

    eval_every = 3
    #plot_every = 2000

    # train the model
    #log_dict = model.train(A_orig=adj, val_ones=val_ones, val_zeros=val_zeros, \
    #        stopping=stopping, eval_every=eval_every, max_patience=5, max_iters=30000)
    log_dict = model.train(A_orig=adj, val_ones=val_ones, val_zeros=val_zeros, \
            stopping=stopping, eval_every=eval_every, max_patience=5, max_iters=4)

    sample_walks = model.generate_discrete(10000, reuse=True)

    samples = []
    for x in range(60):
        if (x + 1) % 10 == 0:
            print(x + 1)
        samples.append(sample_walks.eval({model.tau: 0.5}))
    #print(samples)

    random_walks = np.array(samples).reshape([-1, rw_len])
    scores_matrix = utils.score_matrix_from_random_walks(random_walks, n).tocsr()
    sampled_graph = utils.graph_from_scores(scores_matrix, train_graph.sum())

    return sampled_graph

def main():
    A, _X_obs, _z_obs = utils.load_npz('data/cora_ml.npz')
    A = A + A.T
    A[A > 1] = 1

    scores = fit(A)

    print('--------------------------')
    print(scores)

main()
