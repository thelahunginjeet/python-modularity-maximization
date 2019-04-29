# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.linalg import eig
from itertools import product


def transform_net_and_part(network,partition):
    '''
    Accepts an input network and a community partition (keys are nodes,
    values are community ID) and returns a version of the network and
    partition with nodes in the range 0,...,len(G.nodes())-1.  This
    lets you directly map edges to elements of the modularity matrix.

    Returns the modified network and partition.
    '''
    network = nx.convert_node_labels_to_integers(network, first_label=0, label_attribute="node_name")
    node_to_name = nx.get_node_attributes(network, 'node_name')
    # reverse the node_name dict to flip the partition
    name_to_node = {v:k for k,v in node_to_name.items()}
    int_partition = {}
    for k in partition:
        int_partition[name_to_node[k]] = partition[k]
    return network,int_partition


def reverse_partition(partition):
    '''
    Accepts an input graph partition in the form node:community_id and returns
    a dictionary of the form community_id:[node_1,node_2,...].
    '''
    reverse_partition = {}
    for p in partition:
        if partition[p] in reverse_partition:
            reverse_partition[partition[p]].append(p)
        else:
            reverse_partition[partition[p]] = [p]
    return reverse_partition


def modularity(network, partition):
    '''
    Computes the modularity; works for Directed and Undirected Graphs, both
    unweighted and weighted.
    '''
    # put the network and partition into integer node format
    network,partition = transform_net_and_part(network,partition)
    # get the modularity matrix
    Q = get_base_modularity_matrix(network)
    if type(network) == nx.Graph:
        norm_fac = 2.*(network.number_of_edges())
        if nx.is_weighted(network):
            # 2*0.5*sum_{ij} A_{ij}
            norm_fac = nx.to_scipy_sparse_matrix(network).sum()
    elif type(network) == nx.DiGraph:
        norm_fac = 1.*network.number_of_edges()
        if nx.is_weighted(network):
            # sum_{ij} A_{ij}
            norm_fac = nx.to_scipy_sparse_matrix(network).sum()
    else:
        print('Invalid graph type')
        raise TypeError
    # reverse the partition dictionary
    rev_part = reverse_partition(partition)
    # get the list of all within-community pairs
    pairs = []
    for p in rev_part:
        for i,j in product(rev_part[p],rev_part[p]):
            pairs.append((i,j))
    # now sum up all the appropriate values
    return sum([Q[x] for x in pairs])/norm_fac


def get_base_modularity_matrix(network):
    '''
    Obtain the modularity matrix for the whole network.  Assumes any edge weights
    use the key 'weight' in the edge attribute.

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network of interest

    Returns
    -------
    np.matrix
        The modularity matrix for `network`

    Raises
    ------
    TypeError
        When the input `network` does not fit either nx.Graph or nx.DiGraph
    '''

    if type(network) == nx.Graph:
        if nx.is_weighted(network):
            return sparse.csc_matrix(nx.modularity_matrix(network,weight='weight'))
        return sparse.csc_matrix(nx.modularity_matrix(network))
    elif type(network) == nx.DiGraph:
        if nx.is_weighted(network):
            return sparse.csc_matrix(nx.directed_modularity_matrix(network,weight='weight'))
        return sparse.csc_matrix(nx.directed_modularity_matrix(network))
    else:
        raise TypeError('Graph type not supported. Use either nx.Graph or nx.Digraph')


def _get_delta_Q(X, a):
    '''
    Calculate the delta modularity
    .. math::
        \deltaQ = s^T \cdot \^{B_{g}} \cdot s
    .. math:: \deltaQ = s^T \cdot \^{B_{g}} \cdot s

    Parameters
    ----------
    X : np.matrix
        B_hat_g
    a : np.matrix
        s, which is the membership vector

    Returns
    -------
    float
        The corresponding :math:`\deltaQ`
    '''

    delta_Q = (a.T.dot(X)).dot(a)
    return delta_Q[0,0]


def get_mod_matrix(network, comm_nodes=None, B=None):
    '''
    This function computes the modularity matrix
    for a specific group in the network.
    (a.k.a., generalized modularity matrix)

    Specifically,
    .. math::
        B^g_{i,j} = B_ij - \delta_{ij} \sum_(k \in g) B_ik
        m = \abs[\Big]{E}
        B_ij = A_ij - \dfrac{k_i k_j}{2m}
        OR...
        B_ij = \(A_ij - \frac{k_i^{in} k_j^{out}}{m}

    When `comm_nodes` is None or all nodes in `network`, this reduces to :math:`B`

    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network of interest
    comm_nodes : iterable (list, np.array, or tuple)
        List of nodes that defines a community
    B : np.matrix
        Modularity matrix of `network`

    Returns
    -------
    np.matrix
        The modularity of `comm_nodes` within `network`
    '''

    if comm_nodes is None:
        comm_nodes = list(network)
        return get_base_modularity_matrix(network)

    if B is None:
        B = get_base_modularity_matrix(network)

    # subset of mod matrix in g
    indices = [list(network).index(u) for u in comm_nodes]
    B_g = B[indices, :][:, indices]
    #print 'Type of `B_g`:', type(B_g)

    # B^g_(i,j) = B_ij - δ_ij * ∑_(k∈g) B_ik
    # i, j ∈ g
    B_hat_g = np.zeros((len(comm_nodes), len(comm_nodes)), dtype=float)

    # ∑_(k∈g) B_ik
    B_g_rowsum = np.asarray(B_g.sum(axis=1))[:, 0]
    if type(network) == nx.Graph:
        B_g_colsum = np.copy(B_g_rowsum)
    elif type(network) == nx.DiGraph:
        B_g_colsum = np.asarray(B_g.sum(axis=0))[0, :]

    for i in range(B_hat_g.shape[0]):
        for j in range(B_hat_g.shape[0]):
            if i == j:
                B_hat_g[i,j] = B_g[i,j] - 0.5 * (B_g_rowsum[i] + B_g_colsum[i])
            else:
                B_hat_g[i,j] = B_g[i,j]

    if type(network) == nx.DiGraph:
        B_hat_g = B_hat_g + B_hat_g.T

    return sparse.csc_matrix(B_hat_g)


def largest_eig(A):
    '''
        A wrapper over `scipy.linalg.eig` to produce
        largest eigval and eigvector for A when A.shape is small
    '''
    vals, vectors = eig(A.todense())
    real_indices = [idx for idx, val in enumerate(vals) if not bool(val.imag)]
    vals = [vals[i].real for i in range(len(real_indices))]
    vectors = [vectors[i] for i in range(len(real_indices))]
    max_idx = np.argsort(vals)[-1]
    return np.asarray([vals[max_idx]]), np.asarray([vectors[max_idx]]).T
