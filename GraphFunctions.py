import random
from itertools import combinations

from scipy.spatial import distance

from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from collections import deque


def generate_graph(n, k, p, q, seed=None, ensure_connected_blocks=False,
                   connect_blocks_in_ring=False):
    """
    Generates a graph consisting of k Erdős-Rényi subgraphs with random inter-subgraph edges.

    Args:
        n (int): The number of nodes in each subgraph.
        k (int): The number of subgraphs to generate.
        p (float): The probability of creating an edge between two nodes within the same subgraph.
        q (float): The probability of creating an edge between nodes from different subgraphs.
        seed (int or None): Random seed.
        ensure_connected_blocks (bool): Resample every within-block graph until it is connected.
        connect_blocks_in_ring (bool): Add one random bridge between each consecutive pair of blocks.

    Returns:
        tuple: A tuple containing:
            - G (networkx.Graph): The generated graph with inter- and intra-subgraph edges.
            - truth (list): A list where each element corresponds to the community label of the node
              (in the form of a list of k values repeated n times for each subgraph).
    """
    rng = random.Random(seed)
    G = nx.Graph()
    subgraphs = []

    truth = []

    for i in range(k):
        t = [i] * n
        truth += t

    for i in range(k):
        while True:
            subgraph_seed = rng.randrange(2**32)
            subgraph = nx.erdos_renyi_graph(n, p, seed=subgraph_seed)
            if not ensure_connected_blocks or nx.is_connected(subgraph):
                break
        mapping = {node: node + i * n for node in subgraph.nodes()}
        nx.relabel_nodes(subgraph, mapping, copy=False)
        G = nx.compose(G, subgraph)
        subgraphs.append(set(mapping.values()))

    # Add random inter-subgraph edges with probability q
    for i in range(k):
        for j in range(i + 1, k):
            for u in subgraphs[i]:
                for v in subgraphs[j]:
                    if rng.random() < q:
                        G.add_edge(u, v)

    if connect_blocks_in_ring and k > 1:
        for i in range(k):
            left = tuple(sorted(subgraphs[i]))
            right = tuple(sorted(subgraphs[(i + 1) % k]))
            G.add_edge(rng.choice(left), rng.choice(right))

    return G, truth


import numpy as np
import networkx as nx


def permute_graph_with_truth(G, truth=None):
    """
    Randomly permutes node labels of a NetworkX graph and updates the truth list accordingly.

    Assumes:
    - Nodes in G are labeled 0 to n-1
    - truth[i] is the label of node i

    Returns:
    - G_permuted: the relabeled graph
    - truth_permuted: list of labels where truth_permuted[i] is the label of node i in G_permuted
    """
    n = len(G)
    old_nodes = list(range(n))
    new_nodes = list(old_nodes)
    np.random.shuffle(new_nodes)

    # Mapping: old -> new
    mapping = {old: new for old, new in zip(old_nodes, new_nodes)}
    G_permuted = nx.relabel_nodes(G, mapping)

    G_permuted_sorted = G.__class__()  # preserves Graph/DiGraph type
    G_permuted_sorted.add_nodes_from(sorted(G_permuted.nodes()))
    G_permuted_sorted.add_edges_from(G_permuted.edges(data=True))

    if truth is not None:
        # Inverse mapping: new -> old
        inverse_mapping = {v: k for k, v in mapping.items()}
        truth_permuted = [truth[inverse_mapping[i]] for i in range(n)]
        return G_permuted_sorted, truth_permuted

    return G_permuted_sorted, None


def create_graphs_hop_distance(G, friend_bound, enemy_bound):
    """
    Creates two graphs based on hop distance between nodes: one for "friend" relationships and one for
    "enemy" relationships based on path lengths between nodes.

    Args:
        G (networkx.Graph): The input graph to analyze. It must be undirected.
        friend_bound (float): A threshold value (between 0 and 1) determining the maximum relative hop
                              distance for an edge to be considered a "friend" edge.
        enemy_bound (float): A threshold value between 0 and 1 determining the minimum relative hop
                             distance for an edge to be considered an "enemy" edge.

    Returns:
        tuple: A tuple containing two graphs:
            - G_F (networkx.Graph): The graph representing "friend" edges, where nodes are connected
                                    if their hop distance is within the `friend_bound`.
            - G_E (networkx.Graph): The graph representing "enemy" edges, where nodes are connected
                                    if their hop distance is greater than `enemy_bound`.
    """
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))

    # Calculate graph diameter
    max_path = 0
    for v in shortest_paths.values():
        m = max(v.values())
        if max_path < m:
            max_path = m

    G_F = nx.Graph()
    G_E = nx.Graph()

    # Add nodes to G_F and G_E
    G_F.add_nodes_from(G.nodes())
    G_E.add_nodes_from(G.nodes())
    for u in G.nodes():
        for v in G.nodes():
            if u < v:  # Avoid duplicate edges since the graph is undirected
                if shortest_paths[u].keys().__contains__(v):
                    path_length = shortest_paths[u][v]
                else:
                    path_length = max_path
                if path_length <= 1 or path_length <= max_path * friend_bound:
                    G_F.add_edge(u, v)
                elif path_length > max_path * enemy_bound:
                    G_E.add_edge(u, v)

    return G_F, G_E


def create_relations_hop_distance(G, friend_bound, enemy_bound):
    """
    Creates two graphs based on hop distance between nodes: one for "friend" relationships and one for
    "enemy" relationships based on path lengths between nodes.

    Args:
        G (networkx.Graph): The input graph to analyze. It must be undirected.
        friend_bound (float): A threshold value (between 0 and 1) determining the maximum relative hop
                              distance for an edge to be considered a "friend" edge.
        enemy_bound (float): A threshold value between 0 and 1 determining the minimum relative hop
                             distance for an edge to be considered an "enemy" edge.

    Returns:
        tuple: A tuple containing two graphs:
            - G_F (networkx.Graph): The graph representing "friend" edges, where nodes are connected
                                    if their hop distance is within the `friend_bound`.
            - G_E (networkx.Graph): The graph representing "enemy" edges, where nodes are connected
                                    if their hop distance is greater than `enemy_bound`.
    """
    nodes = list(G.nodes())

    # Compute shortest paths once
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))

    # Graph diameter
    max_path = max(
        max(lengths.values())
        for lengths in shortest_paths.values()
    )

    friend_limit = max(max_path * friend_bound, 1)
    enemy_cutoff = max_path * enemy_bound

    friendship_edges = {n: set() for n in nodes}
    enemy_edges = {n: set() for n in nodes}

    # Iterate each unordered pair only once
    for u, v in combinations(nodes, 2):

        # Faster lookup
        path_length = shortest_paths[u].get(v, max_path)

        if path_length <= friend_limit:
            friendship_edges[u].add(v)
            friendship_edges[v].add(u)

        elif path_length > enemy_cutoff:
            enemy_edges[u].add(v)
            enemy_edges[v].add(u)

    return friendship_edges, enemy_edges


def create_relations_hop_distance_np(G, friend_bound, enemy_bound):
    """
    Creates two graphs based on hop distance between nodes: one for "friend" relationships and one for
    "enemy" relationships based on path lengths between nodes.

    Args:
        G (networkx.Graph): The input graph to analyze. It must be undirected.
        friend_bound (float): A threshold value (between 0 and 1) determining the maximum relative hop
                              distance for an edge to be considered a "friend" edge.
        enemy_bound (float): A threshold value between 0 and 1 determining the minimum relative hop
                             distance for an edge to be considered an "enemy" edge.

    Returns:
        friendship_graph (Numpy.Array): A numpy array representing "friend" edges, where nodes are connected
                                    if their hop distance is within the `friend_bound`.
        enemy_graph (numpy.Array): A numpy array representing "enemy" edges, where nodes are connected
                                    if their hop distance is greater than `enemy_bound`.
    """
    nodes = sorted(G.nodes())
    n = len(nodes)

    if nodes != list(range(n)):
        raise ValueError("graph nodes must be labeled consecutively from 0 to n-1")

    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=np.uint8)
    D = shortest_path(A, directed=False, unweighted=True, return_predecessors=False)

    max_path = np.max(D[np.isfinite(D)])

    friend_limit = max(max_path * friend_bound, 1)
    enemy_cutoff = max_path * enemy_bound

    upper = np.triu(np.ones((n, n), dtype=bool), k=1)

    friend_mask = upper & (D <= friend_limit)
    # FEN categories must be disjoint.  On an exact shared cutoff, friendship
    # takes precedence, matching the if/elif implementation above.
    enemy_mask = upper & ~friend_mask & (D > enemy_cutoff)

    fi, fj = np.where(friend_mask)
    ei, ej = np.where(enemy_mask)

    friend_src = np.concatenate([fi, fj]).astype(np.int32)
    friend_dst = np.concatenate([fj, fi]).astype(np.int32)

    enemy_src = np.concatenate([ei, ej]).astype(np.int32)
    enemy_dst = np.concatenate([ej, ei]).astype(np.int32)

    friendship_graph = build_adj_list(friend_src, friend_dst, n)
    enemy_graph = build_adj_list(enemy_src, enemy_dst, n)

    return friendship_graph, enemy_graph


def create_graphs_hop_distance_abs(G, friend_bound, enemy_bound):
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))

    # Calculate graph diameter
    max_path = 0
    for v in shortest_paths.values():
        m = max(v.values())
        if max_path < m:
            max_path = m

    G_F = nx.Graph()
    G_E = nx.Graph()

    # Add nodes to G_F and G_E
    G_F.add_nodes_from(G.nodes())
    G_E.add_nodes_from(G.nodes())
    for u in G.nodes():
        for v in G.nodes():
            if u < v:  # Avoid duplicate edges since the graph is undirected
                if shortest_paths[u].keys().__contains__(v):
                    path_length = shortest_paths[u][v]
                else:
                    path_length = max_path
                if path_length <= 1 or path_length <= friend_bound:
                    G_F.add_edge(u, v)
                elif path_length > enemy_bound:
                    G_E.add_edge(u, v)

    return G_F, G_E


def create_graphs_kNN(agents, k, l):
    n = len(agents)
    friend_edges, enemy_edges = calculate_relationships_kNN(agents, k, l)
    friend_graph = create_graph(friend_edges, n)
    enemy_graph = create_graph(enemy_edges, n)
    return friend_graph, enemy_graph


def create_graphs_euclid(agents, friend_bound, enemy_bound):
    n = len(agents)
    friend_edges, enemy_edges = calculate_euclidian_relationships(agents, friend_bound, enemy_bound)
    friend_graph = create_graph(friend_edges, n)
    enemy_graph = create_graph(enemy_edges, n)
    return friend_graph, enemy_graph


def build_adj_list(src, dst, n):
    order = np.argsort(src)

    src = src[order]
    dst = dst[order]

    counts = np.bincount(src, minlength=n)

    splits = np.cumsum(counts[:-1])

    return np.split(dst, splits)


def create_relations_euclid(
        agents,
        friend_bound,
        enemy_bound
):
    agents = np.asarray(agents)

    n = len(agents)

    dists = pdist(agents, metric="euclidean")

    max_distance = dists.max(initial=1.0)

    normalized = dists / max_distance

    i_idx, j_idx = np.triu_indices(n, k=1)

    # Friends
    friend_mask = normalized <= friend_bound

    friend_i = i_idx[friend_mask]
    friend_j = j_idx[friend_mask]

    friend_src = np.concatenate((friend_i, friend_j))
    friend_dst = np.concatenate((friend_j, friend_i))

    # Enemies
    # FEN categories must be disjoint.  On an exact shared cutoff, friendship
    # takes precedence.
    enemy_mask = ~friend_mask & (normalized > enemy_bound)

    enemy_i = i_idx[enemy_mask]
    enemy_j = j_idx[enemy_mask]

    enemy_src = np.concatenate((enemy_i, enemy_j))
    enemy_dst = np.concatenate((enemy_j, enemy_i))

    friendship_graph = build_adj_list(
        friend_src,
        friend_dst,
        n
    )

    enemy_graph = build_adj_list(
        enemy_src,
        enemy_dst,
        n
    )

    return friendship_graph, enemy_graph


def my_make_circles(n, radius=0.2, random_state=None):
    # Parameters for the circle clusters
    n_points_per_cluster = n // 3  # Number of points in each cluster
    noise_std = 0.05  # Standard deviation of noise

    # Cluster centers
    centers = [(0.5, 0.5), (0.7, 0.3), (0.1, 0.7)]

    # Radius for clusters

    # Generate clusters
    rng = np.random.default_rng(random_state)
    data = []
    for center_x, center_y in centers:
        angles = rng.uniform(0, 2 * np.pi, n_points_per_cluster)
        x = center_x + radius * np.cos(angles) + rng.normal(0, noise_std, n_points_per_cluster)
        y = center_y + radius * np.sin(angles) + rng.normal(0, noise_std, n_points_per_cluster)
        data.append(np.column_stack((x, y)))

    data = np.vstack(data)

    truth = []
    for i in range(3):
        truth = truth + [i] * n_points_per_cluster

    return data, truth


# Create a graph from edges.
def create_graph(edges, n):
    G = nx.Graph()

    # add nodes in one call (fast)
    G.add_nodes_from(range(n))

    # ensure NumPy array (faster iteration)
    edges = np.asarray(edges, dtype=np.int32)

    # unpack efficiently
    G.add_edges_from(map(tuple, edges))

    return G


# Perform BFS from the start node and return all nodes within distance l.
def bfs(graph, start, l):
    visited = {start}
    queue = deque([(start, 0)])  # (node, current_distance)

    while queue:
        node, dist = queue.popleft()

        if dist < l:  # Only explore neighbors within distance l
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return visited


def calculate_relationships_kNN(agents, k, l):
    """
    Calculates friendship and enemy relationships based on k-nearest neighbors and a distance threshold.

    Args:
        agents (numpy.ndarray or list): A list or array of agent positions in a d-dimensional space,
                                        where each entry represents an agent's coordinates.
        k (int): The number of nearest neighbors to consider for determining friendships.
        l (float): A distance threshold to determine enmities. Agents with a Manhattan distance greater than
                  or equal to `l` are considered enemies.

    Returns:
        tuple: A tuple containing two lists:
            - friendship_edges (list of tuples): A list of pairs of agent indices that are considered friends.
            - enemy_edges (list of tuples): A list of pairs of agent indices that are considered enemies, based on
                                           the distance threshold `l`.

    """

    # Initialize variables
    n = len(agents)
    friendship_edges = []
    enemy_edges = []

    # Use NearestNeighbors to find k nearest neighbors for each agent
    nbrs = NearestNeighbors(n_neighbors=k).fit(agents)
    distances, indices = nbrs.kneighbors(agents)

    # Check pairwise distances and determine friendships and enmities
    for i in range(n):
        for j in indices[i]:
            # Check if agent i and j are within each other's k-nearest neighbors
            if i in indices[j]:
                friendship_edges.append((i, j))

            # Calculate the distance between i and j in the k-nearest neighbors graph

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(friendship_edges)

    for i in range(n):
        # Get all nodes within distance l in the graph
        for j in range(i + 1, n):
            manhattan_distance = sum(np.abs(x - y) for x, y in zip(agents[i], agents[j]))
            if manhattan_distance >= l:
                enemy_edges.append((i, j))

    return friendship_edges, enemy_edges


# Calculate friendship and enemy graphs based on the euclidian distances.
def calculate_euclidian_relationships(agents, friendship_bound, enemy_bound):
    """
    Calculates friendship and enemy relationships based on Euclidean distances between agents.

    Args:
        agents (numpy.ndarray or list): A list or array of agent positions in a d-dimensional space,
                                         where each entry represents an agent's coordinates.
        friendship_bound (float): The maximum normalized Euclidean distance below which agents are considered friends.
        enemy_bound (float): The minimum normalized Euclidean distance above which agents are considered enemies.

    Returns:
        tuple: A tuple containing two lists:
            - friendship_edges (list of tuples): A list of pairs of agent indices that are considered friends.
            - enemy_edges (list of tuples): A list of pairs of agent indices that are considered enemies, based on
                                           the Euclidean distance threshold.
    """
    n = len(agents)
    distances = np.zeros((n, n))
    friendship_edges = []
    enemy_edges = []
    max_distance = 0

    for i, j in combinations(range(n), 2):
        d = distance.euclidean(agents[i], agents[j])
        distances[i, j] = d
        distances[j, i] = d
        if d > max_distance:
            max_distance = d
    if max_distance == 0:
        max_distance = 1

    for i, j in combinations(range(n), 2):
        if distances[i, j] / max_distance <= friendship_bound:
            friendship_edges.append((i, j))
        else:
            if distances[j, i] / max_distance > enemy_bound:
                enemy_edges.append((i, j))

    return friendship_edges, enemy_edges


def calculate_euclidian_relationships_fast(
        agents,
        friendship_bound,
        enemy_bound
):
    agents = np.asarray(agents)

    # Compute all pairwise Euclidean distances efficiently
    dists = pdist(agents, metric="euclidean")

    max_distance = dists.max(initial=1.0)

    normalized = dists / max_distance

    n = len(agents)

    # Indices corresponding to condensed pdist output
    i_idx, j_idx = np.triu_indices(n, k=1)

    friendship_mask = normalized <= friendship_bound
    enemy_mask = ~friendship_mask & (normalized > enemy_bound)

    friendship_edges = list(
        zip(i_idx[friendship_mask], j_idx[friendship_mask])
    )

    enemy_edges = list(
        zip(i_idx[enemy_mask], j_idx[enemy_mask])
    )

    return friendship_edges, enemy_edges


def calculate_relationships(agents, l1, l2, k1, k2):
    """
    Calculates friendship and enemy relationships between agents based on trait differences.

    Args:
        agents (list of lists or numpy.ndarray): A list of agents, where each agent is represented by a list
                                                 or array of traits (e.g., features or attributes).
        l1 (float): The threshold for trait difference to count as a friendship condition (for each trait).
        l2 (int): The minimum number of traits where the difference is less than or equal to l1 for agents to be friends.
        k1 (float): The threshold for trait difference to count as an enmity condition (for each trait).
        k2 (int): The minimum number of traits where the difference is greater than or equal to k1 for agents to be enemies.

    Returns:
        tuple: A tuple containing two lists:
            - friendship_edges (list of tuples): A list of pairs of agent indices that are considered friends
                                                  based on the trait difference conditions.
            - enemy_edges (list of tuples): A list of pairs of agent indices that are considered enemies based
                                            on the trait difference conditions.
    """
    n = len(agents)
    friendship_edges = []
    enemy_edges = []

    for i, j in combinations(range(n), 2):
        diff = [abs(agents[i][dim] - agents[j][dim]) for dim in range(len(agents[0]))]
        friend_condition = sum(d <= l1 for d in diff) >= l2
        enemy_condition = sum(d >= k1 for d in diff) >= k2

        if friend_condition:
            friendship_edges.append((i, j))
        if enemy_condition:
            enemy_edges.append((i, j))

    return friendship_edges, enemy_edges


# Generate n agents with d-dimensional trait vectors.
# Each trait is an integer between 0 and 9.
def generate_agents(n, d):
    return [tuple(random.random() for _ in range(d)) for _ in range(n)]


def randomize_graph_node_labels(G, truth=None):
    r = np.arange(len(G))
    np.random.shuffle(r)
    G_r = [G[r[i]] for i in range(len(r))]
    if truth is not None:
        truth_r = [truth[r[i]] for i in range(len(r))]
        return G_r, truth_r

    return G_r, None


def randomize_graph_pos_labels(G, truth=None):
    r = np.arange(len(G))
    np.random.shuffle(r)
    G_r = [G[r[i]] for i in range(len(r))]
    if truth is not None:
        truth_r = [truth[r[i]] for i in range(len(r))]
        return G_r, truth_r

    return G_r, None
