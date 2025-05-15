
import random
from itertools import combinations

from scipy.spatial import distance

import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from collections import deque


def generate_graph(n, k, p, q):
    """
    Generates a graph consisting of k Erdős-Rényi subgraphs with random inter-subgraph edges.

    Args:
        n (int): The number of nodes in each subgraph.
        k (int): The number of subgraphs to generate.
        p (float): The probability of creating an edge between two nodes within the same subgraph.
        q (float): The probability of creating an edge between nodes from different subgraphs.

    Returns:
        tuple: A tuple containing:
            - G (networkx.Graph): The generated graph with inter- and intra-subgraph edges.
            - truth (list): A list where each element corresponds to the community label of the node
              (in the form of a list of k values repeated n times for each subgraph).
    """
    G = nx.Graph()
    subgraphs = []

    truth = []

    for i in range(k):
        t = [i]*n
        truth += t

    for i in range(k):
        subgraph = nx.erdos_renyi_graph(n, p)
        mapping = {node: node + i * n for node in subgraph.nodes()}
        nx.relabel_nodes(subgraph, mapping, copy=False)
        G = nx.compose(G, subgraph)
        subgraphs.append(set(mapping.values()))

    # Add random inter-subgraph edges with probability q
    for i in range(k):
        for j in range(i + 1, k):
            for u in subgraphs[i]:
                for v in subgraphs[j]:
                    if random.random() < q:
                        G.add_edge(u, v)

    return G, truth



import random

def permute_graph_with_truth(G, truth):
    """

    Permutes node labels in a graph and updates the truth list accordingly.
    
    Parameters:
    - G: networkx.Graph with nodes labeled from 0 to n-1
    - truth: list of length n, where truth[i] is the cluster label of node i

    Returns:
    - G_permuted: graph with permuted node labels
    - truth_permuted: list where truth_permuted[i] is the label of node i in G_permuted
    """
    n = len(truth)
    assert set(G.nodes()) == set(range(n)), "Graph nodes must be labeled 0 to n-1"

    perm = list(range(n))
    random.shuffle(perm)  # perm[i] is the new label of node i
    inverse_perm = [0] * n
    for i, p in enumerate(perm):
        inverse_perm[p] = i  # inverse_perm[new_id] = old_id

    mapping = {i: perm[i] for i in range(n)}
    G_permuted = nx.relabel_nodes(G, mapping)

    if truth is not None:
        truth_permuted = [truth[inverse_perm[i]] for i in range(n)]

        return G_permuted, truth_permuted
    return G_permuted,None

def create_graphs_hop_distance(G,friend_bound,enemy_bound):
    """
    Creates two graphs based on hop distance between nodes: one for "friend" relationships and one for
    "enemy" relationships based on path lengths between nodes.

    Args:
        G (networkx.Graph): The input graph to analyze. It must be undirected.
        friend_bound (float): A threshold value (between 0 and 1) determining the maximum relative hop
                              distance for an edge to be considered a "friend" edge.
        enemy_bound (float): A threshold value (greater than 1) determining the minimum relative hop
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
                if path_length <= 1 or path_length <= max_path*friend_bound:
                    G_F.add_edge(u, v)
                elif path_length >= max_path*enemy_bound:
                    G_E.add_edge(u, v)

    return G_F, G_E

def create_graphs_hop_distance_abs(G,friend_bound,enemy_bound):
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
                elif path_length >= enemy_bound:
                    G_E.add_edge(u, v)

    return G_F, G_E

def create_graphs_kNN(agents,k,l):
    n = len(agents)
    friend_edges, enemy_edges = calculate_relationships_kNN(agents, k, l)
    friend_graph = create_graph(friend_edges, n)
    enemy_graph = create_graph(enemy_edges, n)
    return friend_graph, enemy_graph

def create_graphs_euclid(agents,friend_bound,enemy_bound):
    n = len(agents)
    friend_edges, enemy_edges = calculate_euclidian_relationships(agents, friend_bound, enemy_bound)
    friend_graph = create_graph(friend_edges, n)
    enemy_graph = create_graph(enemy_edges, n)
    return friend_graph, enemy_graph

def my_make_circles(n, radius=0.2):
    # Parameters for the circle clusters
    n_points_per_cluster = n // 3  # Number of points in each cluster
    noise_std = 0.05  # Standard deviation of noise

    # Cluster centers
    centers = [(0.5, 0.5), (0.7, 0.3), (0.1, 0.7)]

    # Radius for clusters

    # Generate clusters
    data = []
    for center_x, center_y in centers:
        angles = np.random.uniform(0, 2 * np.pi, n_points_per_cluster)
        x = center_x + radius * np.cos(angles) + np.random.normal(0, noise_std, n_points_per_cluster)
        y = center_y + radius * np.sin(angles) + np.random.normal(0, noise_std, n_points_per_cluster)
        data.append(np.column_stack((x, y)))

    data = np.vstack(data)

    truth = []
    for i in range(3):
        truth = truth+ [i]*n_points_per_cluster

    return data, truth

# Create a graph from edges.
def create_graph(edges, n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
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
    print(friendship_edges)
    print(enemy_edges)
    return friendship_edges, enemy_edges

# Calculate friendship and enemy graphs based on the euclidian distances.
def calculate_euclidian_relationships(agents,friendship_bound,enemy_bound):
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
    distances = np.zeros((n,n))
    friendship_edges = []
    enemy_edges = []
    max_distance = 0

    for i, j in combinations(range(n), 2):
        d = distance.euclidean(agents[i], agents[j])
        distances[i,j] = d
        distances[j,i] = d
        if d > max_distance:
            max_distance = d
    if max_distance == 0:
        max_distance = 1
    for i, j in combinations(range(n), 2):
        if distances[i,j]/max_distance <= friendship_bound:
            friendship_edges.append((i, j))
        else:
            if distances[j,i]/max_distance >= enemy_bound:
                enemy_edges.append((i, j))

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





def randomize_graph_pos_labels(G,truth = None):
    r = np.arange(len(G))
    np.random.shuffle(r)
    G_r = [G[r[i]] for i in range(len(r))]
    if truth is not None:
        truth_r = [truth[r[i]] for i in range(len(r))]
        return G_r,truth_r

    return G_r,None