
import random
from itertools import combinations

from scipy.spatial import distance

import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from collections import deque


def generate_graph(n, k, p, q):
    """ Creates k disjoint Erdos-Renyi graphs of size n and edge probability p, then connects vertices from different components with probability q"""
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





def create_graphs_hop_distance(G,friend_bound,enemy_bound):
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

    return data

def create_graph(edges, n):
    """Create a graph from edges."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G

def bfs(graph, start, l):
    """Perform BFS from the start node and return all nodes within distance l."""
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
    Calculate friendship and enemy graphs based on k-nearest neighbors and distance threshold l.

    Parameters:
    - agents: List of points in d-dimensional space (numpy array).
    - k: Number of nearest neighbors to consider.
    - l: Distance threshold for enmity.

    Returns:
    - friendship_edges: List of pairs of agent indices that are friends.
    - enemy_edges: List of pairs of agent indices that are enemies.
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

def calculate_euclidian_relationships(agents,friendship_bound,enemy_bound):
    """Calculate friendship and enemy graphs based on the euclidian distances."""
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
    """Calculate friendship and enemy graphs based on the trait difference rules."""
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

def generate_agents(n, d):
    """Generate n agents with d-dimensional trait vectors.
    Each trait is an integer between 0 and 9."""
    return [tuple(random.random() for _ in range(d)) for _ in range(n)]


def randomize_graph_node_labels(G,truth = None):
    new_nodes = list(G.nodes())

    r = np.arange(len(new_nodes))
    np.random.shuffle(r)
    G = nx.relabel_nodes(G, {i: r[i] for i in range(len(r))})
    if truth is not None:
        truth_r = [truth[r[i]] for i in range(len(new_nodes))]
        return G,truth_r

    return G,None
