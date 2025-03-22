import random
from itertools import combinations
import networkx as nx
from sortedcontainers import SortedDict
from sortedcontainers import SortedList

import numpy as np
from sklearn.neighbors import NearestNeighbors


### EZ még rossz : ha egy kilépés dominál, az nem lesz váltaztatva


def generate_agents(n, d):
    """Generate n agents with d-dimensional trait vectors.
    Each trait is an integer between 0 and 9."""
    return [tuple(random.random() for _ in range(d)) for _ in range(n)]


import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from collections import deque


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
    G.add_edges_from(friend_edges)

    for i in range(n):
        # Get all nodes within distance l in the graph
        for j in range(i + 1, n):
            manhattan_distance = np.sum(np.abs(x - y) for x, y in zip(agents[i], agents[j]))
            if manhattan_distance >= l:
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


def create_graph(edges, n):
    """Create a graph from edges."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G


from collections import defaultdict


def locally_popular_clustering(agents, friendship_graph, enemy_graph, initial_clustering, allow_exit=False,
                               print_steps=False):
    """Perform clustering to achieve local popularity."""
    # Initialize clustering and cluster-to-agents mapping
    clustering = initial_clustering.copy()  # Map agent to cluster ID
    cluster_to_agents = defaultdict(set)
    for agent, cluster in clustering.items():
        cluster_to_agents[cluster].add(agent)

    # Precompute (f, e) values for each agent in all clusters
    f_e_values = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # {agent: {cluster: [f, -e]}}
    for v in range(len(agents)):
        for cluster in cluster_to_agents:
            friends_in_cluster = sum(
                1 for neighbor in friendship_graph.neighbors(v) if neighbor in cluster_to_agents[cluster])
            enemies_in_cluster = sum(
                1 for neighbor in enemy_graph.neighbors(v) if neighbor in cluster_to_agents[cluster])
            f_e_values[v][cluster] = [friends_in_cluster, enemies_in_cluster]

    stable = False
    num_switches = 0
    while not stable:
        stable = True

        for v in range(len(agents)):
            current_cluster = clustering[v]
            [f_current, e_current] = f_e_values[v][current_cluster]
            best_move = None
            best_vote = 0  # Track best improvement (f, -e)
            block = False
            # Evaluate moves to existing clusters
            for candidate_cluster in cluster_to_agents:
                if candidate_cluster == current_cluster:
                    continue

                [f_target, e_target] = f_e_values[v][candidate_cluster]
                vote = 0
                current_score = f_current - e_current
                new_score = f_target - e_target
                if new_score > current_score + 1 or (
                        new_score >= current_score and new_score + f_target > current_score + f_current):
                    # Improvement detected
                    vote = f_target - e_target - f_current + e_current
                    block = True
                if f_target > f_current or (f_target == f_current and e_target < e_current):
                    vote += 1  # v improves
                if block and vote > best_vote:
                    best_move = candidate_cluster
                    best_vote = vote

            # Evaluate exit (if allowed)
            if allow_exit:
                vote = 0
                if 0 > f_current - e_current + 1 or (0 >= 2 * f_current - e_current + 1 and 0 >= f_current - e_current):
                    vote = -f_current + e_current
                    block = True
                if f_current == 0 and e_current > 0:
                    vote += 1  # v improves
                if block and vote > best_vote:
                    best_move = "exit"
                    best_vote = vote

            # Make the best move if it's an improvement
            if best_move is not None:
                stable = False

                # Update cluster mappings
                cluster_to_agents[current_cluster].remove(v)
                if best_move == "exit":  # Create a new cluster for this agent
                    new_cluster_id = max(cluster_to_agents.keys()) + 1
                    clustering[v] = new_cluster_id
                    cluster_to_agents[new_cluster_id].add(v)
                    best_move = new_cluster_id
                else:
                    clustering[v] = best_move
                    cluster_to_agents[best_move].add(v)

                # Update (f, e) values for affected agents
                for neighbor in friendship_graph.neighbors(v):
                    if neighbor in clustering:  # Only update if neighbor exists in the clustering
                        f_e_values[neighbor][current_cluster][0] -= 1  # Friend leaves
                        f_e_values[neighbor][best_move][0] += 1  # Friend joins

                for neighbor in enemy_graph.neighbors(v):
                    if neighbor in clustering:  # Only update if neighbor exists in the clustering
                        f_e_values[neighbor][current_cluster][1] -= 1  # Enemy leaves
                        f_e_values[neighbor][best_move][1] += 1  # Enemy joins
                num_switches += 1
                if print_steps:
                    print(f"{v} swithces to {best_move}")
                # print(f"{f_e_values[v][current_cluster]}")
                # print(f"{f_e_values[v][best_move]}")
                break
    print(num_switches)
    return clustering


# Parameters
n = 100  # Number of agents
d = 2  # Number of traits

l1, l2 = 0.1, 2  # l2 coords must be within l1 dist
k1, k2 = 0.2, 1  # k2 coords must be at least k1 dist

# Generate agents and graphs
agents = generate_agents(n, d)
friend_edges, enemy_edges = calculate_relationships(agents, l1, l2, k1, k2)
friend_graph = create_graph(friend_edges, n)
enemy_graph = create_graph(enemy_edges, n)
initial_clustering = {i: i % 5 for i in range(len(agents))}

# Perform clustering
final_clustering = locally_popular_clustering(agents, friend_graph, enemy_graph, initial_clustering)

# Output results
print("Final Clustering:", final_clustering)