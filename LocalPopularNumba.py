import math
from collections import defaultdict
import time

import networkx as nx
import numpy as np
from numba import njit

from GraphFunctions import create_graphs_euclid, create_graphs_hop_distance, create_relations_euclid, \
    create_relations_hop_distance, create_relations_hop_distance_np
from sklearn.metrics import rand_score, silhouette_score, davies_bouldin_score


def locally_popular_clustering_with_euclid_graphs_numba(agents, f, e, initial_clusters=None, always_allow_exit=False,
                                                        print_steps=False, mode='B', max_coalitions=0,
                                                        use_first_move=False, pre=None,local_stable=False):
    """
    Creates initial clustering, and friend/enemy graphs before starting the locally popular algorithm
    Args:
        agents (list): List of agent identifiers.
        initial_clusters (int): Number of initial clusters.
        always_allow_exit (bool, optional): If True, agents can always form new singleton clusters. Defaults to False.
        print_steps (bool, optional): If True, prints details of each move made by agents. Defaults to False.
        mode (str, optional): Determines the move selection rule ('B', 'F', or 'E'). Defaults to 'B'.
        max_coalitions (int, optional): Maximum allowed number of clusters. If 0, no limit is enforced. Defaults to 0.
        use_first_move (bool, optional): If True, uses the first improving move instead of the best move. Defaults to False.
        pre (function, optional): Function used to create an initial clustering of the agents. Defaults to None.

    Returns:
        dict: A mapping from each agent to their final cluster ID after reaching local stability.

    """
    start = time.time()
    if initial_clusters is None:
        initial_clusters = len(agents)

    if pre is not None:
        initial_labels = pre(agents, initial_clusters)
        initial_clustering = {i: initial_labels[i] for i in range(len(agents))}
    else:
        initial_clustering = {i: i % initial_clusters for i in range(len(agents))}

    G_F, G_E = create_relations_euclid(agents, f, e)

    if max_coalitions == 0:
        max_coalitions = initial_clusters

    end = time.time()
    # print(f'Elapsed (preprod): {end - start:.2f} seconds')
    return locally_popular_clustering_numba(agents, G_F, G_E, initial_clustering, always_allow_exit, print_steps, mode,
                                            max_coalitions, use_first_move,local_stable)


def locally_popular_clustering_with_hop_distance_numba(agents, f, e, initial_clusters=None, always_allow_exit=False,
                                                       print_steps=False, mode='B', max_coalitions=0,
                                                       use_first_move=False, pre=None,local_stable=False):
    """
    Creates initial clustering, and friend/enemy graphs before starting the locally popular algorithm
    Args:
        agents (list): List of agent identifiers.
        initial_clusters (int): Number of initial clusters.
        always_allow_exit (bool, optional): If True, agents can always form new singleton clusters. Defaults to False.
        print_steps (bool, optional): If True, prints details of each move made by agents. Defaults to False.
        mode (str, optional): Determines the move selection rule ('B', 'F', or 'E'). Defaults to 'B'.
        max_coalitions (int, optional): Maximum allowed number of clusters. If 0, no limit is enforced. Defaults to 0.
        use_first_move (bool, optional): If True, uses the first improving move instead of the best move. Defaults to False.
        pre (function, optional): Function used to create an initial clustering of the agents. Defaults to None.

    Returns:
        dict: A mapping from each agent to their final cluster ID after reaching local stability.

    """
    if initial_clusters is None:
        initial_clusters = len(agents)

    if pre is not None:
        p = pre(agents, initial_clusters)
        initial_labels = extract_labels_from_communities(p.communities)
        initial_clustering = {i: initial_labels[i] for i in range(len(agents))}
    else:
        initial_clustering = {i: i % initial_clusters for i in range(len(agents))}

    if max_coalitions == 0:
        max_coalitions = initial_clusters

    l = len(list(set(initial_clustering.values())))
    if max_coalitions < l:
        max_coalitions = l

    G_F, G_E = create_relations_hop_distance_np(agents, f, e)
    return locally_popular_clustering_numba(agents, G_F, G_E, initial_clustering, always_allow_exit, print_steps, mode,
                                            max_coalitions, use_first_move,local_stable)


def locally_popular_clustering_numba(agents, friends, enemies, initial_clustering, always_allow_exit=False,
                                     print_steps=False, mode='B', max_coalitions=0, use_first_move=False,local_stable=False):
    n = len(agents)

    clustering = np.zeros(n, dtype=np.int32)

    for v in range(n):
        clustering[v] = initial_clustering[v]

    if mode == 'B':
        mode_int = 0
    elif mode == 'F':
        mode_int = 1
    elif mode == 'E':
        mode_int = 2
    else:
        raise ValueError("mode must be B/F/E")

    clustering, num_moves = solve_with_numba(
        clustering,
        friends,
        enemies,
        max_coalitions,
        1_000_000,
        mode_int,
        always_allow_exit,
        local_stable
    )

    result = {
        v: int(clustering[v])
        for v in range(n)
    }

    # print(f"NumPy + numba swaps: {num_moves}")

    return result


@njit
def solve_with_numba(
        clustering,
        friends,
        enemies,
        max_coalitions,
        max_iter,
        mode,
        allow_exit,
        local_stable = False
):
    n = len(clustering)

    max_clusters = max(math.floor(max_coalitions * 2), 20)

    if max_clusters > n:
        max_clusters = n

    cluster_sizes = np.zeros(max_clusters, dtype=np.int32)

    for i in range(n):
        cluster_sizes[clustering[i]] += 1

    next_cluster_id = np.max(clustering) + 1

    friend_counts, enemy_counts = initialize_counts(
        n,
        max_clusters,
        clustering,
        friends,
        enemies
    )

    num_moves = 0

    for _ in range(max_iter):

        agent, target, vote = find_best_move(
            n,
            max_clusters,
            clustering,
            cluster_sizes,
            friend_counts,
            enemy_counts,
            allow_exit,
            mode,
            local_stable
        )

        if agent == -1:
            break

        next_cluster_id = apply_move(
            agent,
            target,
            clustering,
            cluster_sizes,
            friend_counts,
            enemy_counts,
            friends,
            enemies,
            next_cluster_id
        )

        num_moves += 1

    return clustering, num_moves


@njit
def apply_move(
        v,
        target,
        clustering,
        cluster_sizes,
        friend_counts,
        enemy_counts,
        friends,
        enemies,
        next_cluster_id
):
    old_cluster = clustering[v]

    # remove from old cluster
    cluster_sizes[old_cluster] -= 1

    # create singleton cluster
    if target == -2:

        new_cluster = next_cluster_id
        clustering[v] = new_cluster
        cluster_sizes[new_cluster] = 1

        next_cluster_id += 1

    else:

        new_cluster = target
        clustering[v] = new_cluster
        cluster_sizes[new_cluster] += 1

    nbrs = friends[v]

    for i in range(len(nbrs)):
        u = nbrs[i]

        friend_counts[u, old_cluster] -= 1
        friend_counts[u, new_cluster] += 1

    nbrs = enemies[v]

    for i in range(len(nbrs)):
        u = nbrs[i]

        enemy_counts[u, old_cluster] -= 1
        enemy_counts[u, new_cluster] += 1

    return next_cluster_id


@njit
def find_best_move(
        n,
        n_clusters,
        clustering,
        cluster_sizes,
        friend_counts,
        enemy_counts,
        allow_exit,
        mode,
        local_stable=False
):
    """
    mode:
        0 -> B
        1 -> F
        2 -> E
    """

    best_agent = -1
    best_cluster = -1
    best_vote = 0

    f_weight = 1
    e_weight = 1

    if mode == 1:
        f_weight = n

    if mode == 2:
        e_weight = n

    for v in range(n):

        current = clustering[v]

        f_current = friend_counts[v, current]
        e_current = enemy_counts[v, current]

        # evaluate moves to existing clusters
        for c in range(n_clusters):

            if c == current:
                continue

            if cluster_sizes[c] == 0:
                continue

            f_target = friend_counts[v, c]
            e_target = enemy_counts[v, c]

            vote = (
                    f_target
                    + e_current
                    - f_current
                    - e_target
            )

            if vote < 0:
                continue

            v_vote = (
                    f_weight * f_target
                    + e_weight * e_current
                    - f_weight * f_current
                    - e_weight * e_target
            )

            if v_vote > 0:
                v_vote = 1
            elif v_vote < 0:
                v_vote = -1
                if local_stable:
                    v_vote = -vote
            else:
                v_vote = 0

            vote += v_vote

            if vote > best_vote:
                best_vote = vote
                best_agent = v
                best_cluster = c

        # evaluate exit
        if allow_exit:

            vote = e_current - f_current

            v_vote = (
                    e_weight * e_current
                    - f_weight * f_current
            )

            if v_vote > 0:
                v_vote = 1
            elif v_vote < 0:
                v_vote = -1
                if local_stable:
                    v_vote = -vote
            else:
                v_vote = 0

            vote += v_vote

            if vote > best_vote:
                best_vote = vote
                best_agent = v
                best_cluster = -2  # exit marker

    return best_agent, best_cluster, best_vote


@njit
def initialize_counts(n, n_clusters, clustering,
                      friends, enemies):
    """
    friend_counts[v, c] =
        number of friends of v in cluster c

    enemy_counts[v, c] =
        number of enemies of v in cluster c
    """

    friend_counts = np.zeros((n, n_clusters), dtype=np.int32)
    enemy_counts = np.zeros((n, n_clusters), dtype=np.int32)

    for v in range(n):

        # friends
        for i in range(len(friends[v])):
            u = friends[v][i]
            c = clustering[u]
            friend_counts[v, c] += 1

        # enemies
        for i in range(len(enemies[v])):
            u = enemies[v][i]
            c = clustering[u]
            enemy_counts[v, c] += 1

    return friend_counts, enemy_counts


def extract_labels_from_communities(communities):
    """
    Converts a list of communities into a dictionary mapping nodes to community labels.

    Args:
        communities (list of set): A list where each set contains nodes belonging to the same community.

    Returns:
        dict: A dictionary mapping each node to its corresponding community index (label).
    """
    d = dict()
    for i in range(len(communities)):
        c = communities[i]
        for node in c:
            d[node] = i
    return d
