import math
import time

import numpy as np
from numba import njit

from GraphFunctions import create_relations_euclid, \
    create_relations_hop_distance_np


def locally_popular_clustering_with_euclid_graphs_numba(agents, f_bound, e_bound, initial_clusters=None,
                                                        mode='B', max_coalitions=0, pre=None, local_stable=False):
    """
    Creates initial clustering, and friend/enemy graphs before starting the locally popular algorithm
    Args:
        agents (list): List of agent identifiers.
        f_bound (float): float between 0 and 1. If two points have a shorter distance then diameter*f_bound they are considered friends.
        e_bound (float): float between 0 and 1. If two points have a longer distance then diameter*e_bound they are considered friends.
        initial_clusters (int): Number of initial clusters.
        mode (str, optional): Determines the move selection rule ('B', 'F', or 'E'). Defaults to 'B'.
        max_coalitions (int, optional): Maximum allowed number of clusters. If 0, no limit is enforced. Defaults to 0.
        pre (function, optional): Function used to create an initial clustering of the agents. Defaults to None.
        local_stable (bool, optional): If True, uses the local stable clustering instead of local popular. Defaults to False.

    Returns:
        dict: A mapping from each agent to their final cluster ID after reaching local stability.

    """
    if initial_clusters is None:
        initial_clusters = len(agents)

    if pre is not None:
        initial_labels = pre(agents, initial_clusters)
        initial_clustering = {i: initial_labels[i] for i in range(len(agents))}
    else:
        initial_clustering = {i: i % initial_clusters for i in range(len(agents))}

    G_F, G_E = create_relations_euclid(agents, f_bound, e_bound)

    if max_coalitions == 0:
        max_coalitions = initial_clusters

    return locally_popular_clustering_numba(agents, G_F, G_E, initial_clustering, mode,
                                            max_coalitions, local_stable)


def locally_popular_clustering_with_hop_distance_numba(agents, f_bound, e_bound, initial_clusters=None,
                                                       mode='B', max_coalitions=0, pre=None, local_stable=False):
    """
    Creates initial clustering, and friend/enemy graphs before starting the locally popular algorithm
    Args:
        agents (list): List of agent identifiers.
        f_bound (float): float between 0 and 1. If two points have a shorter distance then diameter*f_bound they are considered friends.
        e_bound (float): float between 0 and 1. If two points have a longer distance then diameter*e_bound they are considered friends.
        initial_clusters (int): Number of initial clusters.
        mode (str, optional): Determines the move selection rule ('B', 'F', or 'E'). Defaults to 'B'.
        max_coalitions (int, optional): Maximum allowed number of clusters. If 0, no limit is enforced. Defaults to 0.
        pre (function, optional): Function used to create an initial clustering of the agents. Defaults to None.
        local_stable (bool, optional): If True, uses the local stable clustering instead of local popular. Defaults to False.

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

    G_F, G_E = create_relations_hop_distance_np(agents, f_bound, e_bound)
    return locally_popular_clustering_numba(agents, G_F, G_E, initial_clustering, mode,
                                            max_coalitions, local_stable)


def locally_popular_clustering_numba(agents, friends, enemies, initial_clustering,
                                     mode='B', max_coalitions=0,
                                     local_stable=False):
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
        local_stable
    )

    # Translating output into dict
    result = {
        v: int(clustering[v])
        for v in range(n)
    }

    return result


@njit
def solve_with_numba(
        clustering,
        friends,
        enemies,
        max_coalitions,
        max_iter,
        mode,
        local_stable=False
):
    n = len(clustering)

    # Determining the maximum numbers of allowed clusters

    max_clusters = max(math.floor(max_coalitions * 2), 20)

    if max_clusters > n:
        max_clusters = n

    cluster_sizes = np.zeros(max_clusters, dtype=np.int32)

    for i in range(n):
        cluster_sizes[clustering[i]] += 1

    friends_in_coalition, enemies_in_coalition = initialize_coalition_counters(
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
            friends_in_coalition,
            enemies_in_coalition,
            mode,
            local_stable
        )

        if agent == -1:
            break

        apply_move(
            agent,
            target,
            clustering,
            cluster_sizes,
            friends_in_coalition,
            enemies_in_coalition,
            friends,
            enemies
        )

        num_moves += 1

    return clustering, num_moves


@njit
def apply_move(
        v,
        target,
        clustering,
        cluster_sizes,
        friends_in_coalition,
        enemies_in_coalition,
        friends,
        enemies
):
    old_cluster = clustering[v]

    # remove from old cluster
    cluster_sizes[old_cluster] -= 1

    # add to new cluster
    new_cluster = target
    clustering[v] = new_cluster
    cluster_sizes[new_cluster] += 1

    # updating friend/enemy trackers
    nbrs = friends[v]
    for i in range(len(nbrs)):
        u = nbrs[i]

        friends_in_coalition[u, old_cluster] -= 1
        friends_in_coalition[u, new_cluster] += 1

    nbrs = enemies[v]
    for i in range(len(nbrs)):
        u = nbrs[i]

        enemies_in_coalition[u, old_cluster] -= 1
        enemies_in_coalition[u, new_cluster] += 1

    return


@njit
def find_best_move(
        n,
        n_clusters,
        clustering,
        cluster_sizes,
        friends_in_coalition,
        enemies_in_coalition,
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

        f_current = friends_in_coalition[v, current]
        e_current = enemies_in_coalition[v, current]


        for c in range(n_clusters):

            if c == current:
                continue

            if cluster_sizes[c] == 0:
                continue

            f_target = friends_in_coalition[v, c]
            e_target = enemies_in_coalition[v, c]

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
            # if local_stable=True, v has to prefer the swap
            if local_stable & v_vote <= 0:
                continue

            if v_vote > 0:
                v_vote = 1
            elif v_vote < 0:
                v_vote = -1

            else:
                v_vote = 0

            vote += v_vote

            if vote > best_vote:
                best_vote = vote
                best_agent = v
                best_cluster = c

    return best_agent, best_cluster, best_vote


@njit
def initialize_coalition_counters(n, n_clusters, clustering,
                                  friends, enemies):
    """
    Returns the following two matrices
    friends_in_coalition[v, c] =
        number of friends of v in coalition c

    enemies_in_coalition[v, c] =
        number of enemies of v in coalition c
    """

    friends_in_coalition = np.zeros((n, n_clusters), dtype=np.int32)
    enemies_in_coalition = np.zeros((n, n_clusters), dtype=np.int32)

    for v in range(n):

        # friends
        for i in range(len(friends[v])):
            u = friends[v][i]
            c = clustering[u]
            friends_in_coalition[v, c] += 1

        # enemies
        for i in range(len(enemies[v])):
            u = enemies[v][i]
            c = clustering[u]
            enemies_in_coalition[v, c] += 1

    return friends_in_coalition, enemies_in_coalition


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
