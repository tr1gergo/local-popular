import numpy as np
from numba import njit

from GraphFunctions import create_relations_euclid, \
    create_relations_hop_distance_np


def normalize_cluster_labels(labels):
    """Map arbitrary baseline labels (including DBSCAN's -1) to 0,...,k-1."""
    mapping = {}
    normalized = np.empty(len(labels), dtype=np.int32)
    for index, label in enumerate(labels):
        key = label.item() if hasattr(label, 'item') else label
        if key not in mapping:
            mapping[key] = len(mapping)
        normalized[index] = mapping[key]
    return normalized


def locally_popular_clustering_with_euclid_graphs_numba(agents, f_bound, e_bound, initial_clusters=None,
                                                        mode='B', max_coalitions=0, pre=None, local_stable=False,
                                                        return_diagnostics=False, max_iter=1_000_000):
    """
    Creates initial clustering, and friend/enemy graphs before starting the locally popular algorithm
    Args:
        agents (list): List of agent identifiers.
        f_bound (float): float between 0 and 1. Points at distance at most diameter*f_bound are friends.
        e_bound (float): float between 0 and 1. If two points have a longer distance than diameter*e_bound they are considered enemies.
        initial_clusters (int): Number of initial clusters.
        mode (str, optional): Determines the move selection rule ('B', 'F', or 'E'). Defaults to 'B'.
        max_coalitions (int, optional): Coalition-label capacity. If 0, use min(n, max(2k, 20)).
        pre (function, optional): Function used to create an initial clustering of the agents. Defaults to None.
        local_stable (bool, optional): If True, uses the local stable clustering instead of local popular. Defaults to False.

    Returns:
        dict or (dict, dict): Final labels, optionally with move/convergence diagnostics.

    """
    if initial_clusters is None:
        initial_clusters = len(agents)

    if pre is not None:
        initial_labels = normalize_cluster_labels(pre(agents, initial_clusters))
        initial_clustering = {i: initial_labels[i] for i in range(len(agents))}
    else:
        initial_clustering = {i: i % initial_clusters for i in range(len(agents))}

    G_F, G_E = create_relations_euclid(agents, f_bound, e_bound)

    if max_coalitions == 0:
        max_coalitions = min(len(agents), max(2 * initial_clusters, 20))

    return locally_popular_clustering_numba(agents, G_F, G_E, initial_clustering, mode,
                                            max_coalitions, local_stable, return_diagnostics, max_iter)


def locally_popular_clustering_with_hop_distance_numba(agents, f_bound, e_bound, initial_clusters=None,
                                                       mode='B', max_coalitions=0, pre=None, local_stable=False,
                                                       return_diagnostics=False, max_iter=1_000_000):
    """
    Creates initial clustering, and friend/enemy graphs before starting the locally popular algorithm
    Args:
        agents (list): List of agent identifiers.
        f_bound (float): float between 0 and 1. Nodes at hop distance at most diameter*f_bound (and at least the one-hop neighborhood) are friends.
        e_bound (float): float between 0 and 1. Nodes beyond the enemy cutoff are enemies.
        initial_clusters (int): Number of initial clusters.
        mode (str, optional): Determines the move selection rule ('B', 'F', or 'E'). Defaults to 'B'.
        max_coalitions (int, optional): Coalition-label capacity. If 0, use min(n, max(2k, 20)).
        pre (function, optional): Function used to create an initial clustering of the agents. Defaults to None.
        local_stable (bool, optional): If True, uses the local stable clustering instead of local popular. Defaults to False.

    Returns:
        dict or (dict, dict): Final labels, optionally with move/convergence diagnostics.

    """
    if initial_clusters is None:
        initial_clusters = len(agents)

    if pre is not None:
        p = pre(agents, initial_clusters)
        initial_labels = extract_labels_from_communities(p.communities)
        initial_clustering = {i: initial_labels[i] for i in range(len(agents))}
    else:
        initial_clustering = {i: i % initial_clusters for i in range(len(agents))}

    l = len(list(set(initial_clustering.values())))
    if max_coalitions == 0:
        max_coalitions = min(len(agents), max(2 * l, 20))
    else:
        max_coalitions = max(max_coalitions, l)

    G_F, G_E = create_relations_hop_distance_np(agents, f_bound, e_bound)
    return locally_popular_clustering_numba(agents, G_F, G_E, initial_clustering, mode,
                                            max_coalitions, local_stable, return_diagnostics, max_iter)


def locally_popular_clustering_numba(agents, friends, enemies, initial_clustering,
                                     mode='B', max_coalitions=0,
                                     local_stable=False, return_diagnostics=False,
                                     max_iter=1_000_000):
    n = len(agents)

    clustering = np.zeros(n, dtype=np.int32)

    for v in range(n):
        clustering[v] = initial_clustering[v]

    if np.min(clustering) < 0:
        raise ValueError("cluster labels must be nonnegative")

    required_capacity = int(np.max(clustering)) + 1
    if max_coalitions == 0:
        initial_count = len(np.unique(clustering))
        max_coalitions = min(n, max(2 * initial_count, 20))
    max_coalitions = min(n, max(max_coalitions, required_capacity))

    if mode == 'B':
        mode_int = 0
    elif mode == 'F':
        mode_int = 1
    elif mode == 'E':
        mode_int = 2
    else:
        raise ValueError("mode must be B/F/E")

    clustering, num_moves, converged = solve_with_numba(
        clustering,
        friends,
        enemies,
        max_coalitions,
        max_iter,
        mode_int,
        local_stable
    )

    # Translating output into dict
    result = {
        v: int(clustering[v])
        for v in range(n)
    }

    if return_diagnostics:
        diagnostics = {
            'moves': int(num_moves),
            'converged': bool(converged),
            'coalition_capacity': int(max_coalitions),
            'final_coalitions': len(set(result.values())),
        }
        return result, diagnostics
    return result


@njit(cache=True)
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

    max_clusters = min(max_coalitions, n)

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

    converged = False

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
            converged = True
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


    return clustering, num_moves, converged


@njit(cache=True)
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


@njit(cache=True)
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

    # All empty target labels represent the same singleton-creation move.  It
    # suffices to inspect the first one.  A singleton agent is not moved to an
    # empty label because that would only relabel the same partition.
    empty_cluster = -1
    for c in range(n_clusters):
        if cluster_sizes[c] == 0:
            empty_cluster = c
            break

    for v in range(n):

        current = clustering[v]

        f_current = friends_in_coalition[v, current]
        e_current = enemies_in_coalition[v, current]


        for c in range(n_clusters):

            if c == current:
                continue

            if cluster_sizes[c] == 0:
                if c != empty_cluster or cluster_sizes[current] == 1:
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
            if local_stable and v_vote <= 0:
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


@njit(cache=True)
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
