
from collections import defaultdict
import time

import networkx as nx

from GraphFunctions import create_relations_hop_distance, \
    create_relations_euclid
from sklearn.metrics import rand_score, silhouette_score, davies_bouldin_score


def locally_popular_clustering_with_euclid_graphs(agents,f,e, initial_clusters=None, always_allow_exit=False,
                               print_steps=False, mode='B', max_coalitions=0, use_first_move=False, pre = None):
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
        initial_labels = pre(agents,initial_clusters)
        initial_clustering = {i: initial_labels[i] for i in range(len(agents))}
    else:
        initial_clustering = {i: i % initial_clusters for i in range(len(agents))}

    G_F,G_E = create_relations_euclid(agents,f,e)
    return locally_popular_clustering(agents,G_F,G_E,initial_clustering,always_allow_exit,print_steps,mode,max_coalitions,use_first_move)


def locally_popular_clustering_with_hop_distance(agents,f,e, initial_clusters=None, always_allow_exit=False,
                               print_steps=False, mode='B', max_coalitions=0, use_first_move=False, pre = None):
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
        p = pre(agents,initial_clusters)
        initial_labels = extract_labels_from_communities(p.communities)
        initial_clustering = {i: initial_labels[i] for i in range(len(agents))}
    else:
        initial_clustering = {i: i % initial_clusters for i in range(len(agents))}

    G_F,G_E = create_relations_hop_distance(agents,f,e)
    return locally_popular_clustering(agents,G_F,G_E,initial_clustering,always_allow_exit,print_steps,mode,max_coalitions,use_first_move)



def locally_popular_clustering(agents, friendship_graph, enemy_graph, initial_clustering, always_allow_exit=False,
                               print_steps=False, mode='B', max_coalitions=0, use_first_move=False):
    """
    Performs iterative clustering of agents to achieve a locally popular clustering.

    Args:
        agents (list): List of agent identifiers.
        friendship_graph (networkx.Graph): Graph representing friendships between agents.
        enemy_graph (networkx.Graph): Graph representing enmities between agents.
        initial_clustering (dict): Mapping from each agent to an initial cluster ID.
        always_allow_exit (bool, optional): If True, agents can always form new singleton clusters. Defaults to False.
        print_steps (bool, optional): If True, prints details of each move made by agents. Defaults to False.
        mode (str, optional): Determines the move selection rule ('B', 'F', or 'E'). Defaults to 'B'.
        max_coalitions (int, optional): Maximum allowed number of clusters. If 0, no limit is enforced. Defaults to 0.
        use_first_move (bool, optional): If True, uses the first improving move instead of the best move. Defaults to False.

    Returns:
        dict: A mapping from each agent to their final cluster ID after reaching local stability.
    """

    # Initialize clustering and cluster-to-agents mapping
    clustering = initial_clustering.copy()  # Map agent to cluster ID
    cluster_to_agents = defaultdict(set)
    for agent, cluster in clustering.items():
        cluster_to_agents[cluster].add(agent)

    if max_coalitions == 0:
        max_coalitions = len(cluster_to_agents)

    # Precompute (f, e) values for each agent in all clusters
    f_e_values = precompute_f_e_values(agents, cluster_to_agents, enemy_graph, friendship_graph)
    total_time = 0
    total_time2 = 0
    stable = False
    num_switches = 0
    next_cluster_id = len(cluster_to_agents)
    while not stable:
        if len(cluster_to_agents) < max_coalitions or always_allow_exit:
            allow_exit = True
        else:
            allow_exit = False
        stable = True

        if use_first_move:
            [v,best_move,best_vote] = find_first_move(allow_exit, cluster_to_agents, clustering, f_e_values, agents, mode)
        else:
            [v, best_move, best_vote] = find_best_move(allow_exit, cluster_to_agents, clustering, f_e_values, agents,
                                                        mode)



        # Make the best move if it's an improvement
        if best_move is not None:

            current_cluster = clustering[v]

            stable = False

            # -------------------------------------------------
            # update cluster membership
            # -------------------------------------------------

            current_members = cluster_to_agents[current_cluster]
            current_members.remove(v)

            if not current_members:
                del cluster_to_agents[current_cluster]

            # -------------------------------------------------
            # determine target cluster
            # -------------------------------------------------

            if best_move == "exit":

                best_move = next_cluster_id
                next_cluster_id += 1

                cluster_to_agents[best_move] = {v}

            else:

                cluster_to_agents[best_move].add(v)

            clustering[v] = best_move

            # -------------------------------------------------
            # local references (faster)
            # -------------------------------------------------

            f_e = f_e_values

            # -------------------------------------------------
            # friendship updates
            # -------------------------------------------------

            for neighbor in friendship_graph[v]:
                neighbor_values = f_e[neighbor]

                neighbor_values[current_cluster][0] -= 1
                neighbor_values[best_move][0] += 1

            # -------------------------------------------------
            # enemy updates
            # -------------------------------------------------

            for neighbor in enemy_graph[v]:
                neighbor_values = f_e[neighbor]

                neighbor_values[current_cluster][1] -= 1
                neighbor_values[best_move][1] += 1

            num_switches += 1

            if print_steps:
                print(f"{v} switches to {best_move}")
            # print(f"{f_e_values[v][current_cluster]}")
            # print(f"{f_e_values[v][best_move]}")


    #print(f'NEW swaps: {num_switches}')
    return clustering



def precompute_f_e_values(agents, cluster_to_agents, enemy_graph, friendship_graph):
    """
    Precomputes the number of friends and enemies each agent has in every cluster.

    Args:
        agents (list): List of agent identifiers (used for indexing).
        cluster_to_agents (dict): Mapping from cluster IDs to sets of agents in each cluster.
        enemy_graph (networkx.Graph): Graph representing enmities between agents.
        friendship_graph (networkx.Graph): Graph representing friendships between agents.

    Returns:
        dict: A nested dictionary of the form f_e_values[agent][cluster] = [num_friends, num_enemies],
              where each entry holds the count of friends and enemies the agent has in the given cluster.
    """
    f_e_values = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # {agent: {cluster: [f, -e]}}

    for v in range(len(agents)):
        for cluster in cluster_to_agents:
            friends_in_cluster = sum(
                1 for neighbor in friendship_graph[v] if neighbor in cluster_to_agents[cluster])
            enemies_in_cluster = sum(
                1 for neighbor in enemy_graph[v] if neighbor in cluster_to_agents[cluster])
            f_e_values[v][cluster] = [friends_in_cluster, enemies_in_cluster]
    return f_e_values



def find_first_move(allow_exit, cluster_to_agents, clustering, f_e_values, agents,mode):
    """
    Finds the first improving move for any agent based on local popularity constraints.

    Args:
        allow_exit (bool): Whether agents are allowed to exit their cluster and form a new one.
        cluster_to_agents (dict): Mapping from cluster IDs to sets of agents currently in each cluster.
        clustering (dict): Mapping from each agent to their current cluster ID.
        f_e_values (dict): Nested dictionary of the form f_e_values[agent][cluster] = [num_friends, num_enemies].
        agents (list): List of agent identifiers.
        mode (str): Specifies which set of constraints to use ('B', 'F', or 'E').

    Returns:
        list: A list [agent, target_cluster, vote_value], where:
              - agent (int): The first agent found who can improve their situation by moving.
              - target_cluster (int or str): The ID of the target cluster, or "exit" for forming a new one.
              - vote_value (int): A placeholder value (currently 0) for compatibility with other move-selection functions.
              Returns [None, None, 0] if no improving move is found.
    """
    for v in range(len(agents)):
        current_cluster = clustering[v]
        [f_current, e_current] = f_e_values[v][current_cluster]

        # Evaluate moves to existing clusters
        for candidate_cluster in cluster_to_agents:
            if candidate_cluster == current_cluster:
                continue
            [f_target, e_target] = f_e_values[v][candidate_cluster]

            if mode == 'B' and constraint_0(f_current, e_current, f_target, e_target):
                return [v,candidate_cluster,0]
            if mode == 'F' and (constraint_1(f_current, e_current, f_target, e_target) or
                    (constraint_2(f_current, e_current, f_target, e_target) and constraint_3(f_current, e_current, f_target, e_target))):
                return [v,candidate_cluster,0]
            if mode == 'E' and (constraint_1(f_current, e_current, f_target, e_target) or
                  (constraint_2(f_current, e_current, f_target, e_target) and constraint_4(f_current, e_current, f_target, e_target))):
                return [v,candidate_cluster,0]

        # Evaluate exit (if allowed)
        if allow_exit:
            if mode == 'B' and constraint_0(f_current, e_current, 0, 0):
                return [v,"exit",0]
            if mode == 'F' and (constraint_1(f_current, e_current, 0, 0) or
                    (constraint_2(f_current, e_current, 0, 0) and constraint_3(f_current, e_current,0, 0))):
                return [v, "exit", 0]
            if mode == 'E' and (constraint_1(f_current, e_current, 0, 0) or
                    (constraint_2(f_current, e_current, 0, 0) and constraint_4(f_current, e_current,0, 0))):
                return [v, "exit", 0]

    return [None,None,0]


def find_best_move(allow_exit, cluster_to_agents, clustering, f_e_values, agents, mode):
    """
    Finds the best possible move for any agent based on the utility function.

    Args:
        allow_exit (bool): Whether agents are allowed to exit their cluster and form a new one.
        cluster_to_agents (dict): Maps cluster IDs to sets of agents currently in each cluster.
        clustering (dict): Maps each agent to their current cluster ID.
        f_e_values (dict): Nested dictionary where f_e_values[agent][cluster] = [num_friends, num_enemies].
        agents (list): List of agent identifiers.
        mode (str): Specifies which constraint mode to use ('B', 'F', or 'E').

    Returns:
        list: A list [agent, target_cluster, vote_value], where:
              - agent (int): The agent with the most favorable move.
              - target_cluster (int or str): The best cluster for the agent to move to, or "exit" to form a new one.
              - vote_value (int): Score indicating the desirability of the move.
              Returns [None, None, 0] if no beneficial move is found.
    """

    n = len(agents)

    # Precompute weights
    f_value = n if mode == 'F' else 1
    e_value = n if mode == 'E' else 1

    best_agent = None
    best_target = None
    best_vote = 0

    candidate_clusters = tuple(cluster_to_agents)

    for v in range(n):

        current_cluster = clustering[v]

        f_current, e_current = f_e_values[v][current_cluster]

        # Cache repeated terms
        base_vote = e_current - f_current
        weighted_base = (
                e_value * e_current
                - f_value * f_current
        )

        fe_v = f_e_values[v]

        # Existing cluster moves
        for candidate_cluster in candidate_clusters:

            if candidate_cluster == current_cluster:
                continue

            f_target, e_target = fe_v[candidate_cluster]

            vote = (
                    f_target
                    - e_target
                    + base_vote
            )

            # Early skip
            if vote < 0:
                continue

            weighted = (
                    f_value * f_target
                    - e_value * e_target
                    + weighted_base
            )

            # Faster sign computation
            vote += (weighted > 0) - (weighted < 0)

            if vote > best_vote:
                best_vote = vote
                best_agent = v
                best_target = candidate_cluster

        # Exit move
        if allow_exit:

            vote = base_vote

            vote += (
                    (weighted_base > 0)
                    - (weighted_base < 0)
            )

            if vote > best_vote:
                best_vote = vote
                best_agent = v
                best_target = "exit"

    return [best_agent, best_target, best_vote]




def constraint_0(f_current,e_current,f_target,e_target):
    if f_target - e_target >= f_current - e_current + 1:
        return True
    return False

def constraint_1(f_current,e_current,f_target,e_target):
    if f_target - e_target >= f_current - e_current +2:
        return True
    return False

def constraint_2(f_current,e_current,f_target,e_target):
    if (f_current - e_current + 1 >= f_target - e_target) and (f_target - e_target >= f_current + e_current):
        return True
    return False

def constraint_3(f_current,e_current,f_target,e_target):
    if 2*f_target-e_target > 2*f_current - e_current:
        return True
    return False

def constraint_4(f_current,e_current,f_target,e_target):
    if f_target-2*e_target > f_current - 2*e_current:
        return True
    return False



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