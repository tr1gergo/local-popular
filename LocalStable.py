
from collections import defaultdict


from GraphFunctions import create_graphs_euclid, create_graphs_hop_distance


def locally_stable_clustering_with_euclid_graphs(agents,f,e, initial_clusters=None, always_allow_exit=False,
                               print_steps=False, mode='B', max_coalitions=0, pre = None):
    """
    Creates initial clustering, and friend/enemy graphs before starting the locally stable algorithm
    Args:
        agents (list): List of agent identifiers.
        initial_clusters (int): Number of initial clusters.
        always_allow_exit (bool, optional): If True, agents can always form new singleton clusters. Defaults to False.
        print_steps (bool, optional): If True, prints details of each move made by agents. Defaults to False.
        mode (str, optional): Determines the move selection rule ('B', 'F', or 'E'). Defaults to 'B'.
        max_coalitions (int, optional): Maximum allowed number of clusters. If 0, no limit is enforced. Defaults to 0.
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

    G_F,G_E = create_graphs_euclid(agents,f,e)
    return locally_stable_clustering(agents,G_F,G_E,initial_clustering,always_allow_exit,print_steps,mode,max_coalitions)


def locally_stable_clustering_with_hop_distance(agents,f,e, initial_clusters=None, always_allow_exit=False,
                               print_steps=False, mode='B', max_coalitions=0, pre = None):
    """
    Creates initial clustering, and friend/enemy graphs before starting the locally stable algorithm
    Args:
        agents (list): List of agent identifiers.
        initial_clusters (int): Number of initial clusters.
        always_allow_exit (bool, optional): If True, agents can always form new singleton clusters. Defaults to False.
        print_steps (bool, optional): If True, prints details of each move made by agents. Defaults to False.
        mode (str, optional): Determines the move selection rule ('B', 'F', or 'E'). Defaults to 'B'.
        max_coalitions (int, optional): Maximum allowed number of clusters. If 0, no limit is enforced. Defaults to 0.
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

    G_F,G_E = create_graphs_hop_distance(agents,f,e)

    return locally_stable_clustering(agents,G_F,G_E,initial_clustering,always_allow_exit,print_steps,mode,max_coalitions)




def locally_stable_clustering(agents, friendship_graph, enemy_graph, initial_clustering, always_allow_exit=False,
                               print_steps=False, mode='B', max_coalitions=0):
    """
    Performs iterative clustering of agents to achieve a locally stable clustering.

    Args:
        agents (list): List of agent identifiers.
        friendship_graph (networkx.Graph): A NetworkX graph representing friendships between agents.
        enemy_graph (networkx.Graph): A NetworkX graph representing enmities between agents.
        initial_clustering (dict): A dictionary mapping each agent to an initial cluster ID.
        always_allow_exit (bool, optional): If True, agents can always form new singleton clusters. Default is False.
        print_steps (bool, optional): If True, prints details of each move made by agents. Default is False.
        mode (str, optional): Determines the move selection rule (e.g., 'B' for a specific strategy). Default is 'B'.
        max_coalitions (int, optional): Maximum allowed number of clusters. If 0, no limit is enforced. Default is 0.

    Returns:
        dict: A dictionary mapping each agent to their final cluster ID after reaching local stability.
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

    stable = False
    num_switches = 0
    while not stable:
        if len(cluster_to_agents) < max_coalitions or always_allow_exit:
            allow_exit = True
        else:
            allow_exit = False
        stable = True

        [v, best_move, best_vote] = find_best_move(allow_exit, cluster_to_agents, clustering, f_e_values, agents,
                                                        mode)

        # Make the best move if it's an improvement
        if best_move is not None:
            current_cluster = clustering[v]
            stable = False
            # Update cluster mappings
            cluster_to_agents[current_cluster].remove(v)
            if len(cluster_to_agents[current_cluster]) == 0:
                cluster_to_agents.pop(current_cluster)


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

    #print(num_switches)
    return clustering


def precompute_f_e_values(agents, cluster_to_agents, enemy_graph, friendship_graph):
    """
    Precomputes the number of friends and enemies each agent has in every cluster.

    Args:
        agents (list): List of agent identifiers (used for indexing).
        cluster_to_agents (dict): A dictionary mapping cluster IDs to sets of agents in each cluster.
        enemy_graph (networkx.Graph): A NetworkX graph representing enmities between agents.
        friendship_graph (networkx.Graph): A NetworkX graph representing friendships between agents.

    Returns:
        dict: A nested dictionary in the form f_e_values[agent][cluster] = [num_friends, num_enemies],
              where each entry holds the count of friends and enemies the agent has in the given cluster.

    """
    f_e_values = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # {agent: {cluster: [f, -e]}}
    for v in range(len(agents)):
        for cluster in cluster_to_agents:
            friends_in_cluster = sum(
                1 for neighbor in friendship_graph.neighbors(v) if neighbor in cluster_to_agents[cluster])
            enemies_in_cluster = sum(
                1 for neighbor in enemy_graph.neighbors(v) if neighbor in cluster_to_agents[cluster])
            f_e_values[v][cluster] = [friends_in_cluster, enemies_in_cluster]
    return f_e_values



def find_best_move(allow_exit, cluster_to_agents, clustering, f_e_values, agents, mode):
    """
    Finds the best possible move for any agent based on the utility function.

    Args:
        allow_exit (bool): Whether agents are allowed to exit their cluster and form a new one.
        cluster_to_agents (dict): A dictionary mapping cluster IDs to sets of agents currently in each cluster.
        clustering (dict): A dictionary mapping each agent to their current cluster ID.
        f_e_values (dict): A nested dictionary in the form f_e_values[agent][cluster] = [num_friends, num_enemies].
        agents (list): List of agent identifiers.
        mode (str): Specifies which set of constraints to use ('B', 'F', or 'E').

    Returns:
        list: A list [agent, target_cluster, vote_value], where:
            - agent: The agent who has the most favorable move available.
            - target_cluster: The best target cluster for the agent to move to, or "exit" to form a new one.
            - vote_value: An integer representing the score or desirability of the move.
          Returns [None, None, 0] if no beneficial move is found.
    """

    n = len(agents)
    f_value, e_value = 1, 1
    if mode == 'F':
        f_value = n
    if mode == 'E':
        e_value = n

    best_move_agent = None
    best_move_target = None
    best_move_vote = 0  # Track best improvement (f, -e)
    for v in range(len(agents)):
        current_cluster = clustering[v]
        [f_current, e_current] = f_e_values[v][current_cluster]

        # Evaluate moves to existing clusters
        for candidate_cluster in cluster_to_agents:
            if candidate_cluster == current_cluster:
                continue

            [f_target, e_target] = f_e_values[v][candidate_cluster]
            vote = f_target + e_current - f_current - e_target
            if vote < 0:
                continue
            v_vote = f_value * f_target + e_value * e_current - f_value * f_current - e_value * e_target
            if v_vote > 0:
                v_vote = 1
            if v_vote <= 0:
                continue
            vote = vote + v_vote

            if (vote > best_move_vote):
                best_move_agent = v
                best_move_target = candidate_cluster
                best_move_vote = vote

        # Evaluate exit (if allowed)
        if allow_exit:
            vote = e_current - f_current
            v_vote = e_value * e_current - f_value * f_current
            if v_vote > 0:
                v_vote = 1
            if v_vote < 0:
                v_vote = -1
            vote = vote + v_vote

            if  vote > best_move_vote:
                best_move_agent = v
                best_move_target = "exit"
                best_move_vote = vote
    return [best_move_agent, best_move_target, best_move_vote]



def extract_labels_from_communities(communities):
    """
    Converts a dictionary of node-to-community mappings into a list of communities,
    where the index i in the list represents node i.

    Args:
        dictionary (dict): A dictionary where the keys are nodes and the values are community labels.

    Returns:
        list: A list of sets, where each set represents a community and contains the nodes assigned to that community.
    """
    d = dict()
    for i in range(len(communities)):
        c = communities[i]
        for node in c:
            d[node] = i

    return d