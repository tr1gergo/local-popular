
from collections import defaultdict
import time

import networkx as nx
from pandas.tseries import frequencies


def locally_popular_clustering(agents, friendship_graph, enemy_graph, initial_clustering, allow_exit=False,
                               print_steps=False, mode='B', max_coalitions=0, use_first_move=False):
    """Perform clustering to achieve local popularity."""
    # Initialize clustering and cluster-to-agents mapping
    n = len(agents)

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
        if len(cluster_to_agents) < max_coalitions:
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
    f_e_values = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # {agent: {cluster: [f, -e]}}
    for v in range(len(agents)):
        for cluster in cluster_to_agents:
            friends_in_cluster = sum(
                1 for neighbor in friendship_graph.neighbors(v) if neighbor in cluster_to_agents[cluster])
            enemies_in_cluster = sum(
                1 for neighbor in enemy_graph.neighbors(v) if neighbor in cluster_to_agents[cluster])
            f_e_values[v][cluster] = [friends_in_cluster, enemies_in_cluster]
    return f_e_values

# returns the first move that creates a locally dominating coalition structure
def find_first_move(allow_exit, cluster_to_agents, clustering, f_e_values, agents,mode):

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
            if v_vote < 0:
                v_vote = -1
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
    d = dict()
    for i in range(len(communities)):
        c = communities[i]
        for node in c:
            d[node] = i

    return d

def time_tester(test_function,repeat):
    times = []
    output = []

    for i in range(repeat):
        start_time = time.perf_counter()
        out = test_function()
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        times = times + [elapsed_time]
        output = output + [out]

    return times, output

from sklearn.metrics import rand_score, silhouette_score, davies_bouldin_score


def calculate_scores_CD(outputs,truth,graph):
    rand_scores = []
    modularity_scores = []

    for output in outputs:
        labels = list(output.values())
        if truth is not None:
            rand_scores += [rand_score(truth,labels)]
        else:
            rand_scores += [-1]

        communities = get_communities_from_dict(output)
        #print(communities)
        modularity_scores += [nx.community.modularity(graph,communities)]

    avg_rand = sum(rand_scores)/len(rand_scores)
    if avg_rand == -1.0:
        avg_rand = 'n.A.'
    avg_modularity = sum(modularity_scores)/len(modularity_scores)
    scores = {'Rand Index':avg_rand, 'Modularity':avg_modularity}
    return scores

def calculate_scores_clustering(outputs,truth,graph):
    rand_scores = []
    silhouette_scores = []
    db_scores = []

    for output in outputs:
        if truth is not None:
            rand_scores += [rand_score(truth,output)]
            silhouette_scores += [silhouette_score(graph,truth)]
            db_scores += [davies_bouldin_score(graph,truth)]
        else:
            rand_scores += [-1]
            silhouette_scores += [-1]




    avg_rand = sum(rand_scores)/len(rand_scores)
    avg_silhouette = sum(silhouette_scores)/len(silhouette_scores)
    avg_db = sum(db_scores)/len(db_scores)
    if avg_rand == -1.0:
        avg_rand = 'n.A.'
    scores = {'Rand Index':avg_rand, 'Silhouette Score':avg_silhouette, 'Davies Bouldin Score':avg_db}
    return scores


def get_communities_from_dict(dictionary):
    communities = {}
    for key, value in dictionary.items():
        if not value in communities:
            communities[value] = {key}
        else:
            communities[value].add(key)

    return communities.values()