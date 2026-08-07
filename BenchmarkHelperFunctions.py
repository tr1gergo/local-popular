
import time

import networkx as nx

from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score


def time_tester(function,permutations):
    """
    Measures execution time of a function over multiple input permutations.

    Args:
        function (callable): The function to be tested. It should accept a single input argument.
        permutations (list): A list of input values (or input structures) to be passed one at a time to the function.

    Returns:
        tuple:
            times (list of float): A list of execution times (in seconds) for each function call.
            output (list): A list of outputs returned by the function for each corresponding input.
    """
    times = []
    output = []

    for permutation in permutations:
        start_time = time.perf_counter()
        out = function(permutation)
        end_time = time.perf_counter()

        times.append(end_time - start_time)
        output.append(out)

    return times, output







def calculate_scores_CD(output, truth, graph):
    """
    Calculates average clustering evaluation metrics (Rand Index and Modularity) over multiple outputs.

    Args:
        output (list of dict): A list of clustering results, where each element is a dictionary mapping nodes to cluster labels.
        truth (list of list or None): A list of ground truth labelings corresponding to each output. If an entry is None, the Rand Index is not computed for that case.
        graph (list of networkx.Graph): A list of NetworkX graphs corresponding to each clustering result.

    Returns:
        dict: A dictionary containing:
            - 'Rand Index' (float or str): The average Rand Index across all test cases, or 'n.A.' if all were skipped.
            - 'Modularity' (float): The average modularity score across all test cases.
    """
    rand_scores = []
    modularity_scores = []

    for i in range(len(output)):
        labels = list(output[i].values())

        if truth[i] is not None:
            rand_scores.append(adjusted_rand_score(truth[i], labels))
        else:
            rand_scores.append(-1)

        communities = get_communities_from_dict(output[i])



        modularity_scores.append(nx.community.modularity(graph[i], communities))


    avg_rand = sum(rand_scores) / len(rand_scores)
    if avg_rand == -1.0:
        avg_rand = 'n.A.'
    avg_modularity = sum(modularity_scores) / len(modularity_scores)

    scores = {'Rand Index': avg_rand, 'Modularity': avg_modularity}
    return scores




def calculate_scores_clustering(output,truth,graph):
    """
    Calculates average clustering evaluation metrics (Rand Index, Silhouette Score, Davies-Bouldin Score) over multiple outputs.

    Args:
        output (list of list): A list of clustering results, where each element is a list of cluster labels corresponding to each node.
        truth (list of list or None): A list of ground truth labelings corresponding to each output. If an entry is None, the evaluation metrics are not computed for that case.
        graph (list of networkx.Graph): A list of NetworkX graphs corresponding to each clustering result, used to calculate silhouette and Davies-Bouldin scores.

    Returns:
        dict: A dictionary containing:
            - 'Rand Index' (float or str): The average Rand Index across all test cases, or 'n.A.' if all were skipped.
            - 'Silhouette Score' (float or str): The average Silhouette Score across all test cases, or 'n.A.' if all were skipped.
            - 'Davies Bouldin Score' (float or str): The average Davies-Bouldin Score across all test cases, or 'n.A.' if all were skipped.
    """
    rand_scores = []
    silhouette_scores = []
    db_scores = []

    for i in range(len(output)):
        if truth[i] is not None:
            rand_scores.append(adjusted_rand_score(truth[i], output[i]))
            if len(set(output[i])) == 1:
                silhouette_scores.append(-100)
                db_scores.append(-100)
            else:
                silhouette_scores.append(silhouette_score(graph[i], output[i]))
                db_scores.append(davies_bouldin_score(graph[i], output[i]))
        else:
            rand_scores.append(-1)
            silhouette_scores.append(-100)
            db_scores.append(-100)

    avg_rand = sum(rand_scores)/len(rand_scores)
    avg_silhouette = sum(silhouette_scores)/len(silhouette_scores)
    avg_db = sum(db_scores)/len(db_scores)

    if avg_rand == -1.0:
        avg_rand = 'n.A.'
    if avg_silhouette == -100.0:
        avg_silhouette = 'n.A.'
    if avg_db == -100.0:
        avg_db = 'n.A.'
    scores = {'Rand Index':avg_rand, 'Silhouette Score':avg_silhouette, 'Davies Bouldin Score':avg_db}
    return scores



def get_communities_from_dict(dictionary):
    """
    Converts a dictionary of node-to-community mappings into a list of communities,
    where the index i in the list represents node i.

    Args:
        dictionary (dict): A dictionary where the keys are nodes and the values are community labels.

    Returns:
        list of set: A list of sets, where each set represents a community and contains the nodes assigned to that community.
    """
    communities = {}
    for key, value in dictionary.items():
        if not value in communities:
            communities[value] = {key}
        else:
            communities[value].add(key)

    return communities.values()