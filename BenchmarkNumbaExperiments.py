"""Reproducible experiment runner used by the four NUMBA benchmark notebooks.

The runner centralizes dataset generation, paired randomization, scoring,
diagnostics, CSV export, and plotting so that the LocPop and LocStab notebooks
cannot silently drift apart.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_breast_cancer, load_iris, make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from BenchmarkHelperFunctions import get_communities_from_dict
from GraphFunctions import (
    create_relations_euclid,
    create_relations_hop_distance_np,
    generate_graph,
    my_make_circles,
)
from LocalPopularNumba import (
    extract_labels_from_communities,
    locally_popular_clustering_numba,
)
from community_detection.leiden import leiden
from community_detection.louvain import louvain
from community_detection.quality_functions import Modularity
import data.cora as cora
import data.jazz as jazz


ROOT = Path(__file__).resolve().parent
DATA_SEED = 20260817
THRESHOLDS = ((0.20, 0.20), (0.25, 0.35), (0.40, 0.40))
DOMAINS = ("B", "AF", "AE")
MODE_BY_DOMAIN = {"B": "B", "AF": "F", "AE": "E"}
RANDOM25_WITHIN_P = 0.40
RANDOM25_BETWEEN_P = 0.001


def normalize_labels(labels) -> np.ndarray:
    """Map arbitrary labels, including DBSCAN's -1, to contiguous integers."""
    mapping = {}
    normalized = np.empty(len(labels), dtype=np.int32)
    for index, value in enumerate(labels):
        key = value.item() if hasattr(value, "item") else value
        if key not in mapping:
            mapping[key] = len(mapping)
        normalized[index] = mapping[key]
    return normalized


def labels_to_dict(labels) -> dict[int, int]:
    labels = normalize_labels(labels)
    return {index: int(label) for index, label in enumerate(labels)}


def labels_from_dict(labels: dict[int, int]) -> np.ndarray:
    return normalize_labels([labels[index] for index in range(len(labels))])


def permute_relations(relations, permutation: np.ndarray):
    """Relabel an adjacency-list relation using new-index -> old-index order."""
    n = len(permutation)
    inverse = np.empty(n, dtype=np.int32)
    inverse[permutation] = np.arange(n, dtype=np.int32)
    permuted = []
    for old_vertex in permutation:
        neighbors = np.asarray(relations[int(old_vertex)], dtype=np.int32)
        permuted.append(np.sort(inverse[neighbors]).astype(np.int32))
    return permuted


def permute_graph(graph: nx.Graph, permutation: np.ndarray) -> nx.Graph:
    mapping = {int(old): int(new) for new, old in enumerate(permutation)}
    relabeled = nx.relabel_nodes(graph, mapping, copy=True)
    ordered = nx.Graph()
    ordered.add_nodes_from(range(len(permutation)))
    ordered.add_edges_from(relabeled.edges(data=True))
    return ordered


def _relation_counts(friends, enemies):
    return (
        int(sum(len(neighbors) for neighbors in friends) // 2),
        int(sum(len(neighbors) for neighbors in enemies) // 2),
    )


def _community_data_diagnostics(datasets) -> pd.DataFrame:
    rows = []
    for name, graph, truth, clusters in datasets:
        components = list(nx.connected_components(graph))
        largest = graph.subgraph(max(components, key=len))
        row = {
            "Dataset": name,
            "Nodes": len(graph),
            "Edges": graph.number_of_edges(),
            "Connected Components": len(components),
            "Largest Component": len(largest),
            "Largest-Component Diameter": nx.diameter(largest),
            "Reference Classes": clusters,
        }
        if truth is not None:
            within = sum(truth[u] == truth[v] for u, v in graph.edges())
            between = graph.number_of_edges() - within
            row.update({
                "Within-Class Edges": within,
                "Between-Class Edges": between,
                "Mean Within-Class Degree": 2 * within / len(graph),
                "Mean Between-Class Degree": 2 * between / len(graph),
            })
        if name == "Random-25":
            row.update({
                "Generator Within Probability": RANDOM25_WITHIN_P,
                "Generator Between Probability": RANDOM25_BETWEEN_P,
                "Connected Blocks": True,
                "Ring Backbone": True,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _community_labels(result, n: int) -> np.ndarray:
    labels = extract_labels_from_communities(result.communities)
    return normalize_labels([labels[index] for index in range(n)])


def _modularity(graph: nx.Graph, labels: np.ndarray) -> float:
    partition = get_communities_from_dict(labels_to_dict(labels))
    return float(nx.community.modularity(graph, partition))


def _silhouette(points: np.ndarray, labels: np.ndarray) -> float:
    count = len(np.unique(labels))
    if count <= 1 or count >= len(labels):
        return np.nan
    return float(silhouette_score(points, labels))


def _record_baseline(method, dataset, repetition, seconds, ari, secondary_name, secondary):
    return {
        "Method": method,
        "Dataset": dataset,
        "Preference": "-",
        "Initialization": "-",
        "Beta Friend": np.nan,
        "Beta Enemy": np.nan,
        "Repetition": repetition,
        "Adjusted Rand Index": ari,
        secondary_name: secondary,
        "Seconds": seconds,
        "Moves": np.nan,
        "Converged": np.nan,
        "Final Coalitions": np.nan,
        "Initial Adjusted Rand Index": np.nan,
    }


def _record_heuristic(method, dataset, domain, initialization, threshold,
                      repetition, seconds, ari, secondary_name, secondary,
                      initial_ari, diagnostics):
    return {
        "Method": method,
        "Dataset": dataset,
        "Preference": domain,
        "Initialization": initialization,
        "Beta Friend": threshold[0],
        "Beta Enemy": threshold[1],
        "Repetition": repetition,
        "Adjusted Rand Index": ari,
        secondary_name: secondary,
        "Seconds": seconds,
        "Moves": diagnostics["moves"],
        "Converged": int(diagnostics["converged"]),
        "Final Coalitions": diagnostics["final_coalitions"],
        "Initial Adjusted Rand Index": initial_ari,
    }


def _summarize_runs(runs: pd.DataFrame, secondary_name: str) -> pd.DataFrame:
    keys = ["Method", "Dataset", "Preference", "Initialization", "Beta Friend", "Beta Enemy"]
    metrics = [
        "Adjusted Rand Index",
        secondary_name,
        "Seconds",
        "Moves",
        "Converged",
        "Final Coalitions",
        "Initial Adjusted Rand Index",
    ]
    rows = []
    for key, group in runs.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, key))
        row["Repetitions"] = len(group)
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[metric] = values.mean()
            row[f"{metric} SD"] = values.std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_outputs(runs: pd.DataFrame, summary: pd.DataFrame, family: str,
                   preprocessing: pd.DataFrame, data_diagnostics: pd.DataFrame | None = None):
    output_dir = ROOT / "csv" / family
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove obsolete exports from the pre-correction notebooks so they cannot
    # be mistaken for part of the regenerated dataset.
    retained_threshold_files = {f"dataset-{index}.csv" for index in range(len(THRESHOLDS))}
    for stale in output_dir.glob("dataset-*.csv"):
        if stale.name not in retained_threshold_files:
            stale.unlink()
    for stale in output_dir.glob("Random-25-*.csv"):
        stale.unlink()

    runs.to_csv(output_dir / "runs.csv", index=False)
    summary.to_csv(output_dir / "results.csv", index=False)
    preprocessing.to_csv(output_dir / "preprocessing.csv", index=False)
    if data_diagnostics is not None:
        data_diagnostics.to_csv(output_dir / "data-diagnostics.csv", index=False)

    baselines = summary[summary["Method"].isin(["KMeans", "DBSCAN", "Louvain", "Leiden"])]
    for index, (beta_f, beta_e) in enumerate(THRESHOLDS):
        heuristic = summary[
            np.isclose(summary["Beta Friend"], beta_f, equal_nan=False)
            & np.isclose(summary["Beta Enemy"], beta_e, equal_nan=False)
        ]
        pd.concat([baselines, heuristic], ignore_index=True).to_csv(
            output_dir / f"dataset-{index}.csv", index=False
        )


def _method_label(row) -> str:
    method = row["Method"]
    if method in {"KMeans", "DBSCAN", "Louvain", "Leiden"}:
        return method
    prefix = "LocStab" if method == "LocStab" else "LocPop"
    return f"{prefix}-{row['Initialization']}"


def _plot_metric(summary: pd.DataFrame, family: str, metric: str, datasets: list[str]):
    output_dir = ROOT / "figures" / family
    output_dir.mkdir(parents=True, exist_ok=True)
    method = "LocStab" if family.startswith("Stable") else "LocPop"
    baseline_order = ["Louvain", "Leiden"] if "Community" in family else ["KMeans", "DBSCAN"]
    init_order = ["S", "P", "KM", "D", "Ld"]
    domain_colors = {"B": "#228833", "AF": "#EE9911", "AE": "#4477AA"}
    threshold_markers = {(0.20, 0.20): "^", (0.25, 0.35): "o", (0.40, 0.40): "s"}

    labels = baseline_order + [
        f"{method}-{initialization}"
        for initialization in init_order
        if ((summary["Method"] == method) & (summary["Initialization"] == initialization)).any()
    ]
    positions = {label: index for index, label in enumerate(labels)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for axis, dataset in zip(axes.flat, datasets):
        data = summary[summary["Dataset"] == dataset]
        if not data[metric].notna().any():
            axis.set_title(dataset)
            axis.text(
                0.5, 0.5, "ARI unavailable\n(no reference labels)",
                transform=axis.transAxes, ha="center", va="center",
                fontsize=13, color="dimgray",
            )
            axis.set_xticks([])
            axis.set_ylabel(metric)
            axis.grid(axis="y", linestyle="--", alpha=0.35)
            continue
        for baseline in baseline_order:
            row = data[data["Method"] == baseline]
            if row.empty or pd.isna(row.iloc[0][metric]):
                continue
            value = row.iloc[0][metric]
            deviation = row.iloc[0][f"{metric} SD"]
            axis.errorbar(
                positions[baseline], value, yerr=2 * deviation,
                fmt="o", color="black", capsize=3, markersize=7,
            )

        heuristic_rows = data[data["Method"] == method]
        for _, row in heuristic_rows.iterrows():
            value = row[metric]
            if pd.isna(value):
                continue
            label = _method_label(row)
            threshold = (round(float(row["Beta Friend"]), 2), round(float(row["Beta Enemy"]), 2))
            domain = row["Preference"]
            offset = {"AF": -0.16, "B": 0.0, "AE": 0.16}[domain]
            axis.errorbar(
                positions[label] + offset,
                value,
                yerr=2 * row[f"{metric} SD"],
                fmt=threshold_markers[threshold],
                color=domain_colors[domain],
                markeredgecolor="black",
                alpha=0.82,
                capsize=2,
                markersize=6,
            )

        axis.set_title(dataset)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=28, ha="right")
        axis.set_ylabel(metric)
        axis.grid(axis="y", linestyle="--", alpha=0.35)

    color_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=domain)
        for domain, color in domain_colors.items()
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker=marker, linestyle="", color="gray", label=str(threshold))
        for threshold, marker in threshold_markers.items()
    ]
    fig.legend(
        handles=color_handles + marker_handles,
        loc="outside lower center",
        ncol=6,
        frameon=True,
        title="Color: preference domain; marker: $(\\beta_f,\\beta_e)$",
    )
    safe_metric = metric.replace(" ", "-")
    fig.suptitle(f"{metric}: {method}", fontsize=16)
    fig.savefig(output_dir / f"summary-{safe_metric}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Also retain one plot per dataset and metric for detailed inspection.
    for dataset in datasets:
        data = summary[summary["Dataset"] == dataset]
        fig, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
        if not data[metric].notna().any():
            axis.set_title(f"{metric} on {dataset}")
            axis.text(
                0.5, 0.5, "ARI unavailable (no reference labels)",
                transform=axis.transAxes, ha="center", va="center",
                fontsize=13, color="dimgray",
            )
            axis.set_xticks([])
            axis.set_ylabel(metric)
            axis.grid(axis="y", linestyle="--", alpha=0.35)
            fig.savefig(output_dir / f"{dataset}-{safe_metric}.png", dpi=220, bbox_inches="tight")
            plt.close(fig)
            continue
        for baseline in baseline_order:
            row = data[data["Method"] == baseline]
            if row.empty or pd.isna(row.iloc[0][metric]):
                continue
            axis.errorbar(
                positions[baseline], row.iloc[0][metric],
                yerr=2 * row.iloc[0][f"{metric} SD"], fmt="o",
                color="black", capsize=3, markersize=7,
            )
        for _, row in data[data["Method"] == method].iterrows():
            if pd.isna(row[metric]):
                continue
            label = _method_label(row)
            threshold = (round(float(row["Beta Friend"]), 2), round(float(row["Beta Enemy"]), 2))
            domain = row["Preference"]
            offset = {"AF": -0.16, "B": 0.0, "AE": 0.16}[domain]
            axis.errorbar(
                positions[label] + offset, row[metric],
                yerr=2 * row[f"{metric} SD"], fmt=threshold_markers[threshold],
                color=domain_colors[domain], markeredgecolor="black",
                alpha=0.82, capsize=2, markersize=6,
            )
        axis.set_title(f"{metric} on {dataset}")
        axis.set_ylabel(metric)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=28, ha="right")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.legend(
            handles=color_handles + marker_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            title="Domain / thresholds",
        )
        fig.savefig(output_dir / f"{dataset}-{safe_metric}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def _plot_all(summary: pd.DataFrame, family: str, secondary_name: str, datasets: list[str]):
    output_dir = ROOT / "figures" / family
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.png"):
        stale.unlink()
    _plot_metric(summary, family, "Adjusted Rand Index", datasets)
    _plot_metric(summary, family, secondary_name, datasets)


def _clustering_datasets():
    moons, moons_truth = make_moons(n_samples=300, noise=0.05, random_state=DATA_SEED)
    circles, circles_truth = my_make_circles(300, random_state=DATA_SEED + 1)
    cancer = load_breast_cancer()
    iris = load_iris()
    raw = [
        ("Moons", moons, np.asarray(moons_truth), 2),
        ("3 Circles", circles, np.asarray(circles_truth), 3),
        ("Cancer", cancer.data, np.asarray(cancer.target), 2),
        ("Iris", iris.data, np.asarray(iris.target), 3),
    ]
    return [
        (name, StandardScaler().fit_transform(points), truth, clusters)
        for name, points, truth, clusters in raw
    ]


def _community_datasets():
    cora_graph = cora.get_graph()
    cora_mapping = {node: index for index, node in enumerate(cora_graph.nodes())}
    cora_graph = nx.relabel_nodes(cora_graph, cora_mapping, copy=True)
    cora_truth = np.asarray([cora_graph.nodes[index]["subject"] for index in range(len(cora_graph))])

    jazz_graph = jazz.get_graph()
    jazz_mapping = {node: index for index, node in enumerate(sorted(jazz_graph.nodes()))}
    jazz_graph = nx.relabel_nodes(jazz_graph, jazz_mapping, copy=True)

    karate_graph = nx.karate_club_graph()
    karate_truth = np.asarray([karate_graph.nodes[index]["club"] for index in range(len(karate_graph))])

    random_graph, random_truth = generate_graph(
        10, 25, RANDOM25_WITHIN_P, RANDOM25_BETWEEN_P,
        seed=DATA_SEED + 2,
        ensure_connected_blocks=True,
        connect_blocks_in_ring=True,
    )
    return [
        ("Karate Club", karate_graph, karate_truth, 2),
        ("Cora", cora_graph, cora_truth, 7),
        ("Jazz", jazz_graph, None, None),
        ("Random-25", random_graph, np.asarray(random_truth), 25),
    ]


def run_clustering_experiment(*, local_stable: bool, repetitions: int = 10):
    family = "StableClustering" if local_stable else "PopularClustering"
    method = "LocStab" if local_stable else "LocPop"
    records = []
    preprocessing_rows = []

    for dataset_index, (name, points, truth, clusters) in enumerate(_clustering_datasets()):
        relation_cache = {}
        for threshold in THRESHOLDS:
            started = time.perf_counter()
            relation_cache[threshold] = create_relations_euclid(points, *threshold)
            friend_edges, enemy_edges = _relation_counts(*relation_cache[threshold])
            preprocessing_rows.append({
                "Dataset": name,
                "Beta Friend": threshold[0],
                "Beta Enemy": threshold[1],
                "Seconds": time.perf_counter() - started,
                "Friend Edges": friend_edges,
                "Enemy Edges": enemy_edges,
            })

        for repetition in range(repetitions):
            seed = DATA_SEED + 10_000 * dataset_index + repetition
            permutation = np.random.default_rng(seed).permutation(len(points))
            permuted_points = points[permutation]
            permuted_truth = truth[permutation]

            started = time.perf_counter()
            km_labels = normalize_labels(KMeans(
                n_clusters=clusters, n_init=20, random_state=seed
            ).fit_predict(permuted_points))
            km_seconds = time.perf_counter() - started
            records.append(_record_baseline(
                "KMeans", name, repetition, km_seconds,
                float(adjusted_rand_score(permuted_truth, km_labels)),
                "Silhouette Score", _silhouette(permuted_points, km_labels),
            ))

            started = time.perf_counter()
            db_labels = normalize_labels(DBSCAN(eps=0.2, min_samples=5).fit_predict(permuted_points))
            db_seconds = time.perf_counter() - started
            records.append(_record_baseline(
                "DBSCAN", name, repetition, db_seconds,
                float(adjusted_rand_score(permuted_truth, db_labels)),
                "Silhouette Score", _silhouette(permuted_points, db_labels),
            ))

            initializations = {
                "S": np.arange(len(points), dtype=np.int32),
                "P": np.arange(len(points), dtype=np.int32) % clusters,
                "KM": km_labels,
                "D": db_labels,
            }

            for threshold in THRESHOLDS:
                base_friends, base_enemies = relation_cache[threshold]
                friends = permute_relations(base_friends, permutation)
                enemies = permute_relations(base_enemies, permutation)
                for domain in DOMAINS:
                    for initialization, initial_labels in initializations.items():
                        started = time.perf_counter()
                        output, diagnostics = locally_popular_clustering_numba(
                            list(range(len(points))), friends, enemies,
                            labels_to_dict(initial_labels),
                            mode=MODE_BY_DOMAIN[domain],
                            max_coalitions=len(points),
                            local_stable=local_stable,
                            return_diagnostics=True,
                        )
                        seconds = time.perf_counter() - started
                        labels = labels_from_dict(output)
                        records.append(_record_heuristic(
                            method, name, domain, initialization, threshold,
                            repetition, seconds,
                            float(adjusted_rand_score(permuted_truth, labels)),
                            "Silhouette Score", _silhouette(permuted_points, labels),
                            float(adjusted_rand_score(initial_labels, labels)), diagnostics,
                        ))

    runs = pd.DataFrame(records)
    summary = _summarize_runs(runs, "Silhouette Score")
    preprocessing = pd.DataFrame(preprocessing_rows)
    _write_outputs(runs, summary, family, preprocessing)
    _plot_all(summary, family, "Silhouette Score", ["Moons", "3 Circles", "Cancer", "Iris"])
    return summary


def run_community_experiment(*, local_stable: bool, repetitions: int = 10):
    family = "StableCommunity" if local_stable else "PopularCommunity"
    method = "LocStab" if local_stable else "LocPop"
    records = []
    preprocessing_rows = []
    quality = Modularity(1.0)

    datasets = _community_datasets()
    data_diagnostics = _community_data_diagnostics(datasets)

    for dataset_index, (name, graph, truth, clusters) in enumerate(datasets):
        relation_cache = {}
        for threshold in THRESHOLDS:
            started = time.perf_counter()
            relation_cache[threshold] = create_relations_hop_distance_np(graph, *threshold)
            friend_edges, enemy_edges = _relation_counts(*relation_cache[threshold])
            preprocessing_rows.append({
                "Dataset": name,
                "Beta Friend": threshold[0],
                "Beta Enemy": threshold[1],
                "Seconds": time.perf_counter() - started,
                "Friend Edges": friend_edges,
                "Enemy Edges": enemy_edges,
            })

        for repetition in range(repetitions):
            seed = DATA_SEED + 10_000 * dataset_index + repetition
            permutation = np.random.default_rng(seed).permutation(len(graph))
            permuted_graph = permute_graph(graph, permutation)
            permuted_truth = None if truth is None else truth[permutation]

            random.seed(seed)
            started = time.perf_counter()
            louvain_labels = _community_labels(louvain(permuted_graph, quality), len(graph))
            louvain_seconds = time.perf_counter() - started
            records.append(_record_baseline(
                "Louvain", name, repetition, louvain_seconds,
                np.nan if permuted_truth is None else float(adjusted_rand_score(permuted_truth, louvain_labels)),
                "Modularity", _modularity(permuted_graph, louvain_labels),
            ))

            random.seed(seed + 1)
            started = time.perf_counter()
            leiden_labels = _community_labels(leiden(permuted_graph, quality), len(graph))
            leiden_seconds = time.perf_counter() - started
            records.append(_record_baseline(
                "Leiden", name, repetition, leiden_seconds,
                np.nan if permuted_truth is None else float(adjusted_rand_score(permuted_truth, leiden_labels)),
                "Modularity", _modularity(permuted_graph, leiden_labels),
            ))

            initializations = {"Ld": leiden_labels}
            if name != "Cora":
                initializations["S"] = np.arange(len(graph), dtype=np.int32)
            if clusters is not None:
                initializations["P"] = np.arange(len(graph), dtype=np.int32) % clusters

            for threshold in THRESHOLDS:
                base_friends, base_enemies = relation_cache[threshold]
                friends = permute_relations(base_friends, permutation)
                enemies = permute_relations(base_enemies, permutation)
                capacity = 50 if name == "Cora" else len(graph)
                for domain in DOMAINS:
                    for initialization, initial_labels in initializations.items():
                        started = time.perf_counter()
                        output, diagnostics = locally_popular_clustering_numba(
                            list(range(len(graph))), friends, enemies,
                            labels_to_dict(initial_labels),
                            mode=MODE_BY_DOMAIN[domain],
                            max_coalitions=capacity,
                            local_stable=local_stable,
                            return_diagnostics=True,
                        )
                        seconds = time.perf_counter() - started
                        labels = labels_from_dict(output)
                        records.append(_record_heuristic(
                            method, name, domain, initialization, threshold,
                            repetition, seconds,
                            np.nan if permuted_truth is None else float(adjusted_rand_score(permuted_truth, labels)),
                            "Modularity", _modularity(permuted_graph, labels),
                            float(adjusted_rand_score(initial_labels, labels)), diagnostics,
                        ))

    runs = pd.DataFrame(records)
    summary = _summarize_runs(runs, "Modularity")
    preprocessing = pd.DataFrame(preprocessing_rows)
    _write_outputs(runs, summary, family, preprocessing, data_diagnostics)
    _plot_all(summary, family, "Modularity", ["Karate Club", "Cora", "Jazz", "Random-25"])
    return summary
