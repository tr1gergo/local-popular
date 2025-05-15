
from matplotlib import pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import ast


def plot_clustering(points, clusters, title):
    """
    Plots the given clustering.

    Args:
        points (list or array): A list or array of points where each entry is a list/array of coordinates.
        clusters (dict): A dictionary where keys are point indices (0 to n-1) and values are cluster indices.

    Returns:
        None
    """
    # Convert points to a numpy array for easier processing
    data = np.array(points)
    cluster_labels = np.array([clusters[i] for i in range(len(points))])  # Extract cluster labels

    # Check if the points are 2D
    if data.shape[1] != 2:
        raise ValueError("Only 2D data can be visualized. Your data has {} dimensions.".format(data.shape[1]))

    # Scatter plot for clustering
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap="viridis", s=50, alpha=0.6)

    # Add legend
    plt.legend(handles=scatter.legend_elements()[0], labels=set(cluster_labels), title="Clusters")
    plt.title(title)
    plt.xlabel("Coordinate 1")
    plt.ylabel("Coordinate 2")
    plt.grid()
    plt.show()

def plot_stuff(points, clusters1, clusters2, clusters3, title="",title_1 = "", title_2 = "", title_3 = "",ground_truth=None):
    data = np.array(points)
    cluster_labels1 = np.array([clusters1[i] for i in range(len(points))])  # Extract cluster labels
    cluster_labels2 = np.array([clusters2[i] for i in range(len(points))])
    cluster_labels3 = np.array([clusters3[i] for i in range(len(points))])

    # Check if the points are 2D
    if data.shape[1] != 2:
        raise ValueError("Only 2D data can be visualized. Your data has {} dimensions.".format(data.shape[1]))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(18, 6))
    fig.suptitle(title)
    ax1.scatter(data[:, 0], data[:, 1], c=cluster_labels1, cmap="viridis", s=50, alpha=0.6)
    ax1.title.set_text(title_1)
    ax2.scatter(data[:, 0], data[:, 1], c=cluster_labels2, cmap="viridis", s=50, alpha=0.6)
    ax2.title.set_text(title_2)
    ax3.scatter(data[:, 0], data[:, 1], c=cluster_labels3, cmap="viridis", s=50, alpha=0.6)
    ax3.title.set_text(title_3)


    ax1.grid()
    ax2.grid()
    ax3.grid()
    plt.show()


def method_group(row):
    m = row['Method']
    if "LS" in m:
        if 'everyone alone' in m:
            return 'LocStab-S'
        elif 'predicted number of clusters' in m:
            return 'LocStab-P'
        elif 'output of leiden' in m:
            return 'LocStab-Ld'
        elif 'output of k-means' in m:
            return 'LocStab-KM'
        elif 'output of dbscan' in m.lower():
            return 'LocStab-D'

    elif "LP" in m:
        if 'everyone alone' in m:
            return 'LocPop-S'
        elif 'predicted number of clusters' in m:
            return 'LocPop-P'
        elif 'output of k-means' in m:
            return 'LocPop-KM'
        elif 'output of dbscan' in m.lower():
            return 'LocPop-D'
        elif 'output of leiden' in m:
            return 'LocPop-Ld'

    elif 'louvain' in m.lower():
        return 'Louvain'
    elif 'leiden' in m.lower():
        return 'Leiden'
    elif 'kmeans' in m.lower():
        return 'K-means'
    elif 'dbscan' in m.lower():
        return 'DBSCAN'
    else:
        return 'Other'

def variant_label(row):
    m = row['Method']
    if 'Enemy-Averse' in m:
        return 'AE'
    elif 'Balanced' in m:
        return 'B'
    elif 'Friend-Oriented' in m:
        return 'AF'
    else:
        return 'Other'

def plot_custom_thresholds_with_kmeans_dbscan(dfs, labels, dataset_name, score_col='Rand Index', mode="LP"):
    # Add threshold column to each DataFrame
    for df, label in zip(dfs, labels):
        df['Threshold'] = str(label)  # make sure label is string for plotting

    # Combine all datasets
    df_all = pd.concat(dfs)

    # Filter for selected dataset
    df_all = df_all[df_all['Dataset'] == dataset_name].copy()
 
    df_all['method_group'] = df_all.apply(method_group, axis=1)
    df_all['variant'] = df_all.apply(variant_label, axis=1)

    # Define order and positions
    if mode == "LP":
        method_order = ['K-means', 'DBSCAN', 'LocPop-S', 'LocPop-P', 'LocPop-KM', 'LocPop-D']
    else: 
        method_order = ['K-means', 'DBSCAN', 'LocStab-S', 'LocStab-P', 'LocStab-KM', 'LocStab-D']
    method_pos = {method: i for i, method in enumerate(method_order)}

    variant_colors = {
        'B': 'green',
        'AF': 'orange',
        'AE': 'blue'
    }
    threshold_markers = {
        str(labels[0]): '^',
        str(labels[1]): 'o',
        str(labels[2]): 's'
    }
    variant_offset = {
        'AF': -0.2,
        'B': 0.0,
        'AE': 0.2
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot LP heuristics
    for method in method_order[2:]:  # skip kmeans/dbscan
        for variant in ['B', 'AF', 'AE']:
            for threshold in threshold_markers.keys():
                sub_df = df_all[
                    (df_all['method_group'] == method) &
                    (df_all['variant'] == variant) &
                    (df_all['Threshold'] == threshold)
                    ]
                x_pos = method_pos[method] + variant_offset[variant] + 2

                # Extract means and stds
                y_vals = [val[0] for val in sub_df[score_col].values]
                y_stds = [val[1] for val in sub_df[score_col].values]

                # Plot mean points
                ax.scatter(
                    [x_pos] * len(y_vals), y_vals,
                    color=variant_colors[variant],
                    marker=threshold_markers[threshold],
                    s=80, edgecolor='black', alpha=0.8,
                    label=f'{variant} - {threshold}' if (threshold == str(labels[0])) else None
                )

                # Plot 2-sigma error bars
                for x, y, std in zip([x_pos] * len(y_vals), y_vals, y_stds):
                    ax.errorbar(
                        x, y,
                        yerr=2 * std,
                        fmt='none',
                        ecolor=variant_colors[variant],
                        elinewidth=1,
                        capsize=3,
                        alpha=0.4
                    )

    # Plot average point for kmeans and dbscan
    for method in ['K-means', 'DBSCAN']:
        if method not in method_pos:
            continue  # skip if method not recognized

        sub_df = df_all[df_all['method_group'] == method]
        if sub_df.empty or sub_df[score_col].isna().all():
            continue  # skip if no valid scores

        y_vals = [val[0] for val in sub_df[score_col].dropna().values]
        y_stds = [val[1] for val in sub_df[score_col].dropna().values]

        avg = np.mean(y_vals)
        std = np.sqrt(np.mean([s ** 2 for s in y_stds]))

        ax.scatter(
            method_pos[method] + 2, avg,
            color='black', marker='o', s=80, edgecolor='white',
            label=None
        )

        ax.errorbar(
            method_pos[method] + 2, avg,
            yerr=2 * std,
            fmt='none',
            ecolor='black',
            elinewidth=1,
            capsize=3,
            alpha=0.6
        )

    ax.set_xticks(range(2, len(method_order) + 2))
    ax.set_xticklabels(method_order, rotation=90)
    ax.set_ylabel(score_col)
    ax.set_title(f'{score_col} by Method on {dataset_name}')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Color legend
    color_legend = [
        Line2D([0], [0], marker='o', color='w', label=v, markerfacecolor=c, markersize=10, markeredgecolor='black')
        for v, c in variant_colors.items()]
    shape_legend = [
        Line2D([0], [0], marker=m, color='w', label=thresh, markerfacecolor='gray', markersize=10,
               markeredgecolor='black')
        for thresh, m in threshold_markers.items()
    ]

    # Combine legends
    handles = color_legend + shape_legend
    ax.legend(
        handles=handles,
        title='',
        loc='upper right',
        bbox_to_anchor=(1.26, 1),
        borderaxespad=0.0,
        fontsize=16,
        ncol=1
    )
    # plt.ylim(0,1.1)
    plt.tight_layout()
    plt.xticks(rotation=0)
    plt.subplots_adjust(right=0.75)  # Space for legend

    return fig, ax

def try_parse_tuple_clustering(x):
    # Handle actual tuples or lists
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return (float(x[0]), float(x[1]))

    # Handle strings like '(0.85, 0.05)'
    if isinstance(x, str) and x.startswith('('):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, (tuple, list)) and len(parsed) == 2:
                return (float(parsed[0]), float(parsed[1]))
        except Exception:
            pass

    # Handle known invalids like 'nan', 'n.A.', empty string, etc.
    return (np.nan, 0.0)

def normalize_score_column_clustering(df, score_cols):
    for col in score_cols:
        df[col] = df[col].apply(try_parse_tuple_clustering)
    return df

def plot_and_save_clustering(dfs,labels, dataset, score, mode ="LP", save_path= None):
    plt.rcParams.update({'font.size': 18})
    for i in range(len(dfs)):
        dfs[i].replace("n.A.", np.nan, inplace= True)
        dfs[i] = normalize_score_column_clustering(dfs[i], [score])

    fig, ax = plot_custom_thresholds_with_kmeans_dbscan(
    dfs=dfs,
    labels=labels,
    dataset_name=dataset,
    score_col=score,
    mode = mode)
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Figure saved as {save_path}")
    plt.show()





def plot_custom_thresholds_with_louvain_leiden(dfs, labels, dataset_name, score_col='Rand Index', mode = "LP"):
    plt.rcParams.update({'font.size': 18})
    # Add threshold column to each DataFrame
    for df, label in zip(dfs, labels):
        df['Threshold'] = str(label)  # make sure label is string for plotting

    # Combine all datasets
    df_all = pd.concat(dfs)

    # Filter for selected dataset
    df_all = df_all[df_all['Dataset'] == dataset_name].copy()

  

    df_all['method_group'] = df_all.apply(method_group, axis=1)
    df_all['variant'] = df_all.apply(variant_label, axis=1)

    # Define order and positions
    if mode == "LS":
        method_order = ['Louvain', 'Leiden', 'LocStab-S', 'LocStab-P','LocStab-Ld']
    else:
        method_order = ['Louvain', 'Leiden', 'LocPop-S', 'LocPop-P','LocPop-Ld']
    method_pos = {method: i for i, method in enumerate(method_order)}

    variant_colors = {
        'B': 'green',
        'AF': 'orange',
        'AE': 'blue'
    }
    threshold_markers = {
        str(labels[0]): '^',
        str(labels[1]): 'o',
        str(labels[2]): 's'
    }
    variant_offset = {
        'AF': -0.2,
        'B': 0.0,
        'AE': 0.2
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot LP heuristics
    for method in method_order[2:]:  # skip Leiden/Louvain
        for variant in ['B', 'AF', 'AE']:
            for threshold in threshold_markers.keys():
                sub_df = df_all[
                    (df_all['method_group'] == method) &
                    (df_all['variant'] == variant) &
                    (df_all['Threshold'] == threshold)
                    ]
                if sub_df.empty:
                    continue

                x_pos = method_pos[method] + variant_offset[variant] + 2
                means = sub_df[score_col].apply(lambda x: float(x[0]) if not pd.isna(x[0]) else np.nan).values
                stds = sub_df[score_col].apply(lambda x: float(x[1]) if not pd.isna(x[1]) else np.nan).values

                ax.scatter(
                    [x_pos] * len(means), means,
                    color=variant_colors[variant],
                    marker=threshold_markers[threshold],
                    s=80, edgecolor='black', alpha=0.8,
                    label=f'{variant} - {threshold}' if (threshold == str(labels[0])) else None
                )

                # Plot 2-sigma error bars
                for mean, std in zip(means, stds):
                    ax.plot(
                        [x_pos, x_pos],
                        [mean - 2 * std, mean + 2 * std],
                        color=variant_colors[variant],
                        alpha=0.4,
                        linewidth=1
                    )

    # Plot Leiden and Louvain averages
    for method in ['Leiden', 'Louvain']:
        if method not in method_pos:
            continue  # skip if method not recognized

        sub_df = df_all[df_all['method_group'] == method]
        if sub_df.empty or sub_df[score_col].isna().all():
            continue  # skip if no valid scores

        y_vals = [val[0] for val in sub_df[score_col].dropna().values]
        y_stds = [val[1] for val in sub_df[score_col].dropna().values]

        avg = np.mean(y_vals)
        std = np.sqrt(np.mean([s ** 2 for s in y_stds]))

        ax.scatter(
            method_pos[method] + 2, avg,
            color='black', marker='o', s=80, edgecolor='white',
            label=None
        )

        ax.errorbar(
            method_pos[method] + 2, avg,
            yerr=2 * std,
            fmt='none',
            ecolor='black',
            elinewidth=1,
            capsize=3,
            alpha=0.6
        )

    ax.set_xticks(range(2, len(method_order) + 2))
    ax.set_xticklabels(method_order, rotation=90)
    ax.set_ylabel(score_col)
    ax.set_title(f'{score_col} by Method on {dataset_name}')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Color legend
    color_legend = [
        Line2D([0], [0], marker='o', color='w', label=v, markerfacecolor=c, markersize=10, markeredgecolor='black')
        for v, c in variant_colors.items()
    ]
    shape_legend = [
        Line2D([0], [0], marker=m, color='w', label=thresh, markerfacecolor='gray', markersize=10,
               markeredgecolor='black')
        for thresh, m in threshold_markers.items()
    ]

    # Adjust legend position
    plt.legend(
        handles=color_legend + shape_legend,
        title='',
        loc='upper right',
        bbox_to_anchor=(1.28, 1),
        borderaxespad=0.5,
        fontsize=16,
        ncol=1
    )
    # plt.ylim(0,1.1)
    plt.tight_layout()
    plt.xticks(rotation=10)
    plt.subplots_adjust(right=0.75)

    return fig, ax
    
def try_parse_tuple_community(x):
    # Handle actual tuples or lists
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return (float(x[0]), float(x[1]))

    # Handle strings like '(0.85, 0.05)'
    if isinstance(x, str) and x.startswith('('):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, (tuple, list)) and len(parsed) == 2:
                return (float(parsed[0]), float(parsed[1]))
        except Exception:
            pass

    # Handle known invalids like 'nan', 'n.A.', empty string, etc.
    return (np.nan, 0.0)    
    

def normalize_score_column_community(df, score_cols):
    for col in score_cols:
        df[col] = df[col].apply(try_parse_tuple_community)
    return df
    


def plot_and_save_community(dfs,labels, dataset, score, mode = "LP", save_path= None):
    for i in range(len(dfs)):
        dfs[i].replace("n.A.", np.nan, inplace= True)
        dfs[i] = normalize_score_column_community(dfs[i], [score])

    fig, ax = plot_custom_thresholds_with_louvain_leiden(
    dfs=dfs,
    labels=labels,
    dataset_name=dataset,
    score_col=score,
    mode = mode)
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Figure saved as {save_path}")
    plt.show()