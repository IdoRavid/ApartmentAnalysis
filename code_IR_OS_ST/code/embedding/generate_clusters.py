import json

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import plotly.express as px
import plotly.io as pio
from embed_posts import validate_embeddings, load_embeddings_from_json
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from umap import umap_ as umap




import os
from datetime import datetime

def visualize_embeddings(df, embedding_column='embedded', labels=None, method='kmeans', method_params=None,
                         n_neighbors=15, min_dist=0.1, title="UMAP Projection"):
    """
    Visualize embeddings in 2D using UMAP.

    Args:
        df (pd.DataFrame): Your DataFrame containing embeddings.
        embedding_column (str): Column name with embedding vectors.
        labels (array-like): Optional labels for coloring.
        method (str): Name of clustering method (for title/file naming).
        method_params (dict): Parameters used (to annotate plot).
        n_neighbors, min_dist: UMAP params.
        title (str): Base title for the plot.
    """
    if not validate_embeddings(df, embedding_column):
        print("Cannot visualize due to invalid embeddings")
        return

    try:
        embedding_matrix = np.vstack(df[embedding_column].values)
        print(f"Embedding matrix shape: {embedding_matrix.shape}")
    except ValueError as e:
        print(f"Error stacking embeddings: {e}")
        return

    reducer = umap.UMAP(n_components=3,n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine', random_state=42)
    reduced = reducer.fit_transform(embedding_matrix)

    # Build full title string
    param_str = ""
    if method_params:
        param_str = ", ".join(f"{k}={v}" for k, v in method_params.items())

    full_title = f"{title} using {method.upper()}"
    if param_str:
        full_title += f" ({param_str})"

    # Plot
    plt.figure(figsize=(12, 10))
    if labels is not None:
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        cmap = 'tab10' if n_clusters <= 10 else 'tab20' if n_clusters <= 20 else 'viridis'
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap=cmap, s=20, alpha=0.7)
        plt.colorbar(scatter, label='Cluster')

        full_title += f" — {n_clusters} clusters"

    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], s=20, alpha=0.7)

    plt.title(full_title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save with dynamic name
    output_dir = "../../output/embedding/"
    os.makedirs(output_dir, exist_ok=True)
    file_base = f"umap_{method.lower()}_{param_str}.png"
    output_path = os.path.join(output_dir, file_base)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved UMAP plot to {output_path}")

    fig = px.scatter_3d(
        x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
        color=labels,
        title=full_title,
        opacity=0.7
    )
    pio.renderers.default = 'browser'
    # fig.show()
    output_dir = "../../output/embedding/"
    os.makedirs(output_dir, exist_ok=True)
    file_base_3d = f"umap_3D_{method.lower()}_{param_str}.html"
    output_path_3d = os.path.join(output_dir, file_base_3d)
    fig.write_html(output_path_3d)

    return reduced



def save_posts_with_clusters_to_csv(df, output_path="../../output/embedding/posts_with_clusters.csv"):
    """
    Save posts with cluster information to CSV for manual analysis.

    Args:
        df (pd.DataFrame): DataFrame with posts and cluster information
        output_path (str): Path where to save the CSV file
    """
    # Create a copy for CSV export (without embedding vectors)
    csv_df = df.copy()

    # Remove the embedding column as it's not useful in CSV format
    if 'embedded' in csv_df.columns:
        csv_df = csv_df.drop('embedded', axis=1)

    # Reorder columns to put cluster first for easier analysis
    if 'cluster' in csv_df.columns:
        cols = ['cluster'] + [col for col in csv_df.columns if col != 'cluster']
        csv_df = csv_df[cols]

        # Sort by cluster for easier manual analysis
        csv_df = csv_df.sort_values('cluster')

    # Save to CSV
    csv_df.to_csv(output_path, index=False, encoding='utf-8-sig')  # utf-8-sig for Excel compatibility
    print(f"Posts with clusters saved to: {output_path}")

    # Print cluster statistics
    if 'cluster' in csv_df.columns:
        cluster_stats = csv_df['cluster'].value_counts().sort_index()
        print("\nCluster Statistics:")
        print("Cluster | Count")
        print("--------|------")
        for cluster, count in cluster_stats.items():
            print(f"   {cluster:2d}   |  {count:4d}")
        print(f"Total   |  {len(csv_df):4d}")

    return csv_df

def save_results_with_clusters(df, method, method_params):
    """Save DataFrame with cluster information back to JSON"""
    data_for_json = {
        'metadata': {
            'model': 'onlplab/alephbert-base',
            'embedding_dimension': len(df['embedded'].iloc[0]) if len(df) > 0 else 0,
            'total_posts': len(df),
            'has_clusters': 'cluster' in df.columns,
            'clustering_method': method,
            'clustering_params': method_params
        },
        'posts': []
    }

    for idx, row in df.iterrows():
        post_data = {
            'index': int(row['index']) if 'index' in row else int(idx),
            'post_text': row['post_text'],
            'embedding': row['embedded'].tolist(),
        }

        # Add cluster information if available
        if 'cluster' in row:
            post_data['cluster'] = int(row['cluster'])

        # Add any other columns from the DataFrame
        for col in df.columns:
            if col not in ['post_text', 'embedded', 'index', 'cluster']:
                # post_data[col] = row[col] if pd.notna(row[col]) else None
                val = row[col]
                if isinstance(val, (list, np.ndarray)):
                    post_data[col] = val.tolist() if isinstance(val, np.ndarray) else val
                elif pd.notna(val):
                    post_data[col] = val
                else:
                    post_data[col] = None

        data_for_json['posts'].append(post_data)

    # Save to JSON file with clusters
    with open("../../output/embedding/embedding_with_clusters.json", 'w', encoding='utf-8') as f:
        json.dump(data_for_json, f, ensure_ascii=False, indent=2)

    print(f"Results with clusters saved to embedded_with_clusters.json")

def mean_pairwise_cosine_distance(centroids):
    # centroids: (k, dim)
    distances = cosine_distances(centroids)
    # Exclude self-distances (diagonal)
    upper_tri_indices = np.triu_indices_from(distances, k=1)
    pairwise_vals = distances[upper_tri_indices]
    return pairwise_vals.mean()

def cluster_embeddings(df, embedding_column='embedded', method='kmeans',
                       n_clusters=15, eps=0.5, min_samples=5, linkage='ward',
                       cluster_column='cluster'):
    """
    Cluster embeddings using the specified method and assign cluster labels to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing embeddings.
        embedding_column (str): Column name with embeddings.
        method (str): 'kmeans', 'dbscan', or 'agglo'.
        n_clusters (int): For KMeans or Agglomerative.
        eps (float): For DBSCAN.
        min_samples (int): For DBSCAN.
        cluster_column (str): Name of output column.

    Returns:
        pd.DataFrame: DataFrame with added cluster column.
    """
    if not validate_embeddings(df, embedding_column):
        print("Cannot cluster due to invalid embeddings")
        return df

    embedding_matrix = np.vstack(df[embedding_column].values)

    if method == 'kmeans':
        print(f"Clustering with KMeans (k={n_clusters})...")
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(embedding_matrix)
        mean_cosine_dist = mean_pairwise_cosine_distance(model.cluster_centers_)
        print(f"Mean pairwise cosine distance between cluster centroids (k={n_clusters}): {mean_cosine_dist:.4f}")

    elif method == 'dbscan':
        print(f"Clustering with DBSCAN (eps={eps}, min_samples={min_samples})...")
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(embedding_matrix)

    elif method == 'agglo':
        print(f"Clustering with Agglomerative (k={n_clusters})...")
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(embedding_matrix)

    else:
        raise ValueError(f"Unknown method: {method}")

    df[cluster_column] = labels
    return df

def main():
    df = load_embeddings_from_json("../../output/embedding/embedded_50.json")
    method = 'kmeans'  # ← try: 'kmeans', 'dbscan', 'agglo'

    LINKAGE = 'ward'
    K_VALUE = 8
    EPS_VALUE = 0.5
    MIN_SAMPLES_VALUE = 20
    method_params = {}

    print("=== Clustering embeddings ===")
    if method == 'dbscan':
        df = cluster_embeddings(df, method=method, eps=EPS_VALUE, min_samples=MIN_SAMPLES_VALUE)
        method_params = {'eps': EPS_VALUE, 'min_samples': MIN_SAMPLES_VALUE}
    if method == 'kmeans':
        df = cluster_embeddings(df, method=method, n_clusters=K_VALUE)
        method_params = {'k': K_VALUE}
    if method == 'agglo':
        df = cluster_embeddings(df, method=method, n_clusters=K_VALUE)
        method_params = {'k': K_VALUE, 'linkage': LINKAGE}

    print("=== Visualizing embeddings with clusters ===")
    if 'cluster' in df.columns:
        visualize_embeddings(
            df,
            labels=df['cluster'],
            method=method,
            method_params=method_params
        )

    print("=== Saving results ===")
    save_results_with_clusters(df, method, method_params)
    save_posts_with_clusters_to_csv(df)

    return df

if __name__ == '__main__':
    df = main()