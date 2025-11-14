import numpy as np
import pandas as pd
import umap.umap_ as umap
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import json
import sys
import os

# Configure matplotlib for Hebrew text support
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Noto Sans Hebrew', 'David', 'Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Ensure console can display Hebrew
if sys.platform.startswith('win'):
    os.system('chcp 65001 > nul')


def fix_hebrew_text(text):
    """Fix Hebrew text display by adding RTL markers"""
    if not text:
        return text

    # Check if text contains Hebrew characters
    has_hebrew = any('\u0590' <= char <= '\u05FF' for char in text)

    if has_hebrew:
        # Add RTL override and pop directional formatting
        return '\u202E' + text + '\u202C'

    return text


def load_embeddings_from_json(file_path="../../output/embedding/embeddings.json"):
    """Load embeddings from JSON file and convert back to DataFrame format"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert back to DataFrame
    posts_data = []
    for post in data['posts']:
        post_dict = post.copy()
        posts_data.append(post_dict)

    df = pd.DataFrame(posts_data)

    # Convert embedding lists back to numpy arrays
    def convert_to_array(x):
        try:
            if isinstance(x, list):
                arr = np.array(x)
                if arr.shape == (768,):
                    return arr
                else:
                    return np.zeros(768)
            else:
                return np.zeros(768)
        except:
            return np.zeros(768)

    df['embedded'] = df['embedding'].apply(convert_to_array)
    df = df.drop('embedding', axis=1)

    print(f"Loaded {len(df)} posts with embeddings")
    return df


def create_interactive_map(df, embedding_column='embedded', labels=None,
                           title="Interactive UMAP Projection"):
    """Create interactive UMAP visualization with multi-selection capability"""

    # Stack all embeddings into a single matrix
    embedding_matrix = np.vstack(df[embedding_column].values)
    print(f"Embedding matrix shape: {embedding_matrix.shape}")

    # Reduce dimensions
    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    reduced = reducer.fit_transform(embedding_matrix)
    print(f"UMAP completed. Reduced to shape: {reduced.shape}")

    # Create the interactive plot
    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot points with proper cluster colors and legend
    if labels is not None:
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Choose colormap based on number of clusters
        if n_clusters <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        elif n_clusters <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

        # Create color mapping
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        point_colors = [label_to_color[label] for label in labels]

        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=point_colors, s=30, alpha=0.7)

        # Create custom legend
        legend_elements = [mpatches.Patch(color=label_to_color[label], label=f'Cluster {label}')
                           for label in unique_labels]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

        title += f" ({n_clusters} clusters)"
    else:
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], s=30, alpha=0.7, c='blue')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Instructions
    instructions = fix_hebrew_text(
        "Left click: Select/deselect posts\nRight click: Show post details\nPress 'p' to print selected IDs\nPress 'r' to remove selected posts\nלחיצה שמאלית: בחירה/ביטול בחירה\nלחיצה ימנית: הצגת פרטי הפוסט\nלחץ 'p' להדפסת מזהים נבחרים\nלחץ 'r' להסרת פוסטים נבחרים")
    textbox = ax.text(0.02, 0.98, instructions,
                      transform=ax.transAxes, fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                      wrap=True, ha='left')

    # Store selected points and their highlights
    selected_points = set()
    highlights = {}
    removed_points = set()  # Track removed points

    def update_visualization():
        """Update the visualization after removing points"""
        nonlocal scatter, reduced, df, labels

        if not removed_points:
            return

        # Create mask for non-removed points
        remaining_mask = ~np.isin(range(len(df)), list(removed_points))

        # Filter data
        remaining_reduced = reduced[remaining_mask]
        remaining_df = df[remaining_mask].reset_index(drop=True)

        if labels is not None:
            remaining_labels = labels[remaining_mask]
        else:
            remaining_labels = None

        # Clear the current plot
        ax.clear()

        # Replot with remaining points
        if remaining_labels is not None:
            unique_labels = np.unique(remaining_labels)
            n_clusters = len(unique_labels)

            if n_clusters <= 10:
                colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            elif n_clusters <= 20:
                colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
            else:
                colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

            label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
            point_colors = [label_to_color[label] for label in remaining_labels]

            scatter = ax.scatter(remaining_reduced[:, 0], remaining_reduced[:, 1],
                                 c=point_colors, s=30, alpha=0.7)

            # Recreate legend
            legend_elements = [mpatches.Patch(color=label_to_color[label], label=f'Cluster {label}')
                               for label in unique_labels]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

            current_title = f"Interactive UMAP Projection ({n_clusters} clusters) - {len(remaining_df)} posts"
        else:
            scatter = ax.scatter(remaining_reduced[:, 0], remaining_reduced[:, 1],
                                 s=30, alpha=0.7, c='blue')
            current_title = f"Interactive UMAP Projection - {len(remaining_df)} posts"

        ax.set_title(current_title, fontsize=14, fontweight='bold')
        ax.set_xlabel("UMAP-1", fontsize=12)
        ax.set_ylabel("UMAP-2", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Re-add instructions
        ax.text(0.02, 0.98, instructions,
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                wrap=True, ha='left')

        # Update global variables
        reduced = remaining_reduced
        df = remaining_df
        if labels is not None:
            labels = remaining_labels

        # Clear selections since indices have changed
        selected_points.clear()
        highlights.clear()

        fig.canvas.draw()
        print(f"Visualization updated. {len(df)} posts remaining.")

    def on_click(event):
        if event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        click_point = np.array([event.xdata, event.ydata])
        distances = np.sum((reduced - click_point) ** 2, axis=1)
        closest_idx = np.argmin(distances)

        if event.button == 1:  # Left click - select/deselect
            if closest_idx in selected_points:
                # Deselect
                selected_points.remove(closest_idx)
                if closest_idx in highlights:
                    highlights[closest_idx].remove()
                    del highlights[closest_idx]
                print(f"Deselected post #{closest_idx}")
            else:
                # Select
                selected_points.add(closest_idx)
                highlight = ax.scatter(reduced[closest_idx, 0], reduced[closest_idx, 1],
                                       s=150, facecolors='none', edgecolors='red', linewidths=3)
                highlights[closest_idx] = highlight
                print(f"Selected post #{closest_idx}")

            print(f"Currently selected: {len(selected_points)} posts")
            fig.canvas.draw_idle()

        elif event.button == 3:  # Right click - show details
            post_info = df.iloc[closest_idx]

            print(f"\n{'=' * 40}")
            print(f"POST #{closest_idx}")
            print(f"{'=' * 40}")
            print(f"Text: {post_info['post_text'][:200]}{'...' if len(post_info['post_text']) > 200 else ''}")

            if 'cluster' in post_info:
                print(f"Cluster: {post_info['cluster']}")

            print(f"UMAP coords: ({reduced[closest_idx, 0]:.3f}, {reduced[closest_idx, 1]:.3f})")
            print(f"{'=' * 40}\n")

    def on_key(event):
        if event.key == 'p':
            if selected_points:
                selected_list = sorted(list(selected_points))
                print(f"\n{'=' * 50}")
                print(f"SELECTED POST IDs ({len(selected_list)} posts):")
                print(f"{'=' * 50}")
                print(selected_list)
                print(f"{'=' * 50}\n")

                # Also save to file
                with open("selected_posts.txt", "w") as f:
                    f.write("Selected post IDs:\n")
                    f.write(str(selected_list))
                print("Selected IDs saved to 'selected_posts.txt'")
            else:
                print("No posts selected")

        elif event.key == 'r':  # Remove selected posts
            if selected_points:
                confirm = input(
                    f"\nAre you sure you want to remove {len(selected_points)} selected posts? (y/n): ").strip().lower()
                if confirm == 'y':
                    # Add selected points to removed set
                    removed_points.update(selected_points)
                    print(f"Removing {len(selected_points)} posts...")

                    # Update visualization
                    update_visualization()
                else:
                    print("Removal cancelled")
            else:
                print("No posts selected for removal")

    # Connect events
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.tight_layout()
    plt.show()

    return reduced, selected_points


def load_selected_posts(file_path="selected_posts.txt"):
    """Load previously selected post IDs from file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Extract list from the file content
            import ast
            lines = content.split('\n')
            for line in lines:
                if '[' in line and ']' in line:
                    return ast.literal_eval(line.strip())
        return []
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error loading selected posts: {e}")
        return []


def launch_interactive_explorer(embedding_file_path=None):
    """Main function to launch the interactive embedding explorer"""
    print("=" * 60)
    print("INTERACTIVE EMBEDDING EXPLORER")
    print(fix_hebrew_text("חוקר הטמעות אינטראקטיבי"))
    print("=" * 60)

    # Load embeddings
    if embedding_file_path is None:
        embedding_file_path = "../../output/embedding/embeddings.json"

    try:
        df = load_embeddings_from_json(embedding_file_path)
        print(f"Successfully loaded {len(df)} posts")
    except FileNotFoundError:
        print(f"Could not find embedding file: {embedding_file_path}")
        return
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return

    # Check for previously selected posts to remove
    selected_to_remove = load_selected_posts()
    if selected_to_remove:
        print(f"\nFound {len(selected_to_remove)} previously selected posts for removal")
        remove_choice = input("Remove these posts before visualization? (y/n): ").strip().lower()

        if remove_choice == 'y':
            original_count = len(df)
            # Remove posts by index (convert to boolean mask)
            mask = ~df.index.isin(selected_to_remove)
            df = df[mask].reset_index(drop=True)
            removed_count = original_count - len(df)
            print(f"Removed {removed_count} posts. {len(df)} posts remaining.")

            # Clear the selected_posts.txt file
            with open("selected_posts.txt", "w") as f:
                f.write("Selected post IDs:\n[]")
            print("Cleared selected_posts.txt file")
        else:
            print("Keeping all posts for visualization")

    # Check if clusters are available
    has_clusters = 'cluster' in df.columns

    # Create interactive map
    print("\nCreating interactive UMAP visualization...")
    print("Instructions:")
    print("- Left click: Select/deselect posts for deletion")
    print("- Right click: Show post details")
    print("- Press 'p' key: Print selected post IDs")
    print("- Press 'r' key: Remove selected posts from visualization")
    print(fix_hebrew_text("- לחיצה שמאלית: בחירת/ביטול בחירת פוסטים למחיקה"))
    print(fix_hebrew_text("- לחיצה ימנית: הצגת פרטי הפוסט"))
    print(fix_hebrew_text("- לחץ 'p': הדפסת מזהי הפוסטים הנבחרים"))
    print(fix_hebrew_text("- לחץ 'r': הסרת פוסטים נבחרים מהתצוגה"))

    if has_clusters:
        reduced_coords, selected_points = create_interactive_map(
            df,
            labels=df['cluster'].values,
            title="Interactive UMAP Projection (Clustered)"
        )
    else:
        reduced_coords, selected_points = create_interactive_map(
            df,
            title="Interactive UMAP Projection"
        )

    if selected_points:
        print(f"\nFinal selection: {len(selected_points)} posts selected")
        print(f"Selected IDs: {sorted(list(selected_points))}")
    else:
        print("\nNo posts selected")


if __name__ == "__main__":
    launch_interactive_explorer()