import json
import os
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
import numpy as np

#TODO: look for more words to block, need to find labels for clusters.
HEBREW_STOPWORDS = [
        "של", "על", "עם", "גם", "אם", "זה", "או", "אז", "מה", "את", "לא", "כן", "כי",
        "כל", "יש", "הוא", "היא", "הם", "הן", "אנחנו", "אתם", "אתן", "אני", "אבל",  "עוד" , "היי"
        , "דירה" , "בדירה" , "הדירה" , "לדירת" , "חדרים" , "דירת" , "להשכרה" , "כולל" ,
        "ללא" , "עדיין" , "שוב", "מקפיץ" , "מקפיצה", "להיות" , "כולל" , "חדר" , "החדר", "בחדר" ,
        "מחפש", "מחפשת" , "עד" , "ללא" , "כולל" , "לי" , "עד", "מישהו", "לגור", "ואני", "אשמח", "להצעות",
        "הצעות", "אשמח", "להכל", "שצריך", "להכל", "מישהי", "מישהו", "מראה", "דירות", "לפרטים" , "להיכנס",
        "לדירה", "מחפשים", "מחפשות", "כניסה", "בהכל", "ואז", "לחפש", "משהו", "ואז", "קוראים", "נשארת", "ככה"
    ]

MAX_FREQ = 0.6
MIN_POSTS = 4

def load_clustered_json(path):
    """
    Load clustered posts from JSON and convert to DataFrame.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    posts = data['posts']
    df = pd.DataFrame(posts)

    # Convert embedding lists back to numpy-like (or leave as-is)
    df['embedded'] = df['embedding'].apply(lambda x: x if isinstance(x, list) else [])
    df.drop(columns='embedding', inplace=True)

    return df, data.get('metadata', {})

def evaluate_clustering(X, labels):
    """
    Compute silhouette and Davies-Bouldin scores
    :param X: TF-IDF or other feature matrix (N x D)
    :param labels: cluster assignment per document (N,)
    """
    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        print("Silhouette Score not valid for this number of labels.")
        return

    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X.toarray(), labels)
    print(f"Silhouette Score: {sil:.3f}")
    print(f"Davies-Bouldin Score: {dbi:.3f}")

def get_chi2_terms_per_cluster(df, text_col="post_text", cluster_col="cluster",
                                stopwords=None, min_df=3, max_df=0.7, top_n=10):
    """
    Return dict of cluster -> list of top chi² terms.
    """
    texts = df[text_col].fillna("").astype(str).values
    labels = df[cluster_col].values

    vectorizer = CountVectorizer(
        stop_words=stopwords,
        min_df=min_df,
        max_df=max_df,
        token_pattern=r"(?u)\b[^\d\W]+\b"
    )
    X = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    chi2_terms_per_cluster = {}

    for cluster_id in np.unique(labels):
        binary_labels = (labels == cluster_id).astype(int)
        scores, _ = chi2(X, binary_labels)
        top_indices = scores.argsort()[::-1][:top_n]
        chi2_terms_per_cluster[cluster_id] = list(terms[top_indices])

    return chi2_terms_per_cluster


def tfidf_and_chi2_terms(df, text_col='post_text', cluster_col='cluster',
                         top_n=10, output_path=None, stopwords=None):
    """
    For each cluster, write top TF-IDF and chi² terms (separately) to one CSV file,
    organized cluster-by-cluster.
    """
    records = []

    # === Precompute everything ===
    grouped_texts = df.groupby(cluster_col)[text_col].apply(lambda texts: ' '.join(texts.dropna().astype(str)))
    vectorizer = TfidfVectorizer(max_features=10000, stop_words=stopwords,
                                 max_df=MAX_FREQ, min_df=MIN_POSTS,
                                 token_pattern=r"(?u)\b[^\d\W]+\b")
    tfidf_matrix = vectorizer.fit_transform(grouped_texts)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_array = tfidf_matrix.toarray()

    chi2_dict = get_chi2_terms_per_cluster(
        df,
        text_col=text_col,
        cluster_col=cluster_col,
        stopwords=stopwords,
        top_n=top_n
    )

    # === Write cluster-by-cluster ===
    for cluster_id in sorted(df[cluster_col].unique()):
        # TF-IDF terms
        row = tfidf_array[cluster_id]
        top_indices = row.argsort()[-top_n:][::-1]
        for rank, i in enumerate(top_indices, 1):
            records.append({
                'cluster': cluster_id,
                'rank': rank,
                'type': 'tfidf',
                'term': feature_names[i],
                'score': round(row[i], 4)
            })

        # chi² terms
        chi2_terms = chi2_dict.get(cluster_id, [])
        for rank, term in enumerate(chi2_terms, 1):
            records.append({
                'cluster': cluster_id,
                'rank': rank,
                'type': 'chi2',
                'term': term,
                'score': ''
            })

    # === Write to CSV ===
    result_df = pd.DataFrame(records)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"TF-IDF and chi² summary saved to: {output_path}")

    return result_df



def main():
    input_path = "../../output/embedding/embedding_with_clusters.json"

    print("Loading clustered data...")
    df, metadata = load_clustered_json(input_path)
    method = metadata.get("clustering_method", "unknown")
    params = metadata.get("clustering_params", {})
    param_str = "_".join([f"{k}={v}" for k, v in params.items()])
    output_path = f"../../output/embedding/tfidf_chi2_terms_{method}_{param_str}.csv"

    print("\nEvaluating Clustering Quality...")
    vectorizer = TfidfVectorizer(
        stop_words=HEBREW_STOPWORDS,
        max_df=MAX_FREQ,
        min_df=MIN_POSTS,
        max_features=10000
    )
    clean_texts = df["post_text"].fillna("").astype(str).values
    tfidf_matrix = vectorizer.fit_transform(clean_texts)
    evaluate_clustering(tfidf_matrix, df["cluster"].values)

    print("Computing TF-IDF and chi2 top terms per cluster...")
    result_df = tfidf_and_chi2_terms(
        df,
        top_n=10,
        output_path=output_path,
        stopwords=HEBREW_STOPWORDS
    )

    # print("\nTop TF-IDF terms per cluster:")
    # for cluster_id in result_df['cluster'].unique():
    #     terms = result_df[result_df['cluster'] == cluster_id]
    #     size = df[df['cluster'] == cluster_id].shape[0]
    #     print(f"\nCluster {cluster_id} ({size} posts):")
    #     for _, row in terms.iterrows():
    #         print(f"  {row['term']:<15} {row['tfidf']}")


if __name__ == '__main__':
    main()
