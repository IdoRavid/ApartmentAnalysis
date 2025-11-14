from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import pandas as pd
import re
import json
import torch


def simple_hebrew_sentence_tokenizer(text):
    # Splits on ".", "?", "!" followed by space or end of line
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def embed_paragraph(paragraph, model):
    # Handle empty or very short paragraphs
    if not paragraph or len(paragraph.strip()) < 3:
        # Return zero vector with correct dimensions for AlephBERT
        return np.zeros(768)

    # Filter to keep only Hebrew words (no numbers or other characters)
    filtered_paragraph = filter_hebrew_words_only(paragraph)

    # If no Hebrew words remain after filtering, return zero vector
    if not filtered_paragraph or len(filtered_paragraph.strip()) < 3:
        return np.zeros(768)

    sentences = simple_hebrew_sentence_tokenizer(filtered_paragraph)
    if not sentences:
        # If no sentences found, return zero vector
        return np.zeros(768)

    sentence_embeddings = model.encode(sentences)

    # Ensure we have valid embeddings
    if sentence_embeddings.size == 0:
        return np.zeros(768)

    paragraph_embedding = np.mean(sentence_embeddings, axis=0)

    # Validate embedding dimensions
    if paragraph_embedding.shape[0] != 768:
        print(f"Warning: Unexpected embedding dimension {paragraph_embedding.shape[0]}, expected 768")
        return np.zeros(768)

    return paragraph_embedding


def filter_hebrew_words_only(text):
    """
    Filter text to keep only Hebrew words, removing numbers and non-Hebrew characters.
    Hebrew Unicode range: \u0590-\u05FF (includes Hebrew letters, vowels, and punctuation)
    """
    # Remove numbers (both Arabic numerals and Hebrew numerals)
    text_no_numbers = re.sub(r'[\d\u05D0-\u05EA]+(?=\d)|[\d]+', '', text)

    # Keep only Hebrew letters, spaces, and basic Hebrew punctuation
    # Hebrew letters: \u05D0-\u05EA
    # Hebrew vowels and marks: \u05B0-\u05C7
    # Keep spaces and basic sentence punctuation for sentence structure
    hebrew_pattern = r'[\u05D0-\u05EA\u05B0-\u05C7\s\.\?\!\,\;\:]+'

    # Find all Hebrew word sequences
    hebrew_matches = re.findall(hebrew_pattern, text_no_numbers)

    # Join the matches and clean up extra spaces
    filtered_text = ' '.join(hebrew_matches)
    filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()

    return filtered_text


def analyze_sentiment(text, sentiment_pipeline):
    """
    Analyze sentiment of Hebrew text using heBERT sentiment analysis model.

    Args:
        text (str): Hebrew text to analyze
        sentiment_pipeline: Pre-loaded sentiment analysis pipeline

    Returns:
        dict: Contains sentiment label, score, and confidence
    """
    if not text or len(text.strip()) < 3:
        return {
            'sentiment': 'neutral',
            'score': 0.0,
            'confidence': 0.0
        }

    # Filter to Hebrew words only for better sentiment analysis
    filtered_text = filter_hebrew_words_only(text)

    if not filtered_text or len(filtered_text.strip()) < 3:
        return {
            'sentiment': 'neutral',
            'score': 0.0,
            'confidence': 0.0
        }

    try:
        # Run sentiment analysis
        result = sentiment_pipeline(filtered_text)

        # Extract results
        label = result[0]['label'].lower()
        confidence = result[0]['score']

        # Map labels to standardized sentiment names
        if label in ['positive', 'pos', '1']:
            sentiment = 'positive'
            score = confidence
        elif label in ['negative', 'neg', '0']:
            sentiment = 'negative'
            score = -confidence  # Negative score for negative sentiment
        else:
            sentiment = 'neutral'
            score = 0.0

        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': confidence
        }

    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return {
            'sentiment': 'neutral',
            'score': 0.0,
            'confidence': 0.0
        }


def create_sentiment_pipeline():
    """
    Create and return a sentiment analysis pipeline using heBERT sentiment model.
    """
    try:
        print("Loading heBERT sentiment analysis model...")

        # Load the Hebrew sentiment analysis model
        model_name = "avichr/heBERT_sentiment_analysis"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Create pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )

        print("heBERT sentiment analysis model loaded successfully!")
        return sentiment_pipeline

    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        print("Sentiment analysis will be skipped.")
        return None


def embed_posts():
    # Explicitly use AlephBERT model for embeddings
    print("Loading AlephBERT embedding model...")
    word_embedding_model = models.Transformer('onlplab/alephbert-base')
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )
    embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Load sentiment analysis model
    sentiment_pipeline = create_sentiment_pipeline()

    # Load data
    df = pd.read_csv("../../output/geolocating/posts_improved_geolocation.csv")
    df['post_text'] = df['post_text'].fillna('')  # replace NaN with empty string

    # Generate embeddings and sentiment analysis
    embeddings = []
    sentiment_results = []
    problematic_indices = []

    print(f"Processing {len(df)} posts...")

    for idx, text in enumerate(df['post_text']):
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(df)} posts...")

        try:
            # Generate embedding
            embedding = embed_paragraph(text, embedding_model)
            # Validate embedding shape
            if embedding.shape != (768,):
                print(f"Row {idx}: Invalid embedding shape {embedding.shape}, text: '{text[:50]}...'")
                problematic_indices.append(idx)
                embedding = np.zeros(768)  # Use zero vector as fallback
            embeddings.append(embedding.tolist())  # Convert numpy array to list for JSON serialization

            # Analyze sentiment
            if sentiment_pipeline:
                sentiment_result = analyze_sentiment(text, sentiment_pipeline)
            else:
                sentiment_result = {
                    'sentiment': 'neutral',
                    'score': 0.0,
                    'confidence': 0.0
                }
            sentiment_results.append(sentiment_result)

        except Exception as e:
            print(f"Row {idx}: Error processing: {e}, text: '{text[:50]}...'")
            problematic_indices.append(idx)
            embeddings.append(np.zeros(768).tolist())  # Use zero vector as fallback
            sentiment_results.append({
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0
            })

    if problematic_indices:
        print(f"Found {len(problematic_indices)} problematic embeddings at indices: {problematic_indices[:10]}...")

    # Add results to DataFrame
    df['embedded'] = embeddings
    df['sentiment'] = [result['sentiment'] for result in sentiment_results]
    df['sentiment_score'] = [result['score'] for result in sentiment_results]
    df['sentiment_confidence'] = [result['confidence'] for result in sentiment_results]

    # Print sentiment distribution
    print("\nSentiment Distribution:")
    sentiment_counts = df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{sentiment}: {count} posts ({percentage:.1f}%)")

    # Save results
    save_embeddings_to_json(df, embeddings, problematic_indices, "embeddings")
    return df


def save_embeddings_to_json(df, embeddings, problematic_indices, filename):
    # Save to JSON format
    # Convert DataFrame to dictionary format suitable for JSON
    data_for_json = {
        'metadata': {
            'embedding_model': 'onlplab/alephbert-base',
            'sentiment_model': 'avichr/heBERT_sentiment_analysis',
            'embedding_dimension': len(embeddings[0]) if embeddings else 0,
            'total_posts': len(df),
            'problematic_indices': problematic_indices if problematic_indices else [],
            'sentiment_distribution': df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else {}
        },
        'posts': []
    }

    for idx, row in df.iterrows():
        post_data = {
            'index': int(idx),
            'post_text': row['post_text'],
            'embedding': row['embedded'] if isinstance(row['embedded'], list) else row['embedded'].tolist()
        }

        # Add sentiment data if available
        if 'sentiment' in row:
            post_data['sentiment'] = row['sentiment']
            post_data['sentiment_score'] = float(row['sentiment_score']) if pd.notna(row['sentiment_score']) else 0.0
            post_data['sentiment_confidence'] = float(row['sentiment_confidence']) if pd.notna(
                row['sentiment_confidence']) else 0.0

        # Add any other columns from the original DataFrame
        for col in df.columns:
            if col not in ['post_text', 'embedded', 'sentiment', 'sentiment_score', 'sentiment_confidence']:
                value = row[col]
                # Handle different data types safely
                if isinstance(value, np.ndarray):
                    # Convert numpy arrays to lists for JSON serialization
                    post_data[col] = value.tolist()
                elif pd.isna(value) if not isinstance(value, np.ndarray) else False:
                    post_data[col] = None
                else:
                    # Handle other types (int, float, string, etc.)
                    try:
                        # Convert numpy types to Python native types
                        if hasattr(value, 'item'):
                            post_data[col] = value.item()
                        else:
                            post_data[col] = value
                    except (ValueError, TypeError):
                        post_data[col] = str(value)

        data_for_json['posts'].append(post_data)

    # Save to JSON file
    with open(f"../../output/embedding/{filename}.json", 'w', encoding='utf-8') as f:
        json.dump(data_for_json, f, ensure_ascii=False, indent=2)

    print(f"Embeddings and sentiment analysis saved to JSON with {len(embeddings)} posts")
    print(f"Models used: AlephBERT (embeddings) + heBERT (sentiment)")


def load_embeddings_from_json(file="../../output/embedding/embeddings.json"):
    """Load embeddings and sentiment data from JSON file and convert back to DataFrame format"""
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        # Fallback to original file if sentiment version doesn't exist
        print(f"File {file} not found, trying original embedding file...")
        file = "../../output/embedding/embedding.json"
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # Convert back to DataFrame
    posts_data = []
    for post in data['posts']:
        post_dict = post.copy()
        posts_data.append(post_dict)

    df = pd.DataFrame(posts_data)

    # Auto-detect the target shape based on the first valid embedding
    def convert_to_array(x, expected_dim=None):
        if isinstance(x, list):
            arr = np.array(x)
        elif isinstance(x, np.ndarray):
            arr = x
        else:
            print(f"Warning: Invalid embedding type {type(x)}, replacing with zero vector")
            return np.zeros(expected_dim or 50)

        # Infer expected_dim if not given
        nonlocal_embedding_dim = expected_dim
        if nonlocal_embedding_dim is None:
            nonlocal_embedding_dim = len(arr)

        if arr.shape != (nonlocal_embedding_dim,):
            print(f"Warning: Embedding has shape {arr.shape}, expected ({nonlocal_embedding_dim},)")
            return np.zeros(nonlocal_embedding_dim)

        return arr

    # Detect embedding dimension from the first post
    first_shape = len(df['embedding'].iloc[0]) if len(df) > 0 else 768
    df['embedded'] = df['embedding'].apply(lambda x: convert_to_array(x, expected_dim=first_shape))
    df = df.drop('embedding', axis=1)

    model_info = data['metadata'].get('embedding_model', 'AlephBERT')
    sentiment_model = data['metadata'].get('sentiment_model', 'N/A')
    print(f"Loaded {len(df)} posts with embeddings from {model_info}")

    if 'sentiment' in df.columns:
        print(f"Sentiment analysis from {sentiment_model}")
        print("Sentiment distribution:")
        sentiment_counts = df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {sentiment}: {count} posts ({percentage:.1f}%)")

    # Check for problematic embeddings
    if 'problematic_indices' in data['metadata'] and data['metadata']['problematic_indices']:
        print(f"Note: {len(data['metadata']['problematic_indices'])} embeddings were problematic during generation")

    return df


def analyze_sentiment_distribution(df):
    """
    Analyze and display sentiment distribution statistics.
    """
    if 'sentiment' not in df.columns:
        print("No sentiment data found in DataFrame")
        return

    print("=== Sentiment Analysis Results ===")
    print(f"Total posts analyzed: {len(df)}")

    # Basic distribution
    sentiment_counts = df['sentiment'].value_counts()
    print("\nSentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {sentiment}: {count} posts ({percentage:.1f}%)")

    # Score statistics
    if 'sentiment_score' in df.columns:
        print(f"\nSentiment Score Statistics:")
        print(f"  Mean score: {df['sentiment_score'].mean():.3f}")
        print(f"  Median score: {df['sentiment_score'].median():.3f}")
        print(f"  Score range: {df['sentiment_score'].min():.3f} to {df['sentiment_score'].max():.3f}")

    # Confidence statistics
    if 'sentiment_confidence' in df.columns:
        print(f"\nConfidence Statistics:")
        print(f"  Mean confidence: {df['sentiment_confidence'].mean():.3f}")
        print(f"  Median confidence: {df['sentiment_confidence'].median():.3f}")
        print(f"  Low confidence posts (<0.7): {(df['sentiment_confidence'] < 0.7).sum()}")

    # Show some examples
    print(f"\nExample posts by sentiment:")
    for sentiment in sentiment_counts.index[:3]:  # Show top 3 sentiment categories
        example = df[df['sentiment'] == sentiment].iloc[0]
        text_preview = example['post_text'][:100] + "..." if len(example['post_text']) > 100 else example['post_text']
        score = example.get('sentiment_score', 'N/A')
        confidence = example.get('sentiment_confidence', 'N/A')
        print(f"  {sentiment} (score: {score:.3f}, conf: {confidence:.3f}): {text_preview}")


def validate_embeddings(df, embedding_column='embedded', expected_dim=None):
    """
    Validates that all embeddings in a DataFrame column are valid vectors
    of the same shape. Optionally, expected_dim can be enforced.
    """
    print("Validating embeddings...")

    for i, emb in enumerate(df[embedding_column]):
        # Convert list to array if needed
        if isinstance(emb, list):
            emb = np.array(emb)
        elif not isinstance(emb, np.ndarray):
            print(f"Row {i}: Invalid embedding type: {type(emb)}")
            return False

        # Determine expected dimension if not specified
        if expected_dim is None:
            expected_dim = emb.shape[0]

        if emb.shape != (expected_dim,):
            print(f"Row {i}: Embedding shape {emb.shape}, expected ({expected_dim},)")
            return False

    print(f"All embeddings have valid shapes ({expected_dim},)")
    return True


def lower_dimension(df, target_dim, embedding_column='embedded', file_name=None):
    """
    Reduce dimensionality of embeddings using PCA.

    Args:
        df (pd.DataFrame): DataFrame containing embeddings
        target_dim (int): Target number of dimensions
        embedding_column (str): Column name containing embeddings
        file_name (str): Optional filename to save results

    Returns:
        pd.DataFrame: DataFrame with added PCA column
    """
    if not validate_embeddings(df, embedding_column):
        print("Cannot reduce dimension due to invalid embeddings")
        return df

    print(f"Reducing embeddings from 768 to {target_dim} dimensions...")

    # Stack embeddings into matrix
    embeddings_matrix = np.vstack(df[embedding_column].values)
    print(f"Original embeddings matrix shape: {embeddings_matrix.shape}")

    # Apply PCA
    pca = PCA(n_components=target_dim)
    reduced_embeddings = pca.fit_transform(embeddings_matrix)
    print(f"Reduced embeddings matrix shape: {reduced_embeddings.shape}")

    # # Plot cumulative explained variance
    # import matplotlib.pyplot as plt
    # import os
    #
    # plt.figure(figsize=(8, 4))
    # plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    # plt.xlabel("Number of Principal Components")
    # plt.ylabel("Cumulative Explained Variance")
    # plt.title(f"PCA Scree Plot (target_dim = {target_dim})")
    # plt.grid(True)
    #
    # # Ensure output path exists
    # os.makedirs("../../output/embedding", exist_ok=True)
    # plt.show()
    # plt.savefig(f"../../output/embedding/pca_variance_{target_dim}.png")
    # plt.close()

    # Add reduced embeddings to DataFrame
    pca_column_name = f'{embedding_column}_pca_{target_dim}'
    df[pca_column_name] = list(reduced_embeddings)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_[:5]}...")  # Show first 5 components
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    # Save if filename provided
    if file_name:
        print(f"Saving results to {file_name}.json...")
        # Create a copy of df for saving, using the PCA embeddings as the main embedding
        df_to_save = df.copy()
        df_to_save['embedded'] = df_to_save[pca_column_name]  # Use PCA embeddings for saving

        # Convert embeddings to list format for JSON serialization
        embeddings_for_json = [emb.tolist() for emb in df_to_save['embedded']]

        save_embeddings_to_json(df_to_save, embeddings_for_json, [], file_name)

    return df


def debug_embeddings():
    """Debug function to inspect problematic embeddings"""
    try:
        df = load_embeddings_from_json()
        print(f"Loaded {len(df)} posts")

        # Check each embedding
        problematic_count = 0
        for idx, embedding in enumerate(df['embedded']):
            try:
                if not isinstance(embedding, np.ndarray):
                    print(f"Row {idx}: Not an array, type: {type(embedding)}, value: {embedding}")
                    problematic_count += 1
                elif embedding.shape != (768,):
                    print(f"Row {idx}: Shape {embedding.shape}, Text: '{df.iloc[idx]['post_text'][:100]}...'")
                    problematic_count += 1
            except Exception as e:
                print(f"Row {idx}: Error checking embedding: {e}")
                problematic_count += 1

        print(f"Found {problematic_count} problematic embeddings")

        # Show some statistics
        types = [type(emb).__name__ for emb in df['embedded']]
        unique_types = list(set(types))
        print(f"Unique embedding types found: {unique_types}")

        for emb_type in unique_types:
            count = types.count(emb_type)
            print(f"Type {emb_type}: {count} occurrences")

        # For numpy arrays, show shapes
        shapes = []
        for emb in df['embedded']:
            if isinstance(emb, np.ndarray):
                shapes.append(emb.shape)
            else:
                shapes.append(f"non_array_{type(emb).__name__}")

        unique_shapes = list(set(shapes))
        print(f"Unique shapes/types found: {unique_shapes}")

        for shape in unique_shapes:
            count = shapes.count(shape)
            print(f"Shape/Type {shape}: {count} occurrences")

    except Exception as e:
        print(f"Error during debugging: {e}")


def main():
    # Generate embeddings and sentiment analysis
    #df = embed_posts()
    #df.to_csv("../../output/embedding/embedded_posts.csv", index=False, encoding='utf-8-sig')
    # Analyze sentiment distribution
    #analyze_sentiment_distribution(df)

    # Debug embeddings if needed
    # debug_embeddings()

    # # Load and analyze existing embeddings with sentiment
    df = load_embeddings_from_json()
    #
    TARGET_DIM = 50
    # # Reduce dimensions if needed
    df = lower_dimension(df, TARGET_DIM, file_name=f"embedded_{TARGET_DIM}")
    #
    # # Show final sentiment analysis
    # analyze_sentiment_distribution(df)


if __name__ == '__main__':
    main()