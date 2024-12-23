import os
import pandas as pd
from gensim.models import Word2Vec


def load_labels(label_path):
    """Load labels from a .label file."""
    try:
        df = pd.read_csv(label_path, sep="\t", header=None, names=["phrase", "label"])
        print(f"Loaded labels from {label_path}.")
        return df
    except Exception as e:
        print(f"Error loading labels from {label_path}: {e}")
        return None


def load_contextual_data(context_dir):
    """Load contextual reviews for each cuisine."""
    context_data = {}
    if not os.path.exists(context_dir):
        raise FileNotFoundError(f"Contextual reviews directory '{context_dir}' not found.")

    for file_name in os.listdir(context_dir):
        cuisine = file_name.replace("_reviews.txt", "").capitalize()
        file_path = os.path.join(context_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            context_data[cuisine] = [line.strip() for line in f.readlines()]
    return context_data


def train_word2vec(reviews):
    """Train a Word2Vec model."""
    tokenized_reviews = [review.split() for review in reviews]
    model = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=5, min_count=1, workers=4)
    print("Word2Vec training completed.")
    return model


def expand_dish_names(refined_labels, word2vec_model):
    """Expand dish names using Word2Vec."""
    expanded_dishes = {}
    for _, row in refined_labels.iterrows():
        phrase = row["phrase"]
        if phrase not in word2vec_model.wv:
            print(f"'{phrase}' not found in Word2Vec vocabulary. Skipping...")
            expanded_dishes[phrase] = []
            continue
        similar_words = word2vec_model.wv.most_similar(phrase, topn=5)
        expanded_dishes[phrase] = similar_words
    return expanded_dishes


def main(refined_file_paths, context_dir, output_dir):
    """Main function to process all cuisines."""
    context_data = load_contextual_data(context_dir)

    for cuisine, label_path in refined_file_paths.items():
        print(f"\nProcessing {cuisine}...")

        if cuisine not in context_data or not context_data[cuisine]:
            print(f"No reviews found for {cuisine}. Skipping...")
            continue

        reviews = context_data[cuisine]
        print(f"Number of reviews for {cuisine}: {len(reviews)}")

        # Train Word2Vec model
        print(f"Training Word2Vec model for {cuisine}...")
        word2vec_model = train_word2vec(reviews)

        # Load refined labels
        refined_labels = load_labels(label_path)
        if refined_labels is None or "phrase" not in refined_labels.columns:
            print(f"Refined labels for {cuisine} are invalid or missing the 'phrase' column. Skipping...")
            continue

        # Expand dish names
        print(f"Expanding dish names for {cuisine}...")
        expanded_dishes = expand_dish_names(refined_labels, word2vec_model)

        # Save results
        output_path = os.path.join(output_dir, f"expanded_{cuisine}_dishes.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for dish, similar_words in expanded_dishes.items():
                f.write(f"{dish}: {similar_words}\n")
        print(f"Expanded dish names saved to {output_path}.")


if __name__ == "__main__":
    refined_file_paths = {
        "American": "C:/Users/Admin/PycharmProjects/task1/manualAnnotationTask/refined_American_(New).label",
        "Chinese": "C:/Users/Admin/PycharmProjects/task1/manualAnnotationTask/refined_Chinese.label",
        "Indian": "C:/Users/Admin/PycharmProjects/task1/manualAnnotationTask/refined_Indian.label",
        "Italian": "C:/Users/Admin/PycharmProjects/task1/manualAnnotationTask/refined_Italian.label",
        "Mediterranean": "C:/Users/Admin/PycharmProjects/task1/manualAnnotationTask/refined_Mediterranean.label",
        "Mexican": "C:/Users/Admin/PycharmProjects/task1/manualAnnotationTask/refined_Mexican.label",
    }
    context_dir = "contextual_reviews"
    output_dir = "expanded_dishes"

    main(refined_file_paths, context_dir, output_dir)
