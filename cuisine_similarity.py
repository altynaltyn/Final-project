import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

def load_and_preprocess_cuisine_reviews(folder):
    cuisine_texts = {}
    for file in os.listdir(folder):
        if file.endswith('.txt'):
            cuisine_name = file.replace('.txt', '').replace('_', ' ')
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                text = f.read()
                cuisine_texts[cuisine_name] = preprocess_text(text)
    return cuisine_texts

cuisine_folder = "categories/"  # Replace with the path to your cuisine text files
cuisine_reviews = load_and_preprocess_cuisine_reviews(cuisine_folder)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cuisine_reviews.values())
cuisines = list(cuisine_reviews.keys())

similarity_matrix = cosine_similarity(tfidf_matrix)


def plot_similarity_heatmap(matrix, cuisines, title="Cuisine Similarity Heatmap"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=cuisines, yticklabels=cuisines, cmap="viridis", annot=False)
    plt.title(title)
    plt.xlabel("Cuisines")
    plt.ylabel("Cuisines")
    plt.tight_layout()
    plt.show()

plot_similarity_heatmap(similarity_matrix, cuisines)

vectorizer_noidf = TfidfVectorizer(use_idf=False)
tfidf_matrix_noidf = vectorizer_noidf.fit_transform(cuisine_reviews.values())
similarity_matrix_noidf = cosine_similarity(tfidf_matrix_noidf)

plot_similarity_heatmap(similarity_matrix_noidf, cuisines, title="Cuisine Similarity Heatmap (No IDF)")

from gensim import corpora, models


def lda_representation(cuisine_reviews, num_topics=10):
    tokenized_texts = [text.split() for text in cuisine_reviews.values()]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topic_vectors = [lda_model.get_document_topics(c, minimum_probability=0) for c in corpus]

    # Convert topic vectors into dense arrays
    dense_topic_matrix = np.array([[prob for _, prob in topics] for topics in topic_vectors])
    return dense_topic_matrix, cuisines


lda_matrix, cuisines = lda_representation(cuisine_reviews)
similarity_matrix_lda = cosine_similarity(lda_matrix)

plot_similarity_heatmap(similarity_matrix_lda, cuisines, title="Cuisine Similarity Heatmap (LDA)")


from scipy.cluster.hierarchy import linkage, dendrogram

def plot_dendrogram(matrix, cuisines, method="ward"):
    linked = linkage(matrix, method=method)
    plt.figure(figsize=(12, 8))
    dendrogram(linked, labels=cuisines, orientation='top', distance_sort='descending')
    plt.title("Dendrogram of Cuisine Similarities")
    plt.xlabel("Cuisine")
    plt.ylabel("Distance")
    plt.show()

# Use similarity_matrix_noidf or similarity_matrix_lda for clustering
distance_matrix = 1 - similarity_matrix_noidf  # Convert similarity to distance
plot_dendrogram(distance_matrix, cuisines)
