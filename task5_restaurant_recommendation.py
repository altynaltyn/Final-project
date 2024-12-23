import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict
import nltk

# Download necessary NLTK resources
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# File paths
reviews_file = "C:/Users/Admin/PycharmProjects/task1/contextual_reviews/Indian_reviews.txt"
dish_file = "C:/Users/Admin/PycharmProjects/task1/output/dish_rankings.csv"
output_file = "C:/Users/Admin/PycharmProjects/task1/output/restaurant_rankings.csv"
visualization_file = "C:/Users/Admin/PycharmProjects/task1/output/restaurant_recommendation.png"


# Load dish names
def load_dishes(dish_file):
    dishes = pd.read_csv(dish_file)['Dish'].tolist()
    return dishes


# Filter reviews for dishes
def filter_reviews_by_dishes(reviews, dishes):
    filtered_reviews = []
    for review in reviews:
        for dish in dishes:
            if dish in review.lower():
                filtered_reviews.append((dish, review))
                break
    return filtered_reviews


# Perform sentiment analysis
def analyze_sentiment(reviews):
    sentiment_scores = []
    for dish, review in reviews:
        sentiment = sia.polarity_scores(review)['compound']
        sentiment_scores.append((dish, review, sentiment))
    return sentiment_scores


# Rank restaurants
def rank_restaurants(reviews_with_sentiment, reviews_data):
    restaurant_scores = defaultdict(list)
    for dish, review, sentiment in reviews_with_sentiment:
        restaurant = reviews_data[review]['restaurant']
        rating = reviews_data[review]['rating']
        restaurant_scores[restaurant].append((sentiment, rating))

    restaurant_rankings = []
    for restaurant, scores in restaurant_scores.items():
        avg_sentiment = sum(s[0] for s in scores) / len(scores)
        avg_rating = sum(s[1] for s in scores) / len(scores)
        final_score = (avg_sentiment + avg_rating) / 2
        restaurant_rankings.append((restaurant, final_score))

    return sorted(restaurant_rankings, key=lambda x: x[1], reverse=True)


# Visualize rankings
def visualize_rankings(rankings, output_file):
    restaurants = [r[0] for r in rankings[:10]]
    scores = [r[1] for r in rankings[:10]]

    plt.barh(restaurants, scores, color='skyblue')
    plt.xlabel("Ranking Score")
    plt.ylabel("Restaurant")
    plt.title("Top 10 Restaurant Recommendations")
    plt.gca().invert_yaxis()
    plt.savefig(output_file)
    plt.show()


# Main function
def main():
    # Load data
    with open(reviews_file, 'r', encoding='utf-8') as f:
        reviews = f.readlines()
    dishes = load_dishes(dish_file)

    # Filter reviews
    filtered_reviews = filter_reviews_by_dishes(reviews, dishes)
    reviews_with_sentiment = analyze_sentiment(filtered_reviews)

    # Simulate reviews_data (replace this with actual data)
    reviews_data = {review: {'restaurant': f"Restaurant {i % 10}", 'rating': 4.0 + (i % 5 - 2) * 0.5}
                    for i, review in enumerate(reviews)}

    # Rank restaurants
    restaurant_rankings = rank_restaurants(reviews_with_sentiment, reviews_data)

    # Save and visualize rankings
    rankings_df = pd.DataFrame(restaurant_rankings, columns=['Restaurant', 'Score'])
    rankings_df.to_csv(output_file, index=False)
    visualize_rankings(restaurant_rankings, visualization_file)


if __name__ == "__main__":
    main()
