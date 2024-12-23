import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt

# Function to load dish names from student_dn_annotations
def load_dish_names_from_annotations(file_path):
    """Load dish names from a file where each line contains a dish name."""
    dish_names = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            dish_name = line.strip()  # Remove any surrounding whitespace or newlines
            if dish_name:  # Ensure the line is not empty
                dish_names.append(dish_name)
    print(f"Loaded {len(dish_names)} dish names: {dish_names[:10]}...")  # Print the first 10 for debugging
    return dish_names

# Main function to process and visualize dish rankings
def main(annotation_file, review_file, output_dir):
    # Load dish names
    dish_names = load_dish_names_from_annotations(annotation_file)
    print(f"Loaded {len(dish_names)} dish names.")

    # Load reviews
    with open(review_file, 'r', encoding='utf-8') as f:
        reviews = f.readlines()

    # Count dish mentions
    dish_counts = Counter()
    for review in reviews:
        for dish in dish_names:
            if dish in review.lower():
                dish_counts[dish] += 1

    # Save dish rankings
    dish_rankings = pd.DataFrame(dish_counts.items(), columns=["Dish", "Count"]).sort_values(by="Count", ascending=False)
    output_csv = os.path.join(output_dir, "dish_rankings.csv")
    dish_rankings.to_csv(output_csv, index=False)
    print(f"Dish rankings saved to {output_csv}")

    # Visualize rankings
    top_dishes = dish_rankings.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top_dishes["Dish"], top_dishes["Count"], color='skyblue')
    plt.xlabel("Mentions")
    plt.ylabel("Dish")
    plt.title("Top 10 Popular Dishes")
    plt.gca().invert_yaxis()  # Highest rank at the top
    output_png = os.path.join(output_dir, "popular_dishes.png")
    plt.savefig(output_png)
    print(f"Dish ranking visualization saved to {output_png}")

# Update paths and execute
if __name__ == "__main__":
    annotation_file = "C:/Users/Admin/PycharmProjects/task1/student_dn_annotations.txt"  # Path to student_dn_annotations
    review_file = "C:/Users/Admin/PycharmProjects/task1/contextual_reviews/chinese_reviews.txt"  # Replace with the desired cuisine review file
    output_dir = "C:/Users/Admin/PycharmProjects/task1/output"  # Directory for saving outputs
    os.makedirs(output_dir, exist_ok=True)
    main(annotation_file, review_file, output_dir)
