import json
import os

# Paths to Yelp dataset files
path_to_business = "C:/Users/Admin/PycharmProjects/task1/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json"
path_to_reviews = "C:/Users/Admin/PycharmProjects/task1/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json"

context_dir = "contextual_reviews"


# Map business IDs to cuisines
# Map business IDs to cuisines
def get_business_cuisines():
    cuisine_map = {}
    with open(path_to_business, 'r', encoding='utf-8') as f:
        for line in f:
            business = json.loads(line)
            categories = business.get("categories", [])
            if isinstance(categories, list):  # Handle categories as a list
                for category in categories:
                    if category in ["American", "Chinese", "Indian", "Italian", "Mediterranean", "Mexican"]:
                        cuisine_map[business["business_id"]] = category
    return cuisine_map


# Extract reviews for each cuisine
def extract_reviews(cuisine_map):
    os.makedirs(output_dir, exist_ok=True)
    reviews_by_cuisine = {cuisine: [] for cuisine in cuisine_map.values()}
    with open(path_to_reviews, 'r', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line)
            business_id = review["business_id"]
            if business_id in cuisine_map:
                reviews_by_cuisine[cuisine_map[business_id]].append(review["text"])

    # Save reviews to files
    for cuisine, reviews in reviews_by_cuisine.items():
        with open(os.path.join(output_dir, f"{cuisine.lower()}_reviews.txt"), 'w', encoding='utf-8') as f:
            f.write("\n".join(reviews))

if __name__ == "__main__":
    business_to_cuisine = get_business_cuisines()
    extract_reviews(business_to_cuisine)
    print(f"Reviews saved in {output_dir}/")
