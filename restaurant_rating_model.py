import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Connect to MongoDB
client = MongoClient("mongodb+srv://mongodbconnect43:mongoDB%40123@mongodemo.wndvpbd.mongodb.net/")
db = client["FoodMate"]
posts = db["posts"]

def extract_features():
    restaurant_stats = {}
    # Get all restaurant names from Users collection (role: owner)
    owner_restaurants = set()
    try:
        users_coll = db["Users"]
        for user in users_coll.find({"role": "owner"}):
            loc = user.get("location", {})
            name = loc.get("name")
            if name:
                owner_restaurants.add(name)
    except Exception as e:
        print("Error fetching owners:", e)

    # Collect stats from posts
    for post in posts.find():
        restaurant = post.get('restaurant', {}).get('name')
        if not restaurant:
            continue
        if restaurant not in restaurant_stats:
            restaurant_stats[restaurant] = {
                "total_posts": 0,
                "total_likes": 0,
                "total_comments": 0,
                "review_sentiments": {"Positive": 0, "Neutral": 0, "Negative": 0},
                "comment_sentiments": {"Positive": 0, "Neutral": 0, "Negative": 0},
            }
        stats = restaurant_stats[restaurant]
        stats["total_posts"] += 1
        stats["total_likes"] += len(post.get("likes", []))
        stats["review_sentiments"][post.get("review_sentiment", "Neutral")] += 1
        comments = post.get("comments", [])
        stats["total_comments"] += len(comments)
        for comment in comments:
            stats["comment_sentiments"][comment.get("sentiment", "Neutral")] += 1

    # Add restaurants with no posts
    for restaurant in owner_restaurants:
        if restaurant not in restaurant_stats:
            restaurant_stats[restaurant] = {
                "total_posts": 0,
                "total_likes": 0,
                "total_comments": 0,
                "review_sentiments": {"Positive": 0, "Neutral": 0, "Negative": 0},
                "comment_sentiments": {"Positive": 0, "Neutral": 0, "Negative": 0},
            }

    # Convert to DataFrame
    data = []
    for restaurant, stats in restaurant_stats.items():
        total_reviews = sum(stats["review_sentiments"].values())
        total_comments = stats["total_comments"]
        total_posts = stats["total_posts"]

        avg_likes = stats["total_likes"] / total_posts if total_posts else 0
        review_pos = stats["review_sentiments"]["Positive"] / total_reviews if total_reviews else 0
        review_neu = stats["review_sentiments"]["Neutral"] / total_reviews if total_reviews else 0
        review_neg = stats["review_sentiments"]["Negative"] / total_reviews if total_reviews else 0

        comment_pos = stats["comment_sentiments"]["Positive"] / total_comments if total_comments else 0
        comment_neu = stats["comment_sentiments"]["Neutral"] / total_comments if total_comments else 0
        comment_neg = stats["comment_sentiments"]["Negative"] / total_comments if total_comments else 0

        # Heuristic score (0–5) — tweak weights as needed
        if total_posts == 0:
            score = 0
        else:
            score = 1 + (2.5 * review_pos) + (1.5 * comment_pos) + (0.5 * avg_likes)

        data.append({
            "restaurant": restaurant,
            "total_posts": total_posts,
            "avg_likes_per_post": avg_likes,
            "review_pos": review_pos,
            "review_neu": review_neu,
            "review_neg": review_neg,
            "comment_pos": comment_pos,
            "comment_neu": comment_neu,
            "comment_neg": comment_neg,
            "quality_score": round(min(score, 5), 2)
        })

    return pd.DataFrame(data)

# Extract dataset
df = extract_features()
df.to_csv("restaurant_features.csv", index=False)

# Model training
X = df.drop(columns=["restaurant", "quality_score"])
y = df["quality_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, preds))

# Save model
joblib.dump(model, "restaurant_quality_model.pkl")
