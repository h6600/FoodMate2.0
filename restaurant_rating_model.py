# from pymongo import MongoClient
# from textblob import TextBlob
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, r2_score
# import joblib

# # 1. Connect to MongoDB
# client = MongoClient("mongodb+srv://mongodbconnect43:mongoDB%40123@mongodemo.wndvpbd.mongodb.net/")
# db = client["FoodMate"]
# users = db["Users"]
# posts = db["posts"]

# # 2. Aggregate restaurant features
# restaurant_data = {}

# for post in posts.find():
#     restaurant = post.get('restaurant')
#     if not restaurant or 'name' not in restaurant:
#         continue
#     rname = restaurant['name']
#     review = post.get('review', '')

#     if rname not in restaurant_data:
#         restaurant_data[rname] = {
#             'total_likes': 0,
#             'total_comments': 0,
#             'comment_sentiments': [],
#             'review_sentiments': [],
#             'num_posts': 0
#         }

#     restaurant_data[rname]['num_posts'] += 1
#     restaurant_data[rname]['total_likes'] += len(post.get('likes', []))
#     restaurant_data[rname]['total_comments'] += len(post.get('comments', []))

#     # Review sentiment using TextBlob
#     if review:
#         review_polarity = TextBlob(review).sentiment.polarity
#         scaled_review_score = (review_polarity + 1) * 2.5  # Convert from [-1,1] → [0,5]
#         restaurant_data[rname]['review_sentiments'].append(scaled_review_score)

#     # Comment sentiments (already labeled)
#     for comment in post.get('comments', []):
#         sent = comment.get('sentiment', 'Neutral')
#         score = 3.0  # default for Neutral
#         if sent == 'Positive':
#             score = 5.0
#         elif sent == 'Negative':
#             score = 1.0
#         restaurant_data[rname]['comment_sentiments'].append(score)

# # 3. Convert to DataFrame
# rows = []

# for rname, data in restaurant_data.items():
#     avg_review_sent = sum(data['review_sentiments']) / len(data['review_sentiments']) if data['review_sentiments'] else 3.0
#     avg_comment_sent = sum(data['comment_sentiments']) / len(data['comment_sentiments']) if data['comment_sentiments'] else 3.0
#     combined_sent = (avg_review_sent + avg_comment_sent) / 2

#     # Simulated restaurant quality score (for training)
#     score = (
#         0.003 * data['total_likes'] +
#         0.01 * data['total_comments'] +
#         0.5 * combined_sent +
#         0.01 * data['num_posts']
#     )
#     score = round(min(5.0, max(1.0, score)), 2)

#     rows.append({
#         'restaurant_name': rname,
#         'total_likes': data['total_likes'],
#         'total_comments': data['total_comments'],
#         'avg_review_sent': avg_review_sent,
#         'avg_comment_sent': avg_comment_sent,
#         'combined_sentiment': combined_sent,
#         'num_posts': data['num_posts'],
#         'quality_score': score
#     })

# df = pd.DataFrame(rows)

# # 4. Train model
# X = df[['total_likes', 'total_comments', 'avg_review_sent', 'avg_comment_sent', 'num_posts']]
# y = df['quality_score']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # 5. Evaluate
# y_pred = model.predict(X_test)
# print("MAE:", mean_absolute_error(y_test, y_pred))
# print("R²:", r2_score(y_test, y_pred))

# # 6. Save
# joblib.dump(model, 'restaurant_quality_model.pkl')
# print("✅ Saved model to restaurant_quality_model.pkl")


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
users = db["Users"]

def extract_features():
    restaurant_stats = {}

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

        res = restaurant_stats[restaurant]
        res["total_posts"] += 1
        res["total_likes"] += len(post.get("likes", []))
        res["review_sentiments"][post.get("review_sentiment", "Neutral")] += 1

        comments = post.get("comments", [])
        res["total_comments"] += len(comments)
        for comment in comments:
            res["comment_sentiments"][comment.get("sentiment", "Neutral")] += 1

    # Convert to DataFrame
    data = []
    for restaurant, stats in restaurant_stats.items():
        total_reviews = sum(stats["review_sentiments"].values())
        total_comments = stats["total_comments"]

        data.append({
            "restaurant": restaurant,
            "total_posts": stats["total_posts"],
            "avg_likes_per_post": stats["total_likes"] / stats["total_posts"],
            "review_pos": stats["review_sentiments"]["Positive"] / total_reviews if total_reviews else 0,
            "review_neu": stats["review_sentiments"]["Neutral"] / total_reviews if total_reviews else 0,
            "review_neg": stats["review_sentiments"]["Negative"] / total_reviews if total_reviews else 0,
            "comment_pos": stats["comment_sentiments"]["Positive"] / total_comments if total_comments else 0,
            "comment_neu": stats["comment_sentiments"]["Neutral"] / total_comments if total_comments else 0,
            "comment_neg": stats["comment_sentiments"]["Negative"] / total_comments if total_comments else 0,
            # Simulated score (for training); replace this with real ratings if available
            "quality_score": np.random.uniform(2.5, 4.9)
        })

    return pd.DataFrame(data)

# Get data
df = extract_features()
X = df.drop(columns=["restaurant", "quality_score"])
y = df["quality_score"]

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, preds))

# Save
joblib.dump(model, "restaurant_quality_model.pkl")
df.to_csv("restaurant_features.csv", index=False)
