import math
from flask import flash
import os
from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from database.mongo_handler import users, posts
from bson.objectid import ObjectId
from datetime import datetime
from werkzeug.utils import secure_filename
from base64 import b64encode
import base64
import datetime
from utils.classifier import predict_food_tag
from collections import Counter
from sentiment_model import predict_sentiment
import bcrypt
import pickle
import pandas as pd

MODEL_PATH = "restaurant_model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Set upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = users.find_one({"username": request.form['username']})
        if user and user['password'] == request.form['password']:
            session['user'] = user['username']
            session['role'] = user.get('role', 'user')
            if user.get('role') == 'owner':
                return redirect('/restaurant_dashboard')
            else:
                return redirect('/dashboard')
        return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        password = request.form['password']
        confirm = request.form['confirm']
        bio = request.form['bio']
        role = request.form['role']

        # Validate password match
        if password != confirm:
            flash('Passwords do not match.')
            return redirect('/register')

        # Check if user already exists
        if users.find_one({'username': username}):
            flash('Username already exists.')
            return redirect('/register')

        # Handle location based on role
        location_info = {}
        if role == 'owner':
            location_info = {
                "name": request.form.get("restaurant_name"),
                "lat": float(request.form.get("restaurant_lat")),
                "lng": float(request.form.get("restaurant_lng"))
            }
        else:
            location_info = {
                'state': request.form.get('state')
            }


        # Handle optional profile image upload
        profile_pic = None
        if 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file and file.filename != '':
                img_data = file.read()
                profile_pic = base64.b64encode(img_data).decode('utf-8')

        # Build user object
        user_data = {
            'username': username,
            'firstname': firstname,
            'lastname': lastname,
            'email': email,
            'password': password,
            'bio': bio,
            'role': role,
            'location': location_info,
            'created_on': datetime.datetime.now(),
            'profile_pic': profile_pic
        }

        users.insert_one(user_data)
        flash('Registered successfully. Please log in.')
        return redirect('/login')

    return render_template('register.html')


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/login')
    user = users.find_one({"username": session['user']})
    sample_posts = list(posts.find().sort("_id", -1).limit(4))
    return render_template("dashboard.html", user=session['user'], sample_posts=sample_posts)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction = None
    if request.method == 'POST':
        # Basic info
        title = request.form.get('title')
        review = request.form.get('review')
        file = request.files['image']
        user = session['user']


        # Read and encode image
        if file:
            image_data = base64.b64encode(file.read()).decode('utf-8')
        else:
            image_data = ""
        prediction = predict_food_tag(image_data)
        # Location
        restaurant = {
           "name": request.form.get("restaurant_name"),
           "lat": float(request.form.get("restaurant_lat")),
           "lng": float(request.form.get("restaurant_lng"))
        }

        # Save to MongoDB
        post_data = {
            "review": review,
            "review_sentiment": predict_sentiment(review),
            "image_base64": image_data,
            "username": user,
            "timestamp": datetime.datetime.now(),
            "likes": [],
            "comments": [],
            "restaurant": restaurant,
            "predicted_tag": prediction
        }

        posts.insert_one(post_data)
        flash('Post uploaded successfully!')
        return redirect(url_for('feed'))

    return render_template('upload.html')


@app.route('/post/<post_id>')
def view_post(post_id):
    post = posts.find_one({'_id': ObjectId(post_id)})
    user = users.find_one({'username': post.get('username')})
    post['profile_pic'] = user.get('profile_pic') if user else None
    if post:
        return render_template('post.html', post=post)
    return "Post not found", 404

@app.route('/like_post/<post_id>', methods=['POST'])
def like_post(post_id):
    username = session['user']
    post = posts.find_one({'_id': ObjectId(post_id)})

    if username in post.get('likes', []):
        # Already liked, so remove
        posts.update_one({'_id': ObjectId(post_id)}, {'$pull': {'likes': username}})
    else:
        # Remove from dislikes if present
        posts.update_one({'_id': ObjectId(post_id)}, {'$pull': {'dislikes': username}})
        # Add to likes
        posts.update_one({'_id': ObjectId(post_id)}, {'$addToSet': {'likes': username}})

    return redirect(f"/post/{post_id}")

@app.route('/dislike_post/<post_id>', methods=['POST'])
def dislike_post(post_id):
    username = session['user']
    post = posts.find_one({'_id': ObjectId(post_id)})

    if username in post.get('dislikes', []):
        # Already disliked, so remove
        posts.update_one({'_id': ObjectId(post_id)}, {'$pull': {'dislikes': username}})
    else:
        # Remove from likes if present
        posts.update_one({'_id': ObjectId(post_id)}, {'$pull': {'likes': username}})
        # Add to dislikes
        posts.update_one({'_id': ObjectId(post_id)}, {'$addToSet': {'dislikes': username}})

    return redirect(f"/post/{post_id}")

@app.route('/comment_post/<post_id>', methods=['POST'])
def comment_post(post_id):
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    comment_text = request.form.get('comment', '').strip()
    if not comment_text:
        return jsonify({'error': 'Comment is empty'}), 400
    
    sentiment = predict_sentiment(comment_text)

    comment = {
        'username': session['user'],
        'text': comment_text,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'sentiment': sentiment
    }

    posts.update_one(
        {'_id': ObjectId(post_id)},
        {'$push': {'comments': comment}}
    )

    return redirect(url_for('view_post', post_id=post_id))

# --- Profile Page View and Edit ---

# @app.route('/profile')
# def profile():
#     if 'user' not in session:
#         return redirect('/login')
#     user_posts = list(posts.find({"username": session["user"]}))
#     user = users.find_one({'username': session['user']})
#     total_likes = sum(len(post.get('likes', [])) for post in user_posts)
#     total_comments = sum(len(post.get('comments', [])) for post in user_posts)
#     return render_template("profile.html", user_posts=user_posts, user=user,post_count=len(user_posts), total_likes=total_likes, total_comments=total_comments)

@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect('/login')

    user = users.find_one({'username': session['user']})
    role = user.get('role')

    if role == 'owner':
        # Fetch all posts tagged with this restaurant
        location = user.get('location').get('name')
        restaurant_posts = list(posts.find({'restaurant.name': location}).sort('_id', -1))
        return render_template('profile.html', user=user, posts=restaurant_posts, is_owner=True)
    else:
        # Normal user – show their own posts
        user_posts = list(posts.find({'username': session['user']}).sort('_id', -1))
        return render_template('profile.html', user=user, posts=user_posts, is_owner=False)

@app.route('/edit_profile', methods=['POST'])
def edit_profile():
    if 'user' not in session:
        return redirect(url_for('login'))

    updated_data = {
        'firstname': request.form.get('firstname'),
        'lastname': request.form.get('lastname'),
        'email': request.form.get('email'),
        'about': request.form.get('about')
    }

    # Handle profile image upload
    if 'profile_pic' in request.files:
        file = request.files['profile_pic']
        if file and file.filename != '':
            img_data = file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            updated_data['profile_pic'] = img_base64

    # Update in MongoDB
    users.update_one({'username': session['user']}, {'$set': updated_data})

    flash('Profile updated successfully!')
    return redirect(url_for('profile'))

@app.route('/trending_tags')
def trending_tags():
    post_list = list(posts.find({}, {'predicted_tag': 1}))
    tags = [post.get('predicted_tag') for post in post_list if post.get('predicted_tag')]
    tag_counts = Counter(tags).most_common(5)  # Top 5 trending tags
    return jsonify(tag_counts)

@app.route('/delete_post/<post_id>', methods=['POST'])
def delete_post(post_id):
    if 'user' not in session:
        return redirect('/login')

    post = posts.find_one({'_id': ObjectId(post_id)})
    if not post:
        return "Post not found", 404

    # Only allow if the logged-in user owns this post
    if post.get('username') != session['user']:
        return "Unauthorized", 403

    posts.delete_one({'_id': ObjectId(post_id)})
    return redirect('/profile')

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# @app.route('/feed')
# def feed():
#     view = request.args.get('view', 'scroll')  # default to 'scroll'
#     page = int(request.args.get('page', 1))

#     lat = request.args.get('lat', type=float)
#     lng = request.args.get('lng', type=float)

#     user_map = {
#         user['username']: user.get('profile_pic')
#         for user in users.find({}, {'username': 1, 'profile_pic': 1})
#     }

#     # Pagination settings
#     posts_per_page = 8 if view == 'grid' else 6
#     skip = (page - 1) * posts_per_page

#     # Base query
#     all_posts = list(posts.find().sort('_id', -1))

#     # Filter by location if lat/lng provided
#     if lat and lng:
#         filtered_posts = []
#         for post in all_posts:
#             loc = post.get('restaurant', {})
#             if loc.get('lat') and loc.get('lng'):
#                 distance = haversine(lat, lng, loc['lat'], loc['lng'])
#                 if distance <= 10:
#                     post['distance'] = round(distance, 1)
#                     filtered_posts.append(post)
#         all_posts = filtered_posts

#     total_posts = len(all_posts)
#     total_pages = math.ceil(total_posts / posts_per_page)
#     paginated_posts = all_posts[skip:skip + posts_per_page]

#     for post in paginated_posts:
#         post['profile_pic'] = user_map.get(post.get('username'))

#     return render_template(
#         'feed.html',
#         all_posts=paginated_posts,
#         current_page=page,
#         total_pages=total_pages,
#         current_view=view
#     )

@app.route('/promote/<post_id>', methods=['POST'])
def promote_post(post_id):
    if 'user' not in session or session.get('role') != 'owner':
        return redirect('/login')

    user = users.find_one({'username': session['user']})
    post = posts.find_one({'_id': ObjectId(post_id)})

    if post and post.get('restaurant.name') == user.get('location.name'):
        posts.update_one({'_id': ObjectId(post_id)}, {'$set': {'promoted': True}})
    
    return redirect('/profile')

@app.route('/restaurant_dashboard')
def restaurant_dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    current_user = users.find_one({'username': session['user']})
    restaurant_name = current_user.get('location').get('name')
    restaurant_posts = list(posts.find({'restaurant.name': restaurant_name}))

    total_posts = len(restaurant_posts)
    total_likes = sum(len(p.get('likes', [])) for p in restaurant_posts)
    total_dislikes = sum(len(p.get('dislikes', [])) for p in restaurant_posts)
    total_comments = sum(len(p.get('comments', [])) for p in restaurant_posts)

    # Sentiment count from reviews and comments (case-insensitive, robust)
    sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    for post in restaurant_posts:
        # Review sentiment
        review_sentiment = post.get('review_sentiment')
        review_sentiment = review_sentiment.lower()
        if review_sentiment in ['positive', 'neutral', 'negative']:
            sentiment_counts[review_sentiment.capitalize()] += 1
        # Comment sentiments
        for comment in post.get('comments', []):
            comment_sentiment = comment.get('sentiment')
            comment_sentiment = comment_sentiment.lower()
            if comment_sentiment in ['positive', 'neutral', 'negative']:
                sentiment_counts[comment_sentiment.capitalize()] += 1

    # Ensure all sentiment counts are integers
    review_pos, review_neu, review_neg = compute_sentiment_distribution_from_posts(restaurant_posts)
    # Calculate restaurant quality score
    restaurant_score = predict_quality_score(restaurant_name);  

    ## third chart logic
    user_lat = current_user['location']['lat']
    user_lon = current_user['location']['lng']

    # Get nearby restaurants within 10km
    all_nearby = get_nearby_restaurants(user_lat, user_lon)  # each item: {'name': ..., 'lat': ..., 'lon': ...}
   
    # Calculate predicted quality score for each
    scored_restaurants = []
    for r in all_nearby:
        score = predict_quality_score(r['name'])
        scored_restaurants.append({
            'name': r['name'],
            'score': score
        })

    print(scored_restaurants)
    # Sort and pick top 5
    top5 = sorted(scored_restaurants, key=lambda x: x['score'], reverse=True)[:5]

    labels = [r['name'] for r in top5]
    scores = [r['score'] for r in top5]
    stars = ['★' * round(s) + '☆' * (5 - round(s)) for s in scores]
    print(labels)
    print(scores)
    print(stars)
    
    return render_template(
        'restaurant_dashboard.html',
        user=current_user,
        restaurant_name=restaurant_name,
        total_posts=total_posts,
        total_likes=total_likes,
        total_dislikes=total_dislikes,
        total_comments=total_comments,
        restaurant_score=restaurant_score,
        review_pos=review_pos,
        review_neu=review_neu,
        review_neg=review_neg,
        top_labels=labels,
        top_scores=scores,
        top_stars=stars
    )

## to map restaurant score
def predict_quality_score(restaurant_name):
    import joblib
    import pandas as pd

    try:
        model = joblib.load("restaurant_quality_model.pkl")
        df = pd.read_csv("restaurant_features.csv")
        restaurant_data = df[df["restaurant"] == restaurant_name]

        if restaurant_data.empty:
            print("No data for:", restaurant_name)
            return None

        features = restaurant_data.drop(columns=["restaurant", "quality_score"])
        return round(restaurant_data["quality_score"].values[0], 2)

    except Exception as e:
        print("Prediction error:", e)
        return None

def compute_sentiment_distribution_from_posts(posts):
    total_positive = 0
    total_neutral = 0
    total_negative = 0
    total_sentiments = 0

    for post in posts:
        # Review sentiment
        review_sentiment = post.get('review_sentiment', '').strip().lower()
        if review_sentiment in ['positive', 'neutral', 'negative']:
            total_sentiments += 1
            if review_sentiment == 'positive':
                total_positive += 1
            elif review_sentiment == 'neutral':
                total_neutral += 1
            elif review_sentiment == 'negative':
                total_negative += 1

        # Comments sentiments
        for comment in post.get('comments', []):
            comment_sentiment = comment.get('sentiment', '').strip().lower()
            if comment_sentiment in ['positive', 'neutral', 'negative']:
                total_sentiments += 1
                if comment_sentiment == 'positive':
                    total_positive += 1
                elif comment_sentiment == 'neutral':
                    total_neutral += 1
                elif comment_sentiment == 'negative':
                    total_negative += 1

    # Avoid division by zero
    if total_sentiments == 0:
        return 0, 0, 0

    review_pos = total_positive / total_sentiments
    review_neu = total_neutral / total_sentiments
    review_neg = total_negative / total_sentiments

    return review_pos, review_neu, review_neg

## to get nearby restaurants based on user's location
def get_nearby_restaurants(user_lat, user_lon, radius_km=10):
    all_restaurants = list(users.find({"role": "owner"}))  # adjust collection
    nearby = []
    for rest in all_restaurants:
        loc = rest.get('location', {})
        if 'lat' in loc and 'lng' in loc:
            dist = haversine(user_lat, user_lon, loc['lat'], loc['lng'])
            if dist <= radius_km:
                nearby.append({
                    'name': loc['name'],
                })
    return nearby

@app.route('/feed')
def feed():
    view = request.args.get('view', 'scroll')  # default to 'scroll'
    page = int(request.args.get('page', 1))

    lat = request.args.get('lat', type=float)
    lng = request.args.get('lng', type=float)

    user_map = {
        user['username']: user.get('profile_pic')
        for user in users.find({}, {'username': 1, 'profile_pic': 1})
    }

    # Pagination settings
    posts_per_page = 8 if view == 'grid' else 6
    skip = (page - 1) * posts_per_page

    # Get all posts from DB
    all_posts = list(posts.find())

    # Filter by location if lat/lng provided
    if lat and lng:
        filtered_posts = []
        for post in all_posts:
            loc = post.get('restaurant', {})
            if loc.get('lat') and loc.get('lng'):
                distance = haversine(lat, lng, loc['lat'], loc['lng'])
                if distance <= 10:
                    post['distance'] = round(distance, 1)
                    filtered_posts.append(post)
        all_posts = filtered_posts

    # Compute popularity score for each post
    for post in all_posts:
        # Ensure likes is always an integer (handle list or int)
        likes = post.get('likes', 0)
        if isinstance(likes, list):
            likes = len(likes)
        elif not isinstance(likes, int):
            try:
                likes = int(likes)
            except Exception:
                likes = 0

        # Ensure comments is always a list
        comments = post.get('comments', [])
        if not isinstance(comments, list):
            comments = []
        num_comments = len(comments)

        sentiment = post.get('sentiment_score', 0)  # default neutral if not present

        # Weighted popularity score
        post['popularity_score'] = (likes * 2) + num_comments + (sentiment * 10)

    # Sort posts by popularity score (descending)
    all_posts.sort(key=lambda x: x.get('popularity_score', 0), reverse=True)

    # Pagination
    total_posts = len(all_posts)
    total_pages = math.ceil(total_posts / posts_per_page)
    paginated_posts = all_posts[skip:skip + posts_per_page]

    # Attach profile picture
    for post in paginated_posts:
        post['profile_pic'] = user_map.get(post.get('username'))

    return render_template(
        'feed.html',
        all_posts=paginated_posts,
        current_page=page,
        total_pages=total_pages,
        current_view=view
    )


if __name__ == '__main__':
    app.run(debug=True)
