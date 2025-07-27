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
            return redirect('/dashboard')
        return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = {
            'username': request.form['username'],
            'firstname': request.form['firstname'],
            'lastname': request.form['lastname'],
            'email': request.form['email'],
            'password': request.form['password'],
            'about': request.form['about'],
            'user_created_on': datetime.datetime.now()
        }
        # Save to MongoDB
        if users.find_one({'username': data['username']}):
            return "Username already exists", 400
        users.insert_one(data)
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


# @app.route('/feed')
# def feed():
#     view = request.args.get('view', 'scroll')  # default to 'scroll'
#     page = int(request.args.get('page', 1))

#     user_map = {
#         user['username']: user.get('profile_pic')
#         for user in users.find({}, {'username': 1, 'profile_pic': 1})
#     }
#     posts_per_page = 8 if view == 'grid' else 6
#     total_posts = posts.count_documents({})
#     total_pages = math.ceil(total_posts / posts_per_page)
#     skip = (page - 1) * posts_per_page
#     paginated_posts = list(posts.find().sort('_id', -1).skip(skip).limit(posts_per_page))
#     for post in paginated_posts:
#         post['profile_pic'] = user_map.get(post.get('username'))
#     return render_template(
#         'feed.html',
#         all_posts=paginated_posts,
#         current_page=page,
#         total_pages=total_pages,
#         current_view=view
#     )



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
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    post = posts.find_one({'_id': ObjectId(post_id)})
    if not post:
        return jsonify({'error': 'Post not found'}), 404

    username = session['user']
    likes = post.get('likes', [])
    
    if username in likes:
        # Unlike
        posts.update_one({'_id': ObjectId(post_id)}, {'$pull': {'likes': username}})
        action = 'unliked'
    else:
        # Like
        posts.update_one({'_id': ObjectId(post_id)}, {'$addToSet': {'likes': username}})
        action = 'liked'

    return jsonify({'status': action})

@app.route('/comment_post/<post_id>', methods=['POST'])
def comment_post(post_id):
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    comment_text = request.form.get('comment', '').strip()
    if not comment_text:
        return jsonify({'error': 'Comment is empty'}), 400

    comment = {
        'username': session['user'],
        'text': comment_text,
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    }

    posts.update_one(
        {'_id': ObjectId(post_id)},
        {'$push': {'comments': comment}}
    )

    return redirect(url_for('view_post', post_id=post_id))

# --- Profile Page View and Edit ---

@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect('/login')
    user_posts = list(posts.find({"username": session["user"]}))
    user = users.find_one({'username': session['user']})
    total_likes = sum(len(post.get('likes', [])) for post in user_posts)
    total_comments = sum(len(post.get('comments', [])) for post in user_posts)
    return render_template("profile.html", user_posts=user_posts, user=user,post_count=len(user_posts), total_likes=total_likes, total_comments=total_comments)

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

    # Base query
    all_posts = list(posts.find().sort('_id', -1))

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

    total_posts = len(all_posts)
    total_pages = math.ceil(total_posts / posts_per_page)
    paginated_posts = all_posts[skip:skip + posts_per_page]

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
