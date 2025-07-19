import os
from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from database.mongo_handler import users, posts
from bson.objectid import ObjectId
from datetime import datetime 
from werkzeug.utils import secure_filename
from base64 import b64encode
from utils.classifier import predict_food_tag
from datetime import datetime

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
            'user_created_on': datetime.utcnow()
        }
        confirm = request.form['confirm_password']
        if data['password'] != confirm:
            return "Passwords do not match", 400
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
        return redirect('/login')

    prediction = None
    if request.method == 'POST':
        image_file = request.files['image']
        image_base64 = b64encode(image_file.read()).decode('utf-8')

        review = request.form['review']
        prediction = predict_food_tag(image_base64)

        posts.insert_one({
            "username": session['user'],
            "image_base64": image_base64,
            "review": review,
            "predicted_tag": prediction
        })

    return render_template("upload.html", prediction=prediction)

@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect('/login')
    user_posts = list(posts.find({"username": session["user"]}))
    return render_template("profile.html", user_posts=user_posts)

@app.route('/feed')
def feed():
    if 'user' not in session:
        return redirect('/login')

    # Get current page from URL, default to 1
    page = int(request.args.get('page', 1))
    per_page = 6
    skip = (page - 1) * per_page

    total_posts = posts.count_documents({})
    total_pages = (total_posts + per_page - 1) // per_page

    # Fetch limited posts with skip
    feed_posts = posts.find().sort([('_id', -1)]).skip(skip).limit(per_page)

    return render_template(
        'feed.html',
        all_posts=feed_posts,
        current_page=page,
        total_pages=total_pages
    )


@app.route('/upload_profile_pic', methods=['GET', 'POST'])
def upload_profile_pic():
    if 'user' not in session:
        return redirect('/login')

    if request.method == 'POST':
        image_file = request.files['profile_pic']
        image_base64 = b64encode(image_file.read()).decode('utf-8')
        users.update_one(
            {"username": session["user"]},
            {"$set": {"profile_pic": image_base64}}
        )
        return redirect('/profile')

    return render_template("upload_profile_pic.html")

@app.route('/post/<post_id>')
def view_post(post_id):
    post = posts.find_one({'_id': ObjectId(post_id)})
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
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
    }

    posts.update_one(
        {'_id': ObjectId(post_id)},
        {'$push': {'comments': comment}}
    )

    return redirect(url_for('view_post', post_id=post_id))



if __name__ == '__main__':
    app.run(debug=True)
