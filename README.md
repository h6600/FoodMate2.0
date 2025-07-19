# Foodmate 🍔 – An AI-Enhanced Food Social Platform

Foodmate is an interactive, AI-powered social platform for food lovers to share, discover, and review food experiences through real user images and recommendations.

## Features

✅ User Registration & Login (Session-based)  
✅ Upload food posts with image & review  
✅ Base64 image storage in MongoDB  
✅ Food category auto-tagging using PyTorch CNN  
✅ Like & comment system  
✅ Personalized feed with pagination  
✅ Instagram-style Post View  
✅ Profile page showing user’s posts  
✅ Feed grid with toggle options (scroll/grid/list)  
✅ Dashboard with stats & sample posts  
✅ Modern, responsive UI (HTML/CSS/JS)

## Technologies

- Python 3.x
- Flask (Backend)
- MongoDB (Data storage)
- PyTorch (Classifier)
- HTML/CSS/JS (Frontend)
- Jinja2 (Templating)

## Project Structure

```
/foodmate/
│
├── app.py                # Flask main app
├── classifier.py         # CNN model & prediction
├── train_model.py        # Training script (PyTorch)
├── requirements.txt
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── feed.html
│   ├── profile.html
│   ├── post.html
│   └── upload.html
├── static/
│   ├── style.css
│   └── js/
│       └── scripts.js
├── train/.gitkeep
├── val/.gitkeep
├── .gitignore
└── README.md
```

## Setup & Run

1. Clone the repo

```bash
git clone https://github.com/your-username/foodmate.git
cd foodmate
```

2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run MongoDB locally or use MongoDB Atlas

5. Run the app

```bash
python app.py
```

## Optional Enhancements

- [ ] JWT & OAuth login
- [ ] REST API for mobile
- [ ] Admin dashboard
- [ ] Docker containerization
- [ ] Recommender improvements