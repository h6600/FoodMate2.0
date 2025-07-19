from pymongo import MongoClient

client = MongoClient("mongodb+srv://mongodbconnect43:mongoDB%40123@mongodemo.wndvpbd.mongodb.net/")
db = client["FoodMate"]
<<<<<<< HEAD
=======

>>>>>>> c75102664fb183f72dee90ef03e5babf26daf090
users = db["Users"]
posts = db["posts"]
