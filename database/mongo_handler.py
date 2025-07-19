from pymongo import MongoClient

client = MongoClient("mongodb+srv://mongodbconnect43:mongoDB%40123@mongodemo.wndvpbd.mongodb.net/")
db = client["FoodMateDemo"]

users = db["Users"]
posts = db["posts"]
feedback = db["feedback"]
