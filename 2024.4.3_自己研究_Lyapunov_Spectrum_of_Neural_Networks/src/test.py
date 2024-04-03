import pymongo

client = pymongo.MongoClient("mongodb://acan:020301@localhost:27017/")
db = client["local"]
collection = db["test"]

p = [i for i in range(1, 10001)]

post = {"index": 1, "content": p}
collection.insert_one(post)

# 打印集合中所有文档
for post in collection.find():
    print(post)
