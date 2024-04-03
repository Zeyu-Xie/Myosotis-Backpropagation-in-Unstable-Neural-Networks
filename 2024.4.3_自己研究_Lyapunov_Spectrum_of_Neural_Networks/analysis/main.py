import pymongo

client = pymongo.MongoClient("mongodb://acan_read:020301@localhost:27017/")
db = client["local"]
collection = db["data_20240403"]

data = collection.find()

print(data, type(data))