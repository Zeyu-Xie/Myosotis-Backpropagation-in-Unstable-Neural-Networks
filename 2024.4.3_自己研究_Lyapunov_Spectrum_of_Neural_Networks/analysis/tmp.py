import pymongo

client = pymongo.MongoClient("mongodb://acan_read_and_write:020301@localhost:27017/")
db = client["local"]
collection = db.create_collection("N_Form")