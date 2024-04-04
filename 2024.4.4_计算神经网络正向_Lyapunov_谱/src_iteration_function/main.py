import pymongo
import numpy as np
import sys
from f import f

# Config

simulation_name = "Simulation_20240405_0101"

# Connect to MongoDB

client = pymongo.MongoClient('mongodb://acan:020301@localhost:27017/')
db = client["local"]
collection = db[simulation_name]

# Get Data

data_1 = collection.find_one({"index": 2})
data_2 = collection.find_one({"index": 3})

W1 = np.array(data_1["W1"])
W2 = np.array(data_1["W2"])
b1 = np.array(data_1["b1"])
b2 = np.array(data_1["b2"])

# Function 

ans = f({"W1": W1, "W2": W2, "b1": b1, "b2": b2})

print(ans["W1"]-data_2["W1"])