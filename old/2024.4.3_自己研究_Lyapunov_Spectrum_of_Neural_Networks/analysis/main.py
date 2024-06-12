import pymongo
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

client = pymongo.MongoClient(
    "mongodb://acan_read_and_write:020301@localhost:27017/")
db = client["local"]
collection = db["data_20240403"]

rnd = int(sys.argv[1])
print(f"Processing {rnd*100}th to {rnd*100+100}th data...")

clc = list(collection.find().skip(rnd*100).limit(101))

W1 = []
W2 = []
b1 = []
b2 = []

tmp_1 = []
tmp_2 = []
tmp_3 = []
tmp_4 = []

W1 = []
W2 = []
b1 = []
b2 = []

for i in range(101):

    print(f"Processing {i+rnd*100}th data...")

    data = clc[i]
    _W1 = data["W1"]
    _W2 = data["W2"]
    W1.append(np.matrix(_W1))
    W2.append(np.matrix(_W2))
    _b1 = data["b1"]
    _b2 = data["b2"]
    b1.append(np.array(_b1))
    b2.append(np.array(_b2))

for i in range(100):
    tmp_1.append(np.linalg.norm(W1[i]-W1[i+1], ord=2))
    tmp_2.append(np.linalg.norm(b1[i]-b1[i+1], ord=2))
    tmp_3.append(np.linalg.norm(W2[i]-W2[i+1], ord=2))
    tmp_4.append(np.linalg.norm(b2[i]-b2[i+1], ord=2))

tmp_1 = np.array(tmp_1)
tmp_2 = np.array(tmp_2)
tmp_3 = np.array(tmp_3)
tmp_4 = np.array(tmp_4)

collection_norm = db["N_Form"]

# with open(os.path.join(os.path.dirname(__file__), "logs", "log.txt"), "a") as f:
#     for i in range(100):
#         f.write(f"{i+rnd*100} {tmp_1[i]} {tmp_2[i]} {tmp_3[i]} {tmp_4[i]}\n")
#     f.write("\n")
#     f.close()

for i in range(100):
    collection_norm.insert_one(
        {"index": i+rnd*100, "W1": tmp_1[i], "b1": tmp_2[i], "W2": tmp_3[i], "b2": tmp_4[i]})

# fig, axs = plt.subplots(2, 2, figsize=(16, 12))
# axs[0, 0].plot(tmp_1, label="W1")
# axs[0, 0].legend()
# axs[0, 0].set_title("W1")
# axs[0, 1].plot(tmp_2, label="b1")
# axs[0, 1].legend()
# axs[0, 1].set_title("b1")
# axs[1, 0].plot(tmp_3, label="W2")
# axs[1, 0].legend()
# axs[1, 0].set_title("W2")
# axs[1, 1].plot(tmp_4, label="b2")
# axs[1, 1].legend()
# axs[1, 1].set_title("b2")
# plt.suptitle("Normalization Parameters", fontsize=16)
# plt.tight_layout()
# plt.savefig(os.path.join(os.path.dirname(
#     __file__), "images", "norm_params.png"))
