import os
import pandas as pd
import shutil

IMG_DIR_PATH = "../november-train/orig_images/"
PARENT_DIR = "raw_clusters_with_bbox/"

df = pd.read_csv("raw_clusters_with_bbox.csv")
cluster_num = max(df["Cluster"]) + 1
for i in range(cluster_num):
    path = PARENT_DIR + f"cluster_{i}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    
count_clusters = [0] * cluster_num
    
for i in range(len(df)):
    cluster = df["Cluster"].iloc[i]
    name = df["name"].iloc[i]
    count_clusters[cluster] += 1
    
    shutil.copy(IMG_DIR_PATH + name + ".jpg", PARENT_DIR + f"cluster_{cluster}/" + name + ".jpg")
print(count_clusters)