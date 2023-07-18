import pandas as pd
import os

# get folder list in './logs/tmp/Comparison model speed and memory usage according to community size'
path = "./logs/tmp/Comparison model speed and memory usage according to community size/"
file_list = os.listdir(path)
# check whether each file is a folder or not
folder_list = []
for file_ in file_list:
    if os.path.isdir(path + file_):
        folder_list.append(file_)
folder_list.sort()


# get 'log.log' in each folder
df_time = pd.DataFrame(columns=["collab", "yelp", "bitcoin", "wikielec", "redditbody"])
df_space = pd.DataFrame(columns=["collab", "yelp", "bitcoin", "wikielec", "redditbody"])
df_comm_num = pd.DataFrame(
    columns=["collab", "yelp", "bitcoin", "wikielec", "redditbody"]
)
for folder in folder_list:
    dataset = folder.split("_")[2]
    num = folder.split("_")[3]
    with open(f"{path}/{folder}/log.log", "r") as f:
        # find the second "main_model time: 0.23033428192138672" and extract the number
        i = 0
        j = 0
        for line in f:
            if "main_model time" in line:
                i += 1
                if i == 2:
                    df_time.loc[num, dataset] = line.split(" ")[-1].strip()[:4] + " s"
            if "GPU usage" in line:
                j += 1
                if j == 2:
                    df_space.loc[num, dataset] = line.split(" ")[-2] + " MiB"
            if "Comm number max" in line:
                df_comm_num.loc[num, dataset] = int(line.split(" ")[-1].strip()) + 1

df_time.to_csv(path + "time.csv")
df_space.to_csv(path + "space.csv")
df_comm_num.to_csv(path + "comm_num.csv")
