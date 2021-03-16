import nltk
import numpy as np
import pandas as pd

df = pd.read_csv("data/raw/train.csv", index_col="id")
df[["POI", "street"]] = df["POI/street"].str.split("/", 1, expand=True)
df["POI/street"].replace("/", np.nan, inplace=True)
df["POI"].replace("", np.nan, inplace=True)
df["street"].replace("", np.nan, inplace=True)
cond = [True if x not in y else False for x, y in zip(df["POI"], df["raw_address"])]
print(df[cond])
