import pandas as pd
import nltk
import numpy as np

df = pd.read_csv("data/raw/train.csv", index_col='id')
df[['POI', 'street']] = df['POI/street'].str.split('/', 1, expand=True)

cond=[True if x!='' and x not in y else False for x,y in zip(df['POI'], df['raw_address'])]
print(df[cond])
