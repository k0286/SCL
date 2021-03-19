import copy

import numpy as np
import pandas as pd

df = pd.read_csv("data/raw/train.csv", index_col="id")
df[["POI", "street"]] = df["POI/street"].str.split("/", 1, expand=True)
df["POI/street"].replace("/", np.nan, inplace=True)
df["POI"].replace("", np.nan, inplace=True)
df["street"].replace("", np.nan, inplace=True)
# cond = [True if x is not np.nan and x not in y else False for x, y in zip(df["POI"], df["raw_address"])]
# print(df[cond])

df["POI_tokens"] = df["POI"].str.split()
df["street_tokens"] = df["street"].str.split()
df["raw_addr_tokens"] = df["raw_address"].str.replace(",", "").str.split()

# for ner label
"""
O:
B-P
I-P
E-P
B-S
I-S
E-S
"""
# df['length'] = df['raw_address'].str.split().apply(len)
# print(max(df['length']))  # words length
# df['street_loc'] = [(x.find(y), len(y)) if y is not np.nan else np.nan for x,y in zip(df["raw_address"], df['street'])]  # (start, offset)
# df['street_label']=[for x,y in zip(df["raw_address"], df['street'])]
# label = ['O '* x for x in df['length']]
#
# print(df['street_loc'].head(100))


def foo(bar: list) -> list:
    """

    :param x: raw addr
    :param y: street
    :return: label->list
    """
    label = []
    x = bar[0]
    y = copy.deepcopy(bar[1])
    for i in x:
        if y is np.nan or len(y) == 0 or i != y[0]:
            label.append("O")
        elif "B-S" not in label:
            y.pop(0)
            label.append("B-S")
        elif len(y) == 1:
            y.pop(0)
            label.append("E-S")
        else:
            y.pop(0)
            label.append("I-S")
    return label


def bar(foo: list) -> list:
    x = foo[0]  # raw street
    y = copy.deepcopy(foo[1])  # poi
    z = foo[2]  # label
    for ind in range(len(z)):
        if y is np.nan or len(y) == 0 or x[ind] != y[0]:
            pass
        elif "B-P" not in z:
            y.pop(0)
            z[ind] = "B-P"
        elif len(y) == 1:
            y.pop(0)
            z[ind] = "E-P"
        else:
            y.pop(0)
            z[ind] = "I-P"
    return z


df["label"] = df[["raw_addr_tokens", "street_tokens"]].apply(foo, axis=1)
df["label"] = df[["raw_addr_tokens", "POI_tokens", "label"]].apply(bar, axis=1)

# drop_list=[203442, 93166, 190118, 207284, 176809, 2058, 96763]
drop_list = [203442, 93166]
df.drop(df.index[drop_list], inplace=True)
df.to_csv("data/train_ner.csv")
print("hi")
