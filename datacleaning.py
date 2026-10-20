import pandas as pd
import numpy as np
df=pd.read_csv("iris.csv")
df.drop_duplicates(inplace=True)
print(df.describe())
print(df.isnull().sum())