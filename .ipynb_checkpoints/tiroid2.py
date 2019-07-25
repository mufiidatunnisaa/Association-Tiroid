#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

data_tiroid_missing = pd.read_csv('data_tiroid_missing.csv', delimiter = ',', header= None)
data_tiroid_missing.columns = ["column 1", "column 2", "column 3", "column 4", "column 5", "column 6"]
data_tiroid_new = pd.DataFrame(data_tiroid_missing)
data_tiroid_new = data_tiroid_new.replace('?', np.nan)
data_tiroid_new = data_tiroid_new.fillna(data_tiroid_new.median())
print(data_tiroid_new) 

