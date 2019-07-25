import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

data_tiroid_missing = pd.read_csv('data_tiroid_missing.csv', delimiter = ',', header= None)
print(data_tiroid_missing)
