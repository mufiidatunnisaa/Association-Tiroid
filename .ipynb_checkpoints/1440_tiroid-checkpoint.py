#%%
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data = np.genfromtxt('tiroid.csv', delimiter=',')
data
df = pd.DataFrame(data)
df = df.fillna(df.mean())
jumlahData = len(df)
df = df.iloc[:,0:5]
np.savetxt("new_data_tiroid.csv", df, delimiter=",", fmt="%7.3f")



#Min_Max Method
min = 0
max = 1
data_std = (df - df.min()) / (df.max()- df.min())
data_minmax = data_std * (max - min) + min
data_minmax 
np.savetxt("minmax.csv", data_minmax, delimiter=",", fmt="%7.3f")

#Z-score
std = df.std()
data_zscore = (df-df.mean())/std
data_zscore
np.savetxt("zscore.csv", data_zscore, delimiter=",", fmt="%7.3f")

#Sigmoidal
x = (df-df.mean())/std
data_sigmoidal = (1-np.exp(-x))/(1+np.exp(-x))
data_sigmoidal
np.savetxt("sigmoidal.csv", data_sigmoidal, delimiter=",", fmt="%7.3f")

#NN data
label = data[:,5]
knn =KNeighborsClassifier(n_neighbors=3)
knn.fit(df, label)
result = knn.predict(df)
error = (np.sum(label != result)/jumlahData)*100
print("Error tanpa normalisasi= ", error, "%")

#NN data MinMax
knn.fit(data_minmax, label)
result = knn.predict(data_minmax)
error_minmax = (np.sum(label != result)/jumlahData)*100
print("Error minmax = ", error_minmax, "%")

#NN data Zscore
data_zscore = data_zscore.iloc[:,0:5]
knn.fit(data_zscore, label)
result = knn.predict(data_zscore)
error_zscore = (np.sum(label != result)/jumlahData)*100
print("Error zscore = ", error_zscore, "%")

#NN data Sigmoidal
data_sigmoidal = data_sigmoidal.iloc[:,0:5]
knn.fit(data_sigmoidal, label)
result = knn.predict(data_sigmoidal)
error_sigmoidal = (np.sum(label != result)/jumlahData)*100
print("Error sigmoidal = ", error_sigmoidal, "%")

# #%%
