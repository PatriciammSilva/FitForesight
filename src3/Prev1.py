## Previsões Mod1 - 0.60


## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib


## Recuperar modelo e importar dataframe
kmeans = joblib.load('modkmeans3.pkl')
df1 = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets3/df1.csv')

## Normalizar dataset
scaler = StandardScaler()
df1nor = scaler.fit_transform(df1)


## Previsão
labels = kmeans.predict(df1nor)
print(labels)
np.set_printoptions(threshold=np.inf)
print(labels)
cluster_counts = pd.Series(labels).value_counts()
print(cluster_counts)

## Gráfico
plt.scatter(df1nor[:, 0], df1nor[:, 1], c=labels, s=50, cmap='viridis')
cent = kmeans.cluster_centers_
plt.scatter(cent[:, 0], cent[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.show()