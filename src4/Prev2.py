## Previsões Mod2 - 0.65


## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib


## Recuperar modelo e importar dataframe
kmeans = joblib.load('modkmeans4.pkl')
df2 = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets4/df2.csv')

## Normalizar dataset
scaler = StandardScaler()
df2nor = scaler.fit_transform(df2)

## Previsão
kmeans = joblib.load('modkmeans4.pkl')
labels = kmeans.predict(df2nor)
np.set_printoptions(threshold=np.inf)
print(labels)
cluster_counts = pd.Series(labels).value_counts()
print(cluster_counts)

## Gráfico
plt.scatter(df2nor[:, 0], df2nor[:, 1], c=labels, s=50, cmap='viridis')
cent = kmeans.cluster_centers_
plt.scatter(cent[:, 0], cent[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.show()