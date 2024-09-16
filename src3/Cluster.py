## Cluster
  
   
## Packages necessários
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets3/data3.csv')

## Normalizar dataset
scaler = StandardScaler()
dfnor = scaler.fit_transform(df)

## Aplicar kmeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(dfnor)
joblib.dump(kmeans, 'modkmeans3.pkl')

## Previsão
kmeans = joblib.load('modkmeans3.pkl')
labels = kmeans.predict(dfnor)
print(labels)

## Gráfico
plt.scatter(dfnor[:, 0], dfnor[:, 1], c=labels, s=50, cmap='viridis')
cent = kmeans.cluster_centers_
plt.scatter(cent[:, 0], cent[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.show()