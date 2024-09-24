## T-sne

   
## Packages necessários
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import joblib


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets4/data4.csv')

## Normalizar dataset
scaler = StandardScaler()
dfnor = scaler.fit_transform(df)

## Aplicar t-SNE
tsne = TSNE(n_components=2, random_state=42)
dftsne = tsne.fit_transform(dfnor)

## Gráfico
dftsne = pd.DataFrame(data=dftsne, columns=['TSNE1', 'TSNE2'])
plt.figure(figsize=(10, 7))
sns.scatterplot(x='TSNE1', y='TSNE2', data=dftsne,  palette='Set1')
plt.title('t-SNE - Componentes')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.show()


## Normalizar dataset
scaler = StandardScaler()
dfnor = scaler.fit_transform(dftsne)

## Aplicar kmeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(dfnor)
joblib.dump(kmeans, 'modtsne4.pkl')

## Previsão
kmeans = joblib.load('modtsne4.pkl')
labels = kmeans.predict(dfnor)
print(labels)

## Gráfico
plt.scatter(dfnor[:, 0], dfnor[:, 1], c=labels, s=50, cmap='viridis')
cent = kmeans.cluster_centers_
plt.scatter(cent[:, 0], cent[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.show()