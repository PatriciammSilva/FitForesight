## Importação Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import plotly.express as px

## Importação do dataset
data3 = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data3.csv')
print(data3.head())
print(data3.shape)
   # dataset com 25 variáveis e 3982 observações

## Análise  
   # estatísticas descritivas
pd.set_option('display.max_columns', None)
print(data3.describe())
   # correlação 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)  
R3 = data3.corr()
print(R3)
   # gráfico 1 : correlações (estranho)
plt.figure(figsize=(10, 8))
sns.heatmap(R3, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap da Matriz de Correlação')
plt.show()
   # gráfico 2 : correlações (online)
fig = px.imshow(R3, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Heatmap da Matriz de Correlação")
fig.show()
   # gráfico 3 : correlações (online)
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(R3, cmap='coolwarm')
fig.colorbar(cax)
plt.title('Matriz de Correlação com Matplotlib', pad=20)
plt.show()

## Normalizar data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
ndata3 = scaler.fit_transform(data3)

## Cluster - kmeans
from sklearn.cluster import KMeans
   # aplicar K-Means 
kmeans = KMeans(n_clusters=4)
kmeans.fit(ndata3)
   # previsao
labels = kmeans.predict(ndata3)
   # gráfico
plt.scatter(ndata3[:, 0], ndata3[:, 1], c=labels, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.show()

## Cluster hierárquico
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
   # dendrograma
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(ndata3, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Euclidean distances')
plt.show()
   # aplicar cluster
hc = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
labels_hc = hc.fit_predict(ndata3)
   # gráfico
plt.scatter(ndata3[:, 0], ndata3[:, 1], c=labels_hc, s=50, cmap='viridis')
plt.show()

## Cluster DBSCAN
from sklearn.cluster import DBSCAN
   # aplicar DBSCAN 
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels_dbscan = dbscan.fit_predict(ndata3)
   # gráfico
plt.scatter(ndata3[:, 0], ndata3[:, 1], c=labels_dbscan, s=50, cmap='viridis')
plt.show()

## Análise de Componentes Principais
from sklearn.decomposition import PCA
   # Aplicar PCA
pca = PCA(n_components=2)
pca3 = pca.fit_transform(ndata3)
print(pca3)
   # converter em DataFrame
df_pca = pd.DataFrame(data=pca3, columns=['PC1', 'PC2'])
   # gráfico
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=df_pca, palette='viridis', s=100, alpha=0.7)
plt.title('PCA dos Dados Sintéticos')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()
   # variância explicada por cada componente principal
explained_variance = pca.explained_variance_ratio_
print(explained_variance)
print(f'Variância explicada pelo PC1: {explained_variance[0]:.2f}')
print(f'Variância explicada pelo PC2: {explained_variance[1]:.2f}')
   # gráfico da variância explicada
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', color='red')
plt.xlabel('Componente Principal')
plt.ylabel('Proporção de Variância Explicada')
plt.title('Variância Explicada pelos Componentes Principais')
plt.show()
