## Data 1 - 76 variáveis 

## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import plotly.express as px


## Importação Dataset
data1 = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data1.csv')
print(data1.head())
print(data1.shape)


## Estatísticas Descritivas
pd.set_option('display.max_columns', None)
print(data1.describe())


## Correlação 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)  
R1 = data1.corr()
print(R1)
   # gráfico 1 : correlações (estranho)
plt.figure(figsize=(10, 8))
sns.heatmap(R1, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap da Matriz de Correlação')
plt.show()
   # gráfico 2 : correlações (online)
fig = px.imshow(R1, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Heatmap da Matriz de Correlação")
fig.show()
   # gráfico 3 : correlações (online)
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(R1, cmap='coolwarm')
fig.colorbar(cax)
plt.title('Matriz de Correlação com Matplotlib', pad=20)
plt.show()


## Cluster - kmeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
   # Normalizar data
scaler = StandardScaler()
ndata1 = scaler.fit_transform(data1)
   # Aplicar kmeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(ndata1)
   # previsao
labels = kmeans.predict(ndata1)
   # gráfico
plt.scatter(ndata1[:, 0], ndata1[:, 1], c=labels, s=50, cmap='viridis')
cent1 = kmeans.cluster_centers_
plt.scatter(cent1[:, 0], cent1[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.show()


## Análise de Componentes Principais
from sklearn.decomposition import PCA
   # Aplicar PCA
pca = PCA(n_components=4)
pca1 = pca.fit_transform(ndata1)
print(pca1)
   # converter em DataFrame
datapca1 = pd.DataFrame(data=pca1, columns=['PC1', 'PC2', 'PC3', 'PC4'])
print(datapca1)
   # variância explicada por cada componente principal
explained_variance = pca.explained_variance_ratio_
print(explained_variance)
   # gráfico da variância explicada
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', color='red')
plt.xlabel('Componente Principal')
plt.ylabel('Proporção de Variância Explicada')
plt.title('Variância Explicada pelos Componentes Principais')
plt.show()


## Análise Fatorial
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
   # Normalizar data
scaler = StandardScaler()
ndata1 = scaler.fit_transform(data1)
   # Aplicar FA
fa = FactorAnalysis(n_components=4, random_state=42)
fact = fa.fit_transform(ndata1)
print(fact)


## T-sne
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
   # Normalizar data
scaler = StandardScaler()
ndata1 = scaler.fit_transform(data1)
   # Aplicar t-SNE
tsne1 = TSNE(n_components=2, random_state=42)
d1tsne = tsne1.fit_transform(ndata1)
   # Gráfico
d1 = pd.DataFrame(data=d1tsne, columns=['TSNE1', 'TSNE2'])
plt.figure(figsize=(10, 7))
sns.scatterplot(x='TSNE1', y='TSNE2', data=d1, palette='Set1')
plt.title('t-SNE - Componentes')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.show()

## Análise de Componentes Principais (2 dimensões)
from sklearn.decomposition import PCA
   # Aplicar PCA
pca = PCA(n_components=2)
pca1 = pca.fit_transform(ndata1)
print(pca1)
   # converter em DataFrame
datapca1 = pd.DataFrame(data=pca1, columns=['PC1', 'PC2'])
print(datapca1)
   # variância explicada por cada componente principal
explained_variance = pca.explained_variance_ratio_
print(explained_variance)
   # gráfico da variância explicada
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', color='red')
plt.xlabel('Componente Principal')
plt.ylabel('Proporção de Variância Explicada')
plt.title('Variância Explicada pelos Componentes Principais')
plt.show()


## Muldimentional scaling
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
   # Normalizar data
scaler = StandardScaler()
ndata1 = scaler.fit_transform(data1)
   # Aplicar MDS
mds1 = MDS(n_components=2, random_state=42)
d1mds = mds1.fit_transform(ndata1)
   # Gráfico
d1 = pd.DataFrame(data=d1mds, columns=['MDS1', 'MDS2'])
plt.figure(figsize=(10, 7))
sns.scatterplot(x='MDS1', y='MDS2', data=d1, palette='Set1')
plt.title('MDS - Componentes')
plt.xlabel('MDS1')
plt.ylabel('MDS2')
plt.show()


## ICA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
   # Normalizar data
scaler = StandardScaler()
ndata1 = scaler.fit_transform(data1)
   # Aplicar ICA
ica1 = FastICA(n_components=2, random_state=42)
d1ica = ica1.fit_transform(ica1)
   # Gráficos
d1 = pd.DataFrame(data=d1ica, columns=['ICA1', 'ICA2'])
plt.figure(figsize=(10, 7))
sns.scatterplot(x='ICA1', y='ICA2', hue='species', data=d1, palette='Set1')
plt.title('ICA - Componentes Independentes')
plt.xlabel('ICA1')
plt.ylabel('ICA2')
plt.show()
