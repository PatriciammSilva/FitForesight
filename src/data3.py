## Data 3 - 25 variáveis 

## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import plotly.express as px


## Importação Dataset
data3 = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data3.csv')
print(data3.head())
print(data3.shape)


## Estatísticas Descritivas
pd.set_option('display.max_columns', None)
print(data3.describe())


## Correlação 
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


## Cluster - kmeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
   # Normalizar data
scaler = StandardScaler()
ndata3 = scaler.fit_transform(data3)
   # Aplicar kmeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(ndata3)
   # previsao
labels = kmeans.predict(ndata3)
   # gráfico
plt.scatter(ndata3[:, 0], ndata3[:, 1], c=labels, s=50, cmap='viridis')
cent3 = kmeans.cluster_centers_
plt.scatter(cent3[:, 0], cent3[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.show()


## Análise de Componentes Principais
from sklearn.decomposition import PCA
   # Aplicar PCA
pca = PCA(n_components=3)
pca3 = pca.fit_transform(ndata3)
print(pca3)
   # converter em DataFrame
datapca3 = pd.DataFrame(data=pca3, columns=['PC1', 'PC2', 'PC3'])
print(datapca3)
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
ndata3 = scaler.fit_transform(data3)
   # Aplicar FA
fa = FactorAnalysis(n_components=3, random_state=42)
fact = fa.fit_transform(ndata3)
print(fact)


## T-sne
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
   # Normalizar data
scaler = StandardScaler()
ndata3 = scaler.fit_transform(data3)
   # Aplicar t-SNE
tsne3 = TSNE(n_components=2, random_state=42)
d3tsne = tsne3.fit_transform(ndata3)
   # Gráfico
d3 = pd.DataFrame(data=d3tsne, columns=['TSNE1', 'TSNE2'])
plt.figure(figsize=(10, 7))
sns.scatterplot(x='TSNE1', y='TSNE2', data=d3, palette='Set1')
plt.title('t-SNE - Componentes')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.show()

## Análise de Componentes Principais (2 dimensões)
from sklearn.decomposition import PCA
   # Aplicar PCA
pca = PCA(n_components=2)
pca3 = pca.fit_transform(ndata3)
print(pca3)
   # converter em DataFrame
datapca3 = pd.DataFrame(data=pca3, columns=['PC1', 'PC2'])
print(datapca3)
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
ndata3 = scaler.fit_transform(data3)
   # Aplicar MDS
mds3 = MDS(n_components=2, random_state=42)
d3mds = mds3.fit_transform(ndata3)
   # Gráfico
d3 = pd.DataFrame(data=d3mds, columns=['MDS1', 'MDS2'])
plt.figure(figsize=(10, 7))
sns.scatterplot(x='MDS1', y='MDS2', data=d3, palette='Set1')
plt.title('MDS - Componentes')
plt.xlabel('MDS1')
plt.ylabel('MDS2')
plt.show()


## ICA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
   # Normalizar data
scaler = StandardScaler()
ndata3 = scaler.fit_transform(data3)
   # Aplicar ICA
ica3 = FastICA(n_components=2, random_state=42)
d3ica = ica3.fit_transform(ica3)
   # Gráficos
d3 = pd.DataFrame(data=d3ica, columns=['ICA1', 'ICA2'])
plt.figure(figsize=(10, 7))
sns.scatterplot(x='ICA1', y='ICA2', hue='species', data=d3, palette='Set1')
plt.title('ICA - Componentes Independentes')
plt.xlabel('ICA1')
plt.ylabel('ICA2')
plt.show()
