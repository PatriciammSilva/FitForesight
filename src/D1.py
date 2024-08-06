## Data 1 - 76 variáveis 

## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import plotly.express as px


## Importação Dataset
data = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data1.csv')


## Estatísticas Descritivas
# pd.set_option('display.max_columns', None)
print(data.describe())


## Correlação 
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)  
R = data.corr()
print(R)
   # gráfico 1 : correlações (estranho)
plt.figure(figsize=(10, 8))
sns.heatmap(R, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.show()
   # gráfico 2 : correlações (online)
fig = px.imshow(R, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Heatmap da Matriz de Correlação")
fig.show()
   # gráfico 3 : correlações
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(R, cmap='coolwarm')
fig.colorbar(cax)
plt.title('Matriz de Correlação', pad=20)
plt.show()


## Cluster - kmeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
   # Normalizar data
scaler = StandardScaler()
ndata = scaler.fit_transform(data)
   # Aplicar kmeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(ndata)
import joblib
joblib.dump(kmeans, 'kmeans_model.pkl')
kmeans = joblib.load('kmeans_model.pkl')
   # previsao
labels = kmeans.predict(ndata)
   # gráfico
plt.scatter(ndata[:, 0], ndata[:, 1], c=labels, s=50, cmap='viridis')
cent = kmeans.cluster_centers_
plt.scatter(cent[:, 0], cent[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.show()


## Análise de Componentes Principais
from sklearn.decomposition import PCA
   # Aplicar PCA
pca = PCA(n_components=4)
pca1 = pca.fit_transform(ndata)
print(pca)
   # converter em DataFrame
datapca = pd.DataFrame(data=pca1, columns=['PC1', 'PC2', 'PC3', 'PC4'])
print(datapca)
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
ndata = scaler.fit_transform(data)
   # Aplicar FA
fa = FactorAnalysis(n_components=4, random_state=42)
fact = fa.fit_transform(ndata)
print(fact)


## T-sne
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
   # Normalizar data
scaler = StandardScaler()
ndata = scaler.fit_transform(data)
   # Aplicar t-SNE
tsne = TSNE(n_components=2, random_state=42)
dtsne = tsne.fit_transform(ndata)
   # Gráfico
d = pd.DataFrame(data=dtsne, columns=['TSNE1', 'TSNE2'])
plt.figure(figsize=(10, 7))
sns.scatterplot(x='TSNE1', y='TSNE2', data=d, palette='Set1')
plt.title('t-SNE - Componentes')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.show()

## Análise de Componentes Principais (2 dimensões)
from sklearn.decomposition import PCA
   # Aplicar PCA
pca = PCA(n_components=2)
pca = pca.fit_transform(ndata)
print(pca)
   # converter em DataFrame
datapca = pd.DataFrame(data=pca, columns=['PC1', 'PC2'])
print(datapca)
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


## Multidimentional scaling
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
   # Normalizar data
scaler = StandardScaler()
ndata = scaler.fit_transform(data)
   # Aplicar MDS
mds = MDS(n_components=2, random_state=42)
dmds = mds.fit_transform(ndata)
   # Gráfico
d = pd.DataFrame(data=dmds, columns=['MDS1', 'MDS2'])
plt.figure(figsize=(10, 7))
sns.scatterplot(x='MDS1', y='MDS2', data=d, palette='Set1')
plt.title('MDS - Componentes')
plt.xlabel('MDS1')
plt.ylabel('MDS2')
plt.show()


## ICA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
   # Normalizar data
scaler = StandardScaler()
ndata = scaler.fit_transform(data)
   # Aplicar ICA
ica = FastICA(n_components=2, random_state=42)
dica = ica1.fit_transform(data)
   # Gráficos
d = pd.DataFrame(data=dica, columns=['ICA1', 'ICA2'])
plt.figure(figsize=(10, 7))
sns.scatterplot(x='ICA1', y='ICA2', hue='species', data=d1, palette='Set1')
plt.title('ICA - Componentes Independentes')
plt.xlabel('ICA1')
plt.ylabel('ICA2')
plt.show()
