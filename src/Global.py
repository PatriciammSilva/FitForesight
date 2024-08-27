## Global
   # alterar o número do dataset no comando de importação
   
## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.decomposition import FastICA
import statsmodels.api as sm
import statsmodels.formula.api as smf


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data4.csv')


## Normalizar dataset
scaler = StandardScaler()
dfnor = scaler.fit_transform(df)


## Estatísticas Descritivas
pd.set_option('display.max_columns', None)
print(df.describe())


## Correlação 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)  
R = df.corr()
print(R)
   # Gráfico 1 : correlações (estranho)
plt.figure(figsize=(10, 8))
sns.heatmap(R, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.show()
   # Gráfico 2 : correlações (online)
fig = px.imshow(R, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Heatmap da Matriz de Correlação")
fig.show()
   # Gráfico 3 : correlações (online)
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(R, cmap='coolwarm')
fig.colorbar(cax)
plt.title('Matriz de Correlação', pad=20)
plt.show()


## Cluster
kmeans = KMeans(n_clusters=4)
kmeans.fit(dfnor)
joblib.dump(kmeans, 'modkmeans.pkl')
   # Previsão
kmeans = joblib.load('modkmeans.pkl')
labels = kmeans.predict(dfnor)
print(labels)
   # Gráfico
plt.scatter(dfnor[:, 0], dfnor[:, 1], c=labels, s=50, cmap='viridis')
cent = kmeans.cluster_centers_
plt.scatter(cent[:, 0], cent[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.show()


## PCA
pca = PCA(n_components=4)
dfpca = pca.fit_transform(df)
print(dfpca)
   # Converter em DataFrame
dfpca = pd.DataFrame(data=dfpca, columns=['PC1', 'PC2', 'PC3', 'PC4'])
print(dfpca)
   # Variância explicada 
explained_variance = pca.explained_variance_ratio_
print(explained_variance)
   # Gráfico da variância explicada
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', color='red')
plt.xlabel('Componente Principal')
plt.ylabel('Proporção de Variância Explicada')
plt.title('Variância Explicada pelos Componentes Principais')
plt.show()


## FA
fa = FactorAnalysis(n_components=4, random_state=42)
fact = fa.fit_transform(dfnor)
print(fact)


## T-SNE
tsne = TSNE(n_components=2, random_state=42)
dftsne = tsne.fit_transform(dfnor)
   # Gráfico
dftsne = pd.DataFrame(data=dftsne, columns=['TSNE1', 'TSNE2'])
plt.figure(figsize=(10, 7))
sns.scatterplot(x='TSNE1', y='TSNE2', data=dftsne, palette='Set1')
plt.title('t-SNE - Componentes')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.show()


## PCA (2 dimensões)
pca = PCA(n_components=2)
dfpca = pca.fit_transform(df)
print(dfpca)
   # Converter em DataFrame
dfpca = pd.DataFrame(data=dfpca, columns=['PC1', 'PC2'])
print(dfpca)
   # Variância explicada
explained_variance = pca.explained_variance_ratio_
print(explained_variance)
   # Gráfico da variância explicada
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', color='red')
plt.xlabel('Componente Principal')
plt.ylabel('Proporção de Variância Explicada')
plt.title('Variância Explicada pelos Componentes Principais')
plt.show()


## Multidimentional Scaling
mds = MDS(n_components=2, random_state=42)
dfmds = mds.fit_transform(dfnor)
   # Gráfico
dfmds = pd.DataFrame(data=dfmds, columns=['MDS1', 'MDS2'])
plt.figure(figsize=(10, 7))
sns.scatterplot(x='MDS1', y='MDS2', data=dfmds, palette='Set1')
plt.title('MDS - Componentes')
plt.xlabel('MDS1')
plt.ylabel('MDS2')
plt.show()


## ICA
ica = FastICA(n_components=2, random_state=42)
dfica = ica.fit_transform(dfnor)
   # Gráficos
plt.figure(figsize=(12, 8))
for i in range(dfica.shape[1]):
    plt.subplot(dfica.shape[1], 1, i+1)
    plt.plot(dfica[:, i])
    plt.title(f'Componente Independente {i+1}')
    plt.xlabel('Amostra')
    plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()


## Modelos R > 0.60
