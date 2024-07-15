import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import plotly.express as px

## Importação dos datasets
   # men and women 
data1 = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data1.csv')
print(data1.head())
print(data1.shape)
   # dataset com 76 variáveis e 3982 observações

## Analise  
   # estatísticas descritivas
pd.set_option('display.max_columns', None)
print(data1.describe())
   # correlação 
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

## Cluster 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Carregar dados
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Pré-processamento dos dados
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_df)

# Adicionar os rótulos dos clusters ao DataFrame original
df['Cluster'] = kmeans.labels_

# PCA para visualização
pca = PCA(n_components=2)
pca_df = pca.fit_transform(scaled_df)
pca_df = pd.DataFrame(pca_df, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = kmeans.labels_

# Plotar os clusters
plt.figure(figsize=(10, 6))
plt.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df['Cluster'], cmap='viridis')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Clusters of Iris Data')
plt.show()
