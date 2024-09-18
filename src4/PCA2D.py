## PCA

   
## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets4/data4.csv')

## Aplicar PCA
pca = PCA(n_components=2)
dfpca = pca.fit_transform(df)
print(dfpca)

## Converter em DataFrame
dfpca = pd.DataFrame(data=dfpca, columns=['PC1', 'PC2'])
print(dfpca)

## Variância explicada
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

## Gráfico da variância explicada
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', color='red')
plt.xlabel('Componente Principal')
plt.ylabel('Proporção de Variância Explicada')
plt.title('Variância Explicada pelos Componentes Principais')
plt.show()

## Normalizar dataset
scaler = StandardScaler()
dfnor = scaler.fit_transform(dfpca)

## Aplicar kmeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(dfnor)

## Previsão
labels = kmeans.predict(dfnor)
print(labels)

## Gráfico
plt.scatter(dfnor[:, 0], dfnor[:, 1], c=labels, s=50, cmap='viridis')
cent = kmeans.cluster_centers_
plt.scatter(cent[:, 0], cent[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.show()