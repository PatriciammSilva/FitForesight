## PCA

   
## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets3/data3.csv')

## Aplicar PCA
pca = PCA(n_components=4)
dfpca = pca.fit_transform(df)
print(dfpca)

## Converter em DataFrame
dfpca = pd.DataFrame(data=dfpca, columns=['PC1', 'PC2', 'PC3', 'PC4'])
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
