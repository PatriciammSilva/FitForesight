## ICA

   
## Packages necessários
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets4/data4.csv')

## Normalizar dataset
scaler = StandardScaler()
dfnor = scaler.fit_transform(df)

## Aplicar ICA
ica = FastICA(n_components=2, random_state=42)
dfica = ica.fit_transform(dfnor)

## Gráficos
plt.figure(figsize=(12, 8))
for i in range(dfica.shape[1]):
    plt.subplot(dfica.shape[1], 1, i+1)
    plt.plot(dfica[:, i])
    plt.title(f'Componente Independente {i+1}')
    plt.xlabel('Amostra')
    plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

## Normalizar dataset
scaler = StandardScaler()
dfnor = scaler.fit_transform(dfica)

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