## Multidimentional scaling

   
## Packages necessários
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.cluster import KMeans


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets4/data4.csv')

## Normalizar dataset
scaler = StandardScaler()
dfnor = scaler.fit_transform(df)

## Aplicar MDS
mds = MDS(n_components=2, random_state=42)
dfmds = mds.fit_transform(dfnor)
   
## Gráfico
dfmds = pd.DataFrame(data=dfmds, columns=['MDS1', 'MDS2'])
plt.figure(figsize=(10, 7))
sns.scatterplot(x='MDS1', y='MDS2', data=dfmds, palette='Set1')
plt.title('MDS - Componentes')
plt.xlabel('MDS1')
plt.ylabel('MDS2')
plt.show()

## Normalizar dataset
scaler = StandardScaler()
dfnor = scaler.fit_transform(dfmds)

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