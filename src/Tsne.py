## T-sne
   # alterar o número do dataset no comando de importação
   
## Packages necessários
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data1.csv')

## Normalizar dataset
scaler = StandardScaler()
dfnor = scaler.fit_transform(df)

## Aplicar t-SNE
tsne = TSNE(n_components=2, random_state=42)
dftsne = tsne.fit_transform(dfnor)

## Gráfico
dftsne = pd.DataFrame(data=dftsne, columns=['TSNE1', 'TSNE2'])
plt.figure(figsize=(10, 7))
sns.scatterplot(x='TSNE1', y='TSNE2', data=dftsne, palette='Set1')
plt.title('t-SNE - Componentes')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.show()