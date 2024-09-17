## Multidimentional scaling

   
## Packages necessários
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets3/data3.csv')

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