## ICA

   
## Packages necessários
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data4.csv')

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
