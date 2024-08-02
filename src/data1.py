## Data 1 - 25 variáveis 

## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
#import yellowbrick

## Importação Dataset
data1 = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data1.csv')
print(data1.head())
print(data1.shape)

## Estatísticas Descritivas
pd.set_option('display.max_columns', None)
print(data1.describe())

## Correlação 
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
