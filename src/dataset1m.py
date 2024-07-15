import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import plotly.express as px

## Importação dos datasets
   # men and women 
data1m = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data1m.csv')
print(data1m.head())
print(data1m.shape)
   # dataset com 37 variáveis e 3982 observações

## Analise  
   # estatísticas descritivas
pd.set_option('display.max_columns', None)
print(data1m.describe())
   # correlação 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)  
R1m = data1m.corr()
print(R1m)
   # gráfico 1 : correlações (estranho)
plt.figure(figsize=(10, 8))
sns.heatmap(R1m, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap da Matriz de Correlação')
plt.show()
   # gráfico 2 : correlações (online)
fig = px.imshow(R1m, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Heatmap da Matriz de Correlação")
fig.show()
   # gráfico 3 : correlações (online)
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(R1m, cmap='coolwarm')
fig.colorbar(cax)
plt.title('Matriz de Correlação com Matplotlib', pad=20)
plt.show()

## Cluster 