## Correlação 


## Packages necessários
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets2/data2.csv')

## Correlação 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)  
R = df.corr()
print(R)

## Gráfico 1 : correlações (estranho)
plt.figure(figsize=(10, 8))
sns.heatmap(R, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.show()

## Gráfico 2 : correlações (online)
fig = px.imshow(R, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Heatmap da Matriz de Correlação")
fig.show()
   
## Gráfico 3 : correlações (online)
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(R, cmap='coolwarm')
fig.colorbar(cax)
plt.title('Matriz de Correlação', pad=20)
plt.show()