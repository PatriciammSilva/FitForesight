## Correlação 


## Packages necessários
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data4.csv')

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


## Somatórios
cor1 = 0.58 + 0.52 + 0.55 + 0.69 + 0.61 + 0.96 + 0.62 + 0.51 + 0.73
   # 5.77
cor2 = 0.58 + 0.80 + 0.72 + 0.85 + 0.66 + 0.57 + 0.52 + 0.48 + 0.84
   # 6.02
cor3 = 0.52 + 0.80 + 0.77 + 0.73 + 0.62 + 0.48 + 0.48 + 0.46 + 0.70
   # 5.56
cor4 = 0.55 + 0.72 + 0.77 + 0.76 + 0.63 + 0.53 + 0.56 + 0.49 + 0.72
   # 5.73
cor5 = 0.69 + 0.85 + 0.73 + 0.76 + 0.73 + 0.69 + 0.66 + 0.55 + 0.88
   # 6.54
cor6 = 0.61 + 0.66 + 0.62 + 0.63 + 0.73 + 0.60 + 0.74 + 0.53 + 0.72
   # 5.84
cor7 = 0.96 + 0.57 + 0.48 + 0.53 + 0.69 + 0.60 + 0.60 + 0.49 + 0.73
   # 5.65
cor8 = 0.62 + 0.52 + 0.48 + 0.56 + 0.66 + 0.74 + 0.60 + 0.84 + 0.69
   # 5.71
cor9 = 0.51 + 0.48 + 0.46 + 0.49 + 0.55 + 0.53 + 0.49 + 0.84 + 0.69
   # 5.04
cor10 = 0.73 + 0.84 + 0.70 + 0.72 + 0.88 + 0.72 + 0.73 + 0.69 + 0.62
   # 6.63
print(cor1, cor2, cor3, cor4, cor5, cor6, cor7, cor8, cor9, cor10)