## Correlação 


## Packages necessários
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets4/data4.csv')

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
cor1 = R.iloc[:, 0].sum()-1
cor2 = R.iloc[:, 1].sum()-1
cor3 = R.iloc[:, 2].sum()-1
cor4 = R.iloc[:, 3].sum()-1
cor5 = R.iloc[:, 4].sum()-1
cor6 = R.iloc[:, 5].sum()-1
cor7 = R.iloc[:, 6].sum()-1
cor8 = R.iloc[:, 7].sum()-1
cor9 = R.iloc[:, 8].sum()-1
cor10 = R.iloc[:, 9].sum()-1
print(cor1, cor2, cor3, cor4, cor5, cor6, cor7, cor8, cor9, cor10)
   # escolhar as variáveis 2, 5, 10
   