## Correlação 


## Packages necessários
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets3/data3.csv')

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
cor11 = R.iloc[:, 10].sum()-1
cor12 = R.iloc[:, 11].sum()-1
cor13 = R.iloc[:, 12].sum()-1
cor14 = R.iloc[:, 13].sum()-1
cor15 = R.iloc[:, 14].sum()-1
cor16 = R.iloc[:, 15].sum()-1
cor17 = R.iloc[:, 16].sum()-1
cor18 = R.iloc[:, 17].sum()-1
cor19 = R.iloc[:, 18].sum()-1
cor20 = R.iloc[:, 19].sum()-1
cor21 = R.iloc[:, 20].sum()-1
cor22 = R.iloc[:, 21].sum()-1
cor23 = R.iloc[:, 22].sum()-1
cor24 = R.iloc[:, 23].sum()-1
cor25 = R.iloc[:, 24].sum()-1
print(cor1,cor2,cor3,cor4,cor5,cor6,cor7,cor8,cor9,cor10,cor11,cor12,cor13,cor14,cor15,cor16,cor17,cor18,cor19,cor20,cor21,cor22,cor23,cor24,cor25)
   # escolher as variáveis 7, 9,16