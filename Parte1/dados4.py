import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import plotly.express as px
from pandas.plotting import scatter_matrix

## ALTERAR FICHEIRO DE TXT PARA CSV
with open('dados4.txt', 'r') as arquivo_txt:
    linhas = arquivo_txt.readlines()
dados_formatados = [linha.strip().split('\t') for linha in linhas]
with open('arquivo.csv', 'w', newline='') as arquivo_csv:
    escritor_csv = csv.writer(arquivo_csv)
    escritor_csv.writerows(dados_formatados)

## IMPORTAÇÃO DO DATASET
dados = pd.read_csv('/Users/patriciasilva/Desktop/Dados4/dados4.csv')
print(dados.shape)
   # 2208 observações e 166 variáveis 

## SELECIONAR VARIÁVEIS DA PARTE SUPERIOR DO CORPO
dados = dados.iloc[:,[1,4,7,10,11,32,33,34,35,47,
                   52,54,65,66,69,70,81,87,88,89,91,92,94,97,98,
                   99,100,109,110,111,112,113,115,116,
                   121,123,125]]
print(dados.shape)
   # 2208 observações e 37 variáveis
print(dados.columns.tolist())

## CORRELAÇÕES
   # matriz de correlações
corr_matrix = dados.corr()
print(corr_matrix)
   # gráfico de correlações com seaborn  
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap da Matriz de Correlação')
plt.show()
   # gráfico de correlações com plotly
fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Heatmap da Matriz de Correlação")
fig.show()
   # gráfico de correlações matplotlib
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(corr_matrix, cmap='coolwarm')
fig.colorbar(cax)
plt.title('Matriz de Correlação com Matplotlib', pad=20)
plt.show()
   # pares de variáveis
#sns.lmplot(x='', y='', data=dados)
#plt.title('Gráfico de Dispersão com Linha de Regressão')
#plt.show()

## EQUAÇÔES

