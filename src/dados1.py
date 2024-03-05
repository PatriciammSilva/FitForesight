import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import scikit-learn 

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

dados = pd.read_csv('/Users/patriciasilva/Desktop/Dados 1 - Experiência/dados1.csv')
   
   #cabeçalho da base de dados
print(dados.head())

 # dimensões da base de dados
print(dados.shape)

 # substituir valores NA pela mediana da variável
print(dados.isna().sum())
dados['age'] = dados['age'].fillna(dados['age'].median())
dados['height'] = dados['height'].fillna(dados['height'].median())
print(dados.isna().sum())
print(dados.head())
print(dados.shape)

   # algumas estatísticas descritivas dos dados
print(dados.describe())

   # descrição da variável categórica SIZE
   # 'object' tipo da variável SIZE
print(dados.describe(include='object'))
 
   # gráficos boxplot
sns.boxplot([dados['weight'], dados['age'], dados['height']])
print(plt.show())
plt.boxplot([dados['weight'], dados['age'], dados['height']])  
print(plt.show())   
                                         
   # os três boxplots apresentam outliers
 
   # identificar valores outliers na variável weight
Q1 = dados['weight'].quantile(0.25)
Q3 = dados['weight'].quantile(0.75)
IQR = Q3 - Q1
outw = dados[(dados['weight'] < Q1 - 1.5 * IQR) | (dados['weight'] > Q3 + 1.5 * IQR)]
print(outw)
print(outw.shape)
   # 3510 valores outliers

   # identificar valores outliers na variável age
Q1 = dados['age'].quantile(0.25)
Q3 = dados['age'].quantile(0.75)
IQR = Q3 - Q1
outa = dados[(dados['age'] < Q1 - 1.5 * IQR) | (dados['age'] > Q3 + 1.5 * IQR)]
print(outa)
print(outa.shape)
   # 6726 valores outliers

   # identificar valores outliers na variável height
Q1 = dados['height'].quantile(0.25)
Q3 = dados['height'].quantile(0.75)
IQR = Q3 - Q1
outh= dados[(dados['height'] < Q1 - 1.5 * IQR) | (dados['height'] > Q3 + 1.5 * IQR)]
print(outh)
print(outh.shape)
   # 185 valores outliers
 
   # gráficos da variável Size
#sns.countplot(dados['size'])
size1 = [9731, 21127, 28379, 16533, 17747, 63, 16025]
bars = ('XXS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL')
pos = np.arange(len(bars))
plt.bar(pos, size1)
print(plt.show())  
sns.countplot(dados['size'])
print(plt.show())  
plt.pie(size1)
print(plt.show()) 

   # kmeans com os valores outliers 
data = pd.get_dummies(dados,columns=['size'])     
data.head()
features = data.drop(['size_L','size_M','size_S','size_XL','size_XXL','size_XXS','size_XXXL'], axis=1)
features.head()