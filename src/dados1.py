import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
#import yellowbrick


dados = pd.read_csv('/Users/patriciasilva/Desktop/Dados1/dados1.csv')
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
size1 = [9731, 21127, 28379, 16533, 17747, 63, 16025]
bars = ('XXS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL')
pos = np.arange(len(bars))
plt.bar(pos, size1)
print(plt.show())  
sns.countplot(dados['size'])
print(plt.show())  
plt.pie(size1)
print(plt.show()) 

   # kmeans  
   # criar dummys
#data = pd.get_dummies(dados,columns=['size'])  
#data = data.astype(int)
#print(data.head())
   # criar modelo
#features = data.drop(['size_L','size_M','size_S','size_XL','size_XXL','size_XXS','size_XXXL'], axis=1)
#print(features.head())
#from sklearn.preprocessing import StandardScaler
#scale = StandardScaler()     
#features = scale.fit_transform(features)
#features_scaled = pd.DataFrame( features, columns=['weight','age','height'])
#print(features_scaled.head())
#from sklearn.cluster import KMeans
#from yellowbrick.cluster import KElbowVisualizer
#model = KMeans()
#visualizer = KElbowVisualizer(model,k=(1,10),timings=False)
#visualizer.fit(features_scaled)
#visualizer.show()

   # matriz de correlação
#fig, ax = plt.subplots(figsize=(8,6))
#sns.heatmap(dados.corr(), annot=True, fmt='.1g', cmap="viridis",);

#from sklearn.datasets import load_iris
#rom sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA


# Padronizar os recursos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Criar um objeto PCA com 2 componentes
pca = PCA(n_components=2)

# Ajustar e transformar os dados
X_pca = pca.fit_transform(X_scaled)

# Plotar os resultados
plt.figure(figsize=(8, 6))
for i in range(len(iris.target_names)):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=iris.target_names[i])
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA - Análise de Componentes Principais')
plt.legend()
plt.show()


dados['size'] = dados['size'].map({'XXS': 1, 'S': 2, "M" : 3, "L" : \
   4, "XL" : 5, "XXL" : 6, "XXXL" : 7})
