## Previsões Mod1 - 0.60
   # alterar o número do dataset no comando de importação


## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib


## Recuperar modelo e importar dataframe
kmeans = joblib.load('modkmeans.pkl')
df1 = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/df1.csv')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Passo 1: Criar um conjunto de dados de exemplo
X = np.array([[1, 2],
              [1, 4],
              [1, 0],
              [4, 2],
              [4, 4],
              [4, 0]])

# Passo 2: Treinar o modelo K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Passo 3: Plotar os clusters e os centroides
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')

# Passo 4: Inserir um novo ponto
new_point = np.array([[3, 2]])
plt.scatter(new_point[:, 0], new_point[:, 1], s=100, c='blue', label='Novo Ponto')

# Passo 5: Prever o cluster para o novo ponto (opcional)
predicted_cluster = kmeans.predict(new_point)
print(f'O novo ponto pertence ao cluster: {predicted_cluster[0]}')

# Adiciona a legenda e mostra o gráfico
plt.legend()
plt.show()
