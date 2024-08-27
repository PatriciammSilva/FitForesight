## Previsões Modelos - 0.60
   # alterar o número do dataset no comando de importação


## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans


## Recuperar modelo e importar dataframe
kmeans = joblib.load('modkmeans.pkl')
df1 = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/df1.csv')


# Prever os clusters para os dados
y_kmeans = kmeans.predict(df1)

# Visualizar os resultados
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clusters Identificados pelo K-Means')
plt.show()
