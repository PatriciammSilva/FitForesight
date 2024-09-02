## Cluster - 0.70
   # alterar o número do dataset no comando de importação


## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib


## Recuperar modelo e importar dataframe
kmeans = joblib.load('modkmeans.pkl')
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data4.csv')
df3 = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/df3.csv')


## Normalizar dataset
scaler = StandardScaler()
dfnor = scaler.fit_transform(df)
df3nor = scaler.fit_transform(df3)


## Cluster treinado
labels = kmeans.predict(dfnor)
print(labels)


## Previsão
kmeans = joblib.load('modkmeans.pkl')
labels3 = kmeans.predict(df3nor)
np.set_printoptions(threshold=np.inf)
print(labels3)
cluster_counts = pd.Series(labels3).value_counts()
print(cluster_counts)


## Verificação
verif = labels == labels3
print(verif)  
acertos_counts = pd.Series(verif).value_counts()
print(acertos_counts)


## Percentagem de acertos / erros
acerto = 3061/3982
print(acerto)
   # 76.9 %
erro = 921/3982
print(erro)
   # 23.1 %