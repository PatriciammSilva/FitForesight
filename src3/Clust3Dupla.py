## Cluster - 0.70 Dupla


## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib


## Recuperar modelo e importar dataframe
kmeans = joblib.load('modkmeans3.pkl')
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets3/data3.csv')
df2 = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets3/df2.csv')


## Normalizar dataset
scaler = StandardScaler()
dfnor = scaler.fit_transform(df)
df2nor = scaler.fit_transform(df2)


## Cluster treinado
labels = kmeans.predict(dfnor)
print(labels)


## Previsão
labels2 = kmeans.predict(df2nor)
np.set_printoptions(threshold=np.inf)
print(labels2)
cluster_counts = pd.Series(labels2).value_counts()
print(cluster_counts)


## Verificação
verif = labels == labels2
print(verif)  
acertos_counts = pd.Series(verif).value_counts()
print(acertos_counts)


## Percentagem de acertos / erros
acerto = 3009/3982
print(acerto)
   # 75.6 %
erro = 973/3982
print(erro)
   # 24.4 %