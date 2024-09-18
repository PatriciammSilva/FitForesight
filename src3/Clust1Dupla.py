## Cluster - 0.60 Dupla


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
df1dupla = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets3/df1dupla.csv')


## Normalizar dataset
scaler = StandardScaler()
dfnor = scaler.fit_transform(df)
df1duplanor = scaler.fit_transform(df1dupla)


## Cluster treinado
labels = kmeans.predict(dfnor)
print(labels)


## Previsão
labels1dupla = kmeans.predict(df1duplanor)
np.set_printoptions(threshold=np.inf)
print(labels1dupla)
cluster_counts = pd.Series(labels1dupla).value_counts()
print(cluster_counts)


## Verificação
verif = labels == labels1dupla
print(verif)  
acertos_counts = pd.Series(verif).value_counts()
print(acertos_counts)


## Percentagem de acertos / erros
acerto = 3029/3982
print(acerto)
   # 76.1 %
erro = 953/3982
print(erro)
   # 23.9 %