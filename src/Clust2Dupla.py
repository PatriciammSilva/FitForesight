## Cluster - 0.65


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
df2dupla = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/df2dupla.csv')


## Normalizar dataset
scaler = StandardScaler()
dfnor = scaler.fit_transform(df)
df2duplanor = scaler.fit_transform(df2dupla)


## Cluster treinado
labels = kmeans.predict(dfnor)
print(labels)


## Previsão
kmeans = joblib.load('modkmeans.pkl')
labels2dupla = kmeans.predict(df2duplanor)
np.set_printoptions(threshold=np.inf)
print(labels2dupla)
cluster_counts = pd.Series(labels2dupla).value_counts()
print(cluster_counts)


## Verificação
verif = labels == labels2dupla
print(verif)  
acertos_counts = pd.Series(verif).value_counts()
print(acertos_counts)


## Percentagem de acertos / erros
acerto = 2883/3982
print(acerto)
   # 72.4 %
erro = 1099/3982
print(erro)
   # 27.6 %