## Cluster - 0.70


## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib


## Recuperar modelo e importar dataframe
kmeans = joblib.load('modkmeans4.pkl')
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets4/data4.csv')
df3dupla = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets4/df3dupla.csv')


## Normalizar dataset
scaler = StandardScaler()
dfnor = scaler.fit_transform(df)
df3duplanor = scaler.fit_transform(df3dupla)


## Cluster treinado
labels = kmeans.predict(dfnor)
print(labels)


## Previsão
labels3dupla = kmeans.predict(df3duplanor)
np.set_printoptions(threshold=np.inf)
print(labels3dupla)
cluster_counts = pd.Series(labels3dupla).value_counts()
print(cluster_counts)


## Verificação
verif = labels == labels3dupla
print(verif)  
acertos_counts = pd.Series(verif).value_counts()
print(acertos_counts)


## Percentagem de acertos / erros
acerto = 2892/3982
print(acerto)
   # 72.6 %
erro = 1090/3982
print(erro)
   # 27.4 %