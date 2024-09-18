## Cluster - 0.60


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
df1 = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets4/df1.csv')


## Normalizar dataset
scaler = StandardScaler()
dfnor = scaler.fit_transform(df)
df1nor = scaler.fit_transform(df1)


## Cluster treinado
labels = kmeans.predict(dfnor)
print(labels)


## Previsão
labels1 = kmeans.predict(df1nor)
np.set_printoptions(threshold=np.inf)
print(labels1)
cluster_counts = pd.Series(labels1).value_counts()
print(cluster_counts)


## Verificação
verif = labels == labels1
print(verif)  
acertos_counts = pd.Series(verif).value_counts()
print(acertos_counts)


## Percentagem de acertos / erros
acerto = 2887/3982
print(acerto)
   # 72.5 %
erro = 1095/3982
print(erro)
   # 27.5 %