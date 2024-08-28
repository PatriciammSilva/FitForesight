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

