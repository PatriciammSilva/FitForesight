## Previsões Mod3 - 0.70


## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib


## Recuperar modelo e importar dataframe
kmeans = joblib.load('modkmeans3.pkl')
df3 = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets3/df3.csv')

# NÃO HÁ DATASET