## Distancia Euclideana - 0.60


## Packages necessários
import numpy as np
import pandas as pd
from scipy.spatial import distance


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data4.csv')
df1 = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/df1.csv')


## Observações a considerar 
x = df.iloc[0]
y = df1.iloc[0]


# Calcular Distância Euclidiana
dist = distance.euclidean(x, y)
print(dist)
