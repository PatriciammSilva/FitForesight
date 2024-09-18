## Distancia Euclideana - 0.70
   

## Packages necessários
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets4/data4.csv')
df3dupla = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets4/df3dupla.csv')


## Distância euclidianas
np.set_printoptions(threshold=np.inf)
if df.shape == df3dupla.shape:
    distances = np.linalg.norm(df.values - df3dupla.values, axis=1)
    print("Distâncias Euclidianas entre pontos correspondentes:")
    print(distances)
else:
    print("As bases de dados têm tamanhos diferentes. Não é possível calcular a distância.")

## Média das distâncias
mean_distance = np.mean(distances)
print("Média das Distâncias Euclidianas:", mean_distance)