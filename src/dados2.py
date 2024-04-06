import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
 

dados = pd.read_csv('/Users/patriciasilva/Desktop/Dados2/dados2.csv')
 
   # cabeçalho base de dados
print(dados.head())
   # dimensões da base de dados
print(dados.shape) 