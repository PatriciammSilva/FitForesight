import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

data4 = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data4.csv')
print(data4.head())
print(data4.shape)
   
# Ajustar o modelo usando fórmulas
#model = sm.OLS(data4ACR.RADL_LNTH,data4SLEEVE.OUTSEAM_LNTH)
#results = model.fit()

np.random.seed(0)
Y = data4['ACR.RADL_LNTH']
X = data4['SLEEVE.OUTSEAM_LNTH']
# Adicionar uma constante para o modelo (intercepto)
X = sm.add_constant(X)
# Ajustar o modelo
model = sm.OLS(Y, X)
results = model.fit()
# Imprimir o resumo dos resultados
print(results.summary())
