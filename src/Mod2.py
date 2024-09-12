## Mod 2 - 0.65

   
## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data4.csv')


## Variable 1 : ACR.RADL_LNTH
   # Selecionar variáveis 
X1 = df[['NECK_CIRC.BASE', 'WRIST_CIRC.STYLION']]
y1 = df['ACR.RADL_LNTH']
X1 = sm.add_constant(X1)
   # Ajustar o modelo de regressão linear múltipla
mod1 = sm.OLS(y1, X1).fit()
print(mod1.summary())
   # Fazer previsões
v1 = mod1.predict(X1)
print(v1)


## Variable 2 : ARMCIRCBCPS_FLEX
   # Selecionar variáveis 
X2 = df[['NECK_CIRC.BASE', 'WRIST_CIRC.STYLION']]
y2 = df['ARMCIRCBCPS_FLEX']
X2 = sm.add_constant(X2)
   # Ajustar o modelo de regressão linear múltipla
mod2 = sm.OLS(y2, X2).fit()
print(mod2.summary())
   # Fazer previsões
v2 = mod2.predict(X2)
print(v2)


## Variable 3 : CHEST_CIRC
   # Selecionar variáveis 
X3 = df[['NECK_CIRC.BASE', 'WRIST_CIRC.STYLION']]
y3 = df['CHEST_CIRC']
X3 = sm.add_constant(X3)
   # Ajustar o modelo de regressão linear múltipla
mod3 = sm.OLS(y3, X3).fit()
print(mod3.summary())
   # Fazer previsões
v3 = mod3.predict(X3)
print(v3)


## Variable 4 : INTRSCY_DIST
   # Selecionar variáveis 
X4 = df[['NECK_CIRC.BASE', 'WRIST_CIRC.STYLION']]
y4 = df['INTRSCY_DIST']
X4 = sm.add_constant(X4)
   # Ajustar o modelo de regressão linear múltipla
mod4 = sm.OLS(y4, X4).fit()
print(mod4.summary())
   # Fazer previsões
v4 = mod4.predict(X4)
print(v4)


## Variable 6 : SCYE_DEPTH
   # Selecionar variáveis 
X6 = df[['NECK_CIRC.BASE', 'WRIST_CIRC.STYLION']]
y6 = df['SCYE_DEPTH']
X6 = sm.add_constant(X6)
   # Ajustar o modelo de regressão linear múltipla
mod6 = sm.OLS(y6, X6).fit()
print(mod6.summary())
   # Fazer previsões
v6 = mod6.predict(X6)
print(v6)


## Variable 7 : SLEEVE.OUTSEAM_LNTH
   # Selecionar variáveis 
X7 = df[['NECK_CIRC.BASE', 'WRIST_CIRC.STYLION']]
y7 = df['SLEEVE.OUTSEAM_LNTH']
X7 = sm.add_constant(X7)
   # Ajustar o modelo de regressão linear múltipla
mod7 = sm.OLS(y7, X7).fit()
print(mod7.summary())
   # Fazer previsões
v7 = mod7.predict(X7)
print(v7)


## Variable 8 : WAIST_NAT_LNTH
   # Selecionar variáveis 
X8 = df[['WST_NAT_FRONT']]
y8 = df['WAIST_NAT_LNTH']
X8 = sm.add_constant(X8)
   # Ajustar o modelo de regressão linear múltipla
mod8 = sm.OLS(y8, X8).fit()
print(mod8.summary())
   # Fazer previsões
v8 = mod8.predict(X8)
print(v8)



## Criar dataset
data = {'V1': v1, 'V2': v2,'V3': v3, 'V4': v4, 'V5': df['NECK_CIRC.BASE'], 'V6': v6, 'V7': v7, 'V8': v8,'V9': df['WST_NAT_FRONT'], 'V10': df['WRIST_CIRC.STYLION']}
df2 = pd.DataFrame(data)
print(df2)
df2.to_csv('df2.csv', index=False)