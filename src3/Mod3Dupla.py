## Mod 3 Dupla - 0.70

   
## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets3/data3.csv')


## Variable 1 : ACR.RADL_LNTH
   # Selecionar variáveis 
X1 = df[['SCYE_CIRC_OVER_ACROMION']]
y1 = df['ACR.RADL_LNTH']
X1 = sm.add_constant(X1)
   # Ajustar o modelo de regressão linear múltipla
mod1 = sm.OLS(y1, X1).fit()
print(mod1.summary())
   # Fazer previsões
v1 = mod1.predict(X1)
print(v1)


## Variable 2 : ARM_CIRC.AXILLARY
   # Selecionar variáveis 
X2 = df[['CHEST_CIRC_AT_SCYE', 'ELBOW_CIRC.EXTENDED', 'SCYE_CIRC_OVER_ACROMION']]
y2 = df['ARM_CIRC.AXILLARY']
X2 = sm.add_constant(X2)
   # Ajustar o modelo de regressão linear múltipla
mod2 = sm.OLS(y2, X2).fit()
print(mod2.summary())
   # Fazer previsões
v2 = mod2.predict(X2)
print(v2)


## Variable 3 : BIACROMIAL_BRTH  
   # Selecionar variáveis 
X3 = df[['CHEST_CIRC_AT_SCYE', 'ELBOW_CIRC.EXTENDED', 'SCYE_CIRC_OVER_ACROMION']]
y3 = df['BIACROMIAL_BRTH']
X3 = sm.add_constant(X3)
   # Ajustar o modelo de regressão linear múltipla
mod3 = sm.OLS(y3, X3).fit()
print(mod3.summary())
   # Fazer previsões
v3 = mod3.predict(X3)
print(v3)


## Variable 4 : ARMCIRCBCPS_FLEX  
   # Selecionar variáveis 
X4 = df[['CHEST_CIRC_AT_SCYE', 'ELBOW_CIRC.EXTENDED', 'SCYE_CIRC_OVER_ACROMION']]
y4 = df['ARMCIRCBCPS_FLEX']
X4 = sm.add_constant(X4)
   # Ajustar o modelo de regressão linear múltipla
mod4 = sm.OLS(y4, X4).fit()
print(mod4.summary())
   # Fazer previsões
v4 = mod4.predict(X4)
print(v4)


## Variable 5 : CHEST_BRTH   
   # Selecionar variáveis 
X5 = df[['CHEST_CIRC_AT_SCYE', 'ELBOW_CIRC.EXTENDED', 'SCYE_CIRC_OVER_ACROMION']]
y5 = df['CHEST_BRTH']
X5 = sm.add_constant(X5)
   # Ajustar o modelo de regressão linear múltipla
mod5 = sm.OLS(y5, X5).fit()
print(mod5.summary())
   # Fazer previsões
v5 = mod5.predict(X5)
print(v5)


## Variable 6 : CHEST_CIRC  
   # Selecionar variáveis 
X6 = df[['CHEST_CIRC_AT_SCYE', 'ELBOW_CIRC.EXTENDED', 'SCYE_CIRC_OVER_ACROMION']]
y6 = df['CHEST_CIRC']
X6 = sm.add_constant(X6)
   # Ajustar o modelo de regressão linear múltipla
mod6 = sm.OLS(y6, X6).fit()
print(mod6.summary())
   # Fazer previsões
v6 = mod6.predict(X6)
print(v6)


## Variable 8 : CHEST_CIRC.BELOW_BUST_  
   # Selecionar variáveis 
X8 = df[['CHEST_CIRC_AT_SCYE', 'ELBOW_CIRC.EXTENDED', 'SCYE_CIRC_OVER_ACROMION']]
y8 = df['CHEST_CIRC.BELOW_BUST_']
X8 = sm.add_constant(X8)
   # Ajustar o modelo de regressão linear múltipla
mod8 = sm.OLS(y8, X8).fit()
print(mod8.summary())
   # Fazer previsões
v8 = mod8.predict(X8)
print(v8)


## Variable 10 : FOREARM_CIRC.FLEXED   
   # Selecionar variáveis 
X10 = df[['CHEST_CIRC_AT_SCYE', 'ELBOW_CIRC.EXTENDED', 'SCYE_CIRC_OVER_ACROMION']]
y10 = df['FOREARM_CIRC.FLEXED']
X10 = sm.add_constant(X10)
   # Ajustar o modelo de regressão linear múltipla
mod10 = sm.OLS(y10, X10).fit()
print(mod10.summary())
   # Fazer previsões
v10 = mod10.predict(X10)
print(v10)


## Variable 11 : FOREARM.HAND_LENTH  
   # Selecionar variáveis 
X11 = df[['ELBOW_CIRC.EXTENDED']]
y11 = df['FOREARM.HAND_LENTH']
X11 = sm.add_constant(X11)
   # Ajustar o modelo de regressão linear múltipla
mod11 = sm.OLS(y11, X11).fit()
print(mod11.summary())
   # Fazer previsões
v11 = mod11.predict(X11)
print(v11)


## Variable 12 : INTRSCY_DIST  
   # Selecionar variáveis 
X12 = df[['CHEST_CIRC_AT_SCYE', 'ELBOW_CIRC.EXTENDED', 'SCYE_CIRC_OVER_ACROMION']]
y12 = df['INTRSCY_DIST']
X12 = sm.add_constant(X12)
   # Ajustar o modelo de regressão linear múltipla
mod12 = sm.OLS(y12, X12).fit()
print(mod12.summary())
   # Fazer previsões
v12 = mod12.predict(X12)
print(v12)


## Variable 13 : INTRSCY_MID_DIST   
   # Selecionar variáveis 
X13 = df[['CHEST_CIRC_AT_SCYE']]
y13 = df['INTRSCY_MID_DIST']
X13 = sm.add_constant(X13)
   # Ajustar o modelo de regressão linear múltipla
mod13 = sm.OLS(y13, X13).fit()
print(mod13.summary())
   # Fazer previsões
v13 = mod13.predict(X13)
print(v13)


## Variable 14 : NECK_CIRC.BASE  
   # Selecionar variáveis 
X14 = df[['CHEST_CIRC_AT_SCYE', 'ELBOW_CIRC.EXTENDED', 'SCYE_CIRC_OVER_ACROMION']]
y14 = df['NECK_CIRC.BASE']
X14 = sm.add_constant(X14)
   # Ajustar o modelo de regressão linear múltipla
mod14 = sm.OLS(y14, X14).fit()
print(mod14.summary())
   # Fazer previsões
v14 = mod14.predict(X14)
print(v14)


## Variable 17 : SCYE_DEPTH  
   # Selecionar variáveis 
X17 = df[['CHEST_CIRC_AT_SCYE', 'ELBOW_CIRC.EXTENDED', 'SCYE_CIRC_OVER_ACROMION']]
y17 = df['SCYE_DEPTH']
X17 = sm.add_constant(X17)
   # Ajustar o modelo de regressão linear múltipla
mod17 = sm.OLS(y17, X17).fit()
print(mod17.summary())
   # Fazer previsões
v17 = mod17.predict(X17)
print(v17)


## Variable 18 : SHOULDER_ELBOW_LNTH  
   # Selecionar variáveis 
X18 = df[['ELBOW_CIRC.EXTENDED', 'SCYE_CIRC_OVER_ACROMION']]
y18 = df['SHOULDER_ELBOW_LNTH']
X18 = sm.add_constant(X18)
   # Ajustar o modelo de regressão linear múltipla
mod18 = sm.OLS(y18, X18).fit()
print(mod18.summary())
   # Fazer previsões
v18 = mod18.predict(X18)
print(v18)


## Variable 19 : SPINE_TO_ELBOW_LNTH_.SL.  
   # Selecionar variáveis 
X19 = df[['CHEST_CIRC_AT_SCYE', 'ELBOW_CIRC.EXTENDED', 'SCYE_CIRC_OVER_ACROMION']]
y19 = df['SPINE_TO_ELBOW_LNTH_.SL.']
X19 = sm.add_constant(X19)
   # Ajustar o modelo de regressão linear múltipla
mod19 = sm.OLS(y19, X19).fit()
print(mod19.summary())
   # Fazer previsões
v19 = mod19.predict(X19)
print(v19)


## Variable 21 : WAIST_NAT_LNTH  
   # Selecionar variáveis 
X21 = v17
y21 = df['WAIST_NAT_LNTH']
X21 = sm.add_constant(X21)
   # Ajustar o modelo de regressão linear múltipla
mod21 = sm.OLS(y21, X21).fit()
print(mod21.summary())
   # Fazer previsões
v21 = mod21.predict(X21)
print(v21)


## Variable 22 : WAIST_OMPH_LNTH 
d22 = pd.DataFrame({'v17': v17, 'v21': v21})  
   # Selecionar variáveis 
X22 = d22[['v17', 'v21']]
y22 = df['WAIST_OMPH_LNTH']
X22 = sm.add_constant(X22)
   # Ajustar o modelo de regressão linear múltipla
mod22 = sm.OLS(y22, X22).fit()
print(mod22.summary())
   # Fazer previsões
v22 = mod22.predict(X22)
print(v22)


## Variable 23 : WST_NAT_FRONT
X23 = v21
y23 = df['WST_NAT_FRONT']
X23 = sm.add_constant(X23)
   # Ajustar o modelo de regressão linear múltipla
mod23 = sm.OLS(y23, X23).fit()
print(mod23.summary())
   # Fazer previsões
v23 = mod23.predict(X23)
print(v23)


## Variable 24 : WST_OMP_FRONT  
   # Selecionar variáveis 
X24 = v22
y24 = df['WST_OMP_FRONT']
X24 = sm.add_constant(X24)
   # Ajustar o modelo de regressão linear múltipla
mod24 = sm.OLS(y24, X24).fit()
print(mod24.summary())
   # Fazer previsões
v24 = mod24.predict(X24)
print(v24)


## Variable 25 : WRIST_CIRC.STYLION  
   # Selecionar variáveis 
X25 = df[['CHEST_CIRC_AT_SCYE', 'ELBOW_CIRC.EXTENDED', 'SCYE_CIRC_OVER_ACROMION']]
y25 = df['WRIST_CIRC.STYLION']
X25 = sm.add_constant(X25)
   # Ajustar o modelo de regressão linear múltipla
mod25 = sm.OLS(y25, X25).fit()
print(mod25.summary())
   # Fazer previsões
v25 = mod25.predict(X25)
print(v25)


## Variable 15 : RADIALE.STYLION_LNTH  
d15 = pd.DataFrame({'v1': v1, 'v11': v11, 'v18': v18, 'v19': v19, 'v25': v25})  
   # Selecionar variáveis 
X15 = d15[['v1', 'v11', 'v18', 'v19', 'v25']]
y15 = df['RADIALE.STYLION_LNTH']
X15 = sm.add_constant(X15)
   # Ajustar o modelo de regressão linear múltipla
mod15 = sm.OLS(y15, X15).fit()
print(mod15.summary())
   # Fazer previsões
v15 = mod15.predict(X15)
print(v15)


## Variable 20 : SLEEVE.OUTSEAM_LNTH  
d20 = pd.DataFrame({'v1': v1, 'v11': v11, 'v15': v15, 'v18': v18, 'v19': v19, 'v25': v25})  
   # Selecionar variáveis 
X20 = d20[['v1', 'v11', 'v15', 'v18', 'v19', 'v25']]
y20 = df['SLEEVE.OUTSEAM_LNTH']
X20 = sm.add_constant(X20)
   # Ajustar o modelo de regressão linear múltipla
mod20 = sm.OLS(y20, X20).fit()
print(mod20.summary())
   # Fazer previsões
v20 = mod20.predict(X20)
print(v20)



## Criar dataset
data = {'V1': v1, 'V2': v2, 'V3': v3, 'V4': v4, 'V5': v5, 'V6': v6, 'V7': df['CHEST_CIRC_AT_SCYE'], 'V8': v8, 'V9': df['ELBOW_CIRC.EXTENDED'], 'V10': v10, 'V11': v11, 'V12': v12,'V13': v13,'V14': v14,'V15': v15,'V16': df['SCYE_CIRC_OVER_ACROMION'],'V17': v17,'V18': v18,'V19': v19,'V20': v20,'V21': v21,'V22': v22,'V23': v23,'V24': v24,'V25': v25}
df3dupla = pd.DataFrame(data)
print(df3dupla)
df3dupla.to_csv('df3dupla.csv', index=False)