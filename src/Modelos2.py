## Modelos - 0.65
   # alterar o número do dataset no comando de importação


## Packages necessários
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data4.csv')

## Modelo 1 
   # Selecionar variáveis 
X1 = df[['ACR.RADL_LNTH', 'CHEST_CIRC', 'SLEEVE.OUTSEAM_LNTH', 'WAIST_NAT_LNTH']]
y1 = df['WRIST_CIRC.STYLION']
   # Adicionar uma constante (termo de intercepção)
X1 = sm.add_constant(X1)
   # Ajustar o modelo de regressão linear múltipla
mod1 = sm.OLS(y1, X1).fit()
print(mod1.summary())
   # Fazer previsões
pred1 = mod1.predict(X1)
   # Gráfico
plt.scatter(df['ACR.RADL_LNTH'], df['WRIST_CIRC.STYLION'], color='blue', label='Dados')
plt.scatter(df['ACR.RADL_LNTH'], pred1, color='green', label='Previsões', alpha=0.5)
plt.xlabel('CACR.RADL_LNTH')
plt.ylabel('WRIST_CIRC.STYLION')
plt.title('Modelo 1')
plt.legend()
plt.show()


## Modelo 2 
   # Selecionar variáveis 
X2 = df[['ARMCIRCBCPS_FLEX', 'INTRSCY_DIST', 'SCYE_DEPTH']]  
y2 = df['NECK_CIRC.BASE']
   # Adicionar uma constante (termo de intercepção)
X2 = sm.add_constant(X2)
   # Ajustar o modelo de regressão linear múltipla
mod2 = sm.OLS(y2, X2).fit()
print(mod2.summary())
   # Fazer previsões
pred2 = mod2.predict(X2)
   # Gráfico
plt.scatter(df['INTRSCY_DIST'], df['NECK_CIRC.BASE'], color='blue', label='Dados')
plt.scatter(df['INTRSCY_DIST'], pred2, color='green', label='Previsões', alpha=0.5)
plt.xlabel('CACR.RADL_LNTH')
plt.ylabel('NECK_CIRC.BASE')
plt.title('Modelo 2')
plt.legend()
plt.show()


## Modelo 3
   # variável WstNatFront


## Definir variáveis
v1 = -18.741 - 0.058*df['ACR.RADL_LNTH'] + 0.068*df['CHEST_CIRC'] + 0.156*df['SLEEVE.OUTSEAM_LNTH'] + 0.117*df['WST_NAT_FRONT']
v2 = 53.338 + 0.498*df['ARMCIRCBCPS_FLEX'] + 0.227*df['INTRSCY_DIST'] + 0.420*df['SCYE_DEPTH']


## Criar dataset
data = {'V1': v1, 'V2': v2,'V3': df['WST_NAT_FRONT']}
df2 = pd.DataFrame(data)
print(df2)
df2.to_csv('df2.csv', index=False)