## Modelos - 0.60
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
X1 = df[['ACR.RADL_LNTH', 'SLEEVE.OUTSEAM_LNTH', 'WAIST_NAT_LNTH', 'WST_NAT_FRONT']]
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
X2 = df[['INTRSCY_DIST', 'SCYE_DEPTH']]  
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
   # Selecionar variáveis 
X3 = df[['CHEST_CIRC']]  
y3 = df['ARMCIRCBCPS_FLEX']
   # Adicionar uma constante (termo de intercepção)
X3 = sm.add_constant(X3)
   # Ajustar o modelo de regressão linear múltipla
mod3 = sm.OLS(y3, X3).fit()
print(mod3.summary())
   # Fazer previsões
pred3 = mod3.predict(X3)
   # Gráfico
plt.scatter(df['CHEST_CIRC'], df['ARMCIRCBCPS_FLEX'], color='blue', label='Dados')
plt.scatter(df['CHEST_CIRC'], pred3, color='green', label='Previsões', alpha=0.5)
plt.xlabel('CHEST_CIRC')
plt.ylabel('ARMCIRCBCPS_FLEX')
plt.title('Modelo 3')
plt.legend()
plt.show()


