## FA
   # alterar o número do dataset no comando de importação
   
## Packages necessários
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data1.csv')

## Normalizar dataset
scaler = StandardScaler()
dfnor = scaler.fit_transform(df)

## Aplicar FA
fa = FactorAnalysis(n_components=4, random_state=42)
fact = fa.fit_transform(dfnor)
print(fact)