## Estatísticas Descritivas
   # alterar o número do dataset no comando de importação

## Packages necessários
import pandas as pd


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data1.csv')

## Estatísticas Descritivas
pd.set_option('display.max_columns', None)
print(df.describe())
