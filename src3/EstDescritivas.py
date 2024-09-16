## Estatísticas Descritivas


## Packages necessários
import pandas as pd


## Importação Dataset
df = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets3/data3.csv')

## Estatísticas Descritivas
pd.set_option('display.max_columns', None)
print(df.describe())