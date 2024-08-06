## Data 1 - Estatísticas Descritivas

## Packages necessários
import pandas as pd

## Importação Dataset
data1 = pd.read_csv('/Users/patriciasilva/Desktop/Tese/FitForesight/Datasets/data1.csv')

## Estatísticas Descritivas
pd.set_option('display.max_columns', None)
print(data1.describe())
