import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import plotly.express as px

## ALTERAR FICHEIRO DE TXT PARA CSV
with open('dados4.txt', 'r') as arquivo_txt:
    linhas = arquivo_txt.readlines()
dados_formatados = [linha.strip().split('\t') for linha in linhas]
with open('arquivo.csv', 'w', newline='') as arquivo_csv:
    escritor_csv = csv.writer(arquivo_csv)
    escritor_csv.writerows(dados_formatados)

## IMPORTAÇÃO DO DATASET
dados = pd.read_csv('/Users/patriciasilva/Desktop/Dados4/dados4.csv')
print(dados.shape)
   # 2208 observações e 166 variáveis 

## Seleção as variáveis da parte superior do corpo
#data = dados[['AB.EXT.DEPTH.SIT', 'ACROMION_HT', 'ACR_HT.SIT', 'ACR.RADL_LNTH', 'AXILLA_HT', 'ARM_CIRC.AXILLARY', 
#        'BIACROMIAL_BRTH', 'ARMCIRCBCPS_FLEX', 'BIDELTOID_BRTH', 'CERVIC_HT', 'CERVIC_HT_SITTING', 'CHEST_BRTH',
#        'CHEST_CIRC', 'CHEST_CIRC_AT_SCYE', 'CHEST_CIRC.BELOW_BUST_', 'CHEST_DEPTH', 'CHEST_HT', 'ELBOW_CIRC.EXTENDED',
#        'ELBOW_REST_HT', 'FOREARM_CIRC.FLEXED', 'FOREARM_TO_FOREARM_BRTH', 'FOREARM.HAND_LENTH', 'HIP_BRTH','HIP_BRTH_SITTING',
#        'ILIOCRISTALE_HT', 'INTRSCY_DIST', 'INTRSCY_MID_DIST', 'MIDSHOULDER_HT.SITTING', 'SHOULDER_CIRC', 'SHOULDER_ELBOW_LNTH',
#        'SHOULDER_LNTH', 'SPINE_TO_ELBOW_LNTH_.SL.', 'SPINE_TO_SCYE_LNTH_.SL.', 'SPINE_TO_WRIST_LNTH_.SL.', 
#        'SLEEVE.OUTSEAM_LNTH', 'SPAN', 'STATURE', 'STRAP_LNTH', 'SUPRASTERNALE_HT', 'TENTH_RIB', 'VERTICAL_TRUNK_CIRC', 
#        'WAIST_NAT_LNTH', 'WAIST_OMPH_LNTH', 'WAIST_BRTH_OMPHALION', 'WAIST_CIRC_NATURAL', 'WAIST_CIRC.OMPHALION',            
#        'WAIST_DEPTH.OMPHALION', 'WST_NAT_FRONT', 'WST_OMP_FRONT', 'WAIST_HT_NATURAL', 'WAIST_HT.OMPHALION', 
 #       'WAIST_HT_SIT_NATURAL', 'WAIST_HT.UMBILICUS.SITTING', 'WAIST_HIP_LNTH', 'WAIST_NATURAL_TO_WAIST_UMBILICUS']] 
 #print(data.shape)

#dados = dados.iloc[:,['AB-EXT-DEPTH-SIT', 'ACROMION_HT', 'ACR_HT-SIT', 'ACR-RADL_LNTH', 'AXILLA_HT', 'ARM_CIRC-AXILLARY', 
#        'BIACROMIAL_BRTH', 'ARMCIRCBCPS_FLEX', 'BIDELTOID_BRTH', 'CERVIC_HT', 'CERVIC_HT_SITTING', 'CHEST_BRTH',
#        'CHEST_CIRC', 'CHEST_CIRC_AT_SCYE', 'CHEST_CIRC-BELOW_BUST_', 'CHEST_DEPTH', 'CHEST_HT', 'ELBOW_CIRC-EXTENDED',
#        'ELBOW_REST_HT', 'FOREARM_CIRC-FLEXED', 'FOREARM_TO_FOREARM_BRTH', 'FOREARM-HAND_LENTH', 'HIP_BRTH','HIP_BRTH_SITTING',
#        'ILIOCRISTALE_HT', 'INTRSCY_DIST', 'INTRSCY_MID_DIST', 'MIDSHOULDER_HT-SITTING', 'SHOULDER_CIRC', 'SHOULDER_ELBOW_LNTH',
#        'SHOULDER_LNTH', 'SPINE_TO_ELBOW_LNTH_-SL-', 'SPINE_TO_SCYE_LNTH_-SL-', 'SPINE_TO_WRIST_LNTH_-SL-', 
#        'SLEEVE-OUTSEAM_LNTH', 'SPAN', 'STATURE', 'STRAP_LNTH', 'SUPRASTERNALE_HT', 'TENTH_RIB', 'VERTICAL_TRUNK_CIRC', 
#        'WAIST_NAT_LNTH', 'WAIST_OMPH_LNTH', 'WAIST_BRTH_OMPHALION', 'WAIST_CIRC_NATURAL', 'WAIST_CIRC-OMPHALION',            
#        'WAIST_DEPTH-OMPHALION', 'WST_NAT_FRONT', 'WST_OMP_FRONT', 'WAIST_HT_NATURAL', 'WAIST_HT-OMPHALION', 
#        'WAIST_HT_SIT_NATURAL', 'WAIST_HT-UMBILICUS-SITTING', 'WAIST_HIP_LNTH', 'WAIST_NATURAL_TO_WAIST_UMBILICUS']] 
#print(dados.shape)

## SELECIONAR VARIÁVEIS DA PARTE SUPERIOR DO CORPO
dados = dados.iloc[:,[1,2,3,4,6,7,10,11,12,30,31,32,33,34,35,36,37,47,48,
                   52,53,54,65,66,67,69,70,78,90,91,92,94,95,96,97,98,
                   99,100,101,102,108,109,110,111,112,113,114,115,116,
                   117,118,119,120,121,122]]
print(dados.shape)

## CORRELAÇÕES
   # matriz de correlações
corr_matrix = dados.corr()
print(corr_matrix)
   # gráfico de correlações com seaborn  
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap da Matriz de Correlação')
plt.show()
   # gráfico de correlações com plotly
fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Heatmap da Matriz de Correlação")
fig.show()
   # matriz de dispersão com seaborn
sns.pairplot(dados)
plt.suptitle('Matriz de Dispersão (Pairplot)', y=1.02)
plt.show()

