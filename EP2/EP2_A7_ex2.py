import os
import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy as scp
from sklearn.metrics import mean_squared_error

# Dados

df = pd.read_excel("Aula 07 - Ex2.xlsx", sheet_name="ex")

y = df['Preço (R$)']
x = df[["Área (m2)", "# Quartos", "# Banheiros", "Idade (anos)", 
        "Distância_centro (km)", "Garagem", "Índice_Escolas", 
        "Índice_Criminalidade", "Zona_Norte", "Zona_Sul", "Zona_Leste"]]

x = sm.add_constant(x)


# Processo aditivo de regressão linear multivariada
def processo_aditivo(data, target, significance_level=0.05):
    initial_features = data.columns.tolist()
    initial_features.remove("const")
    selected_features = ["const"]
    
    while len(initial_features) > 0:
        best_pvalue = 1
        best_feature = None
        
        for feature in initial_features:
            model = sm.OLS(target, data[selected_features + [feature]]).fit()
            p_value = model.pvalues[feature]
            
            if p_value < best_pvalue:
                best_pvalue = p_value
                best_feature = feature
        
        if best_pvalue < significance_level:
            selected_features.append(best_feature)
            initial_features.remove(best_feature)
        else:
            break  # Para se nenhuma variável melhora o modelo
    
    return selected_features

selected_vars = processo_aditivo(x, y)
print("\n")
print("="*50)
print("Variáveis selecionadas pelo processo aditivo:", selected_vars)
print("="*50)
print("\n")

# Ajuste do modelo final
final_model = sm.OLS(y, x[selected_vars]).fit()
print(final_model.summary())