# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 12:04:25 2025

@authors: taumanni ioannou
"""


# Foi feita uma consulta à base de dados da ANTAQ (https://web3.antaq.gov.br/ea/sense/download.html#pt), 
# e o histórico de 2024 a respeito dos tempos de atracação foi levantado para todos os portos brasileiros. 
# Os dados estão na planilha anexada a este enunciado.

# O exercício consiste em avaliar a estatística de tempo de fila (isto é, o tempo de espera para início da operação) e tempo de operação dos navios. 
# Foi designado para cada aluno(a) um porto e um ou mais berços. 

# Faça a análise de dados, limpando os dados quando necessário, identificando possíveis outliers e fornecendo as estatísticas básicas. 
# Também faça os testes de aderência e teste todas as distribuições estatísticas que foram abordadas em aula.

# Para identificar outliers, use como referência o arquivo anexado a este enunciado.


import pandas as pd
import scipy as scp
from scipy import stats
from scipy.stats import norm, expon, lognorm, gamma, weibull_min, skew, kurtosis
import numpy as np
import matplotlib.pyplot as plt
import os 
import time
from datetime import datetime

#%% Importando os dados
df = pd.read_excel('Base de Dados.xlsx', sheet_name='Planilha3', skiprows=1)

print(df.info())
print(df.describe())

df = df.rename(columns =
    {
        df.columns[9]: 'Data_Atracacao',
        df.columns[10]: 'Data_Chegada',
        df.columns[11]: 'Data_Desatracacao',
        df.columns[12]: 'Data_Inicio_Operacao',
        df.columns[13]: 'Data_Termino_Operacao'
    }
)

# Funcionamento:
# Chegada -> Espera -> Atraca -> Inicio Opera -> Opera -> Fim Opera -> Desatraca

# Calculo dos tempos (em horas)
df['Tempo_Espera'] = ( (df['Data_Inicio_Operacao'] - df['Data_Chegada']).dt.total_seconds() ) / 3600
df['Tempo_Operacao'] = ( (df['Data_Termino_Operacao'] - df['Data_Inicio_Operacao']).dt.total_seconds() ) / 3600
df['Tempo_Atracado'] = ( (df['Data_Desatracacao'] - df['Data_Atracacao']).dt.total_seconds() ) / 3600

dados = df[['Tempo_Espera', 'Tempo_Operacao', 'Tempo_Atracado']].dropna()


print("Dados de interesse")
print(dados.describe())

# Análise dos Outliers
def outliers(df, coluna):
    q1 = df[coluna].quantile(0.25)
    q3 = df[coluna].quantile(0.75)
    
    iqr = q3 - q1
    lim_inf = q1 - 1.5*iqr
    lim_sup = q3 + 1.5*iqr
    
    outlier = df[(df[coluna] < lim_inf) | (df[coluna] > lim_sup)]
    df_clear = df[(df[coluna] >= lim_inf) & (df[coluna] <= lim_sup)]
    
    return df_clear, outlier, (lim_inf, lim_sup)

dado_clear_espera, outlier_espera, lim_espera = outliers(df, 'Tempo_Espera')
dado_clear_operacao, outlier_operacao, lim_operacao = outliers(df, 'Tempo_Operacao')

print(50*"_")
print(f"Tempo de Espera - Limites do IQR: {lim_espera}")
print(f"Tempo de Operação - Limites do IQR: {lim_operacao}")
print(f"N° de outliers no Tempo de Espera: {len(outlier_espera)}")
print(f"N° de outliers no Tempo de Operação: {len(outlier_operacao)}")

remove_outlier = set(dado_clear_espera.index) & set(dado_clear_operacao.index)
dados_clear = df.loc[list(remove_outlier), ['Tempo_Espera', 'Tempo_Operacao']].dropna()

print(50*"_")
print(f"Dados originais: {len(df)}")
print(f"Dados sem outliers: {len(dados_clear)}")


# Estatísticas básicas (média, variância, desvio padrão...)
def estatisticas_basicas (serie, nome):
    media = serie.mean()
    media_aparada = stats.trim_mean(serie, 0.05)
    mediana = serie.median()
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    lim_sup = q1 - 1.5*iqr
    lim_inf = q3 + 1.5*iqr
    desvio = serie.std()
    variancia = serie.var()
    if media != 0:
        coef_var = (desvio / media) * 100  
    else:
        coef_var = np.nan
    n = serie.count()
    curtose = kurtosis(serie, fisher = False) # utilizado 'False' para curtose de Pearson
    assimetria = skew(serie)
    
    resultados = {
        'Estatísticas': ['Média',
                         'Média Aparada (5%)',
                         'Mediana',
                         'Q1',
                         'Q3',
                         'IQR (IIQ)',
                         'Limite Superior',
                         'Limite Inferior',
                         'Desvio Padrão',
                         'Variância',
                         'Coeficiente de Variância', 
                         'N',
                         'Curtose',
                         'Coef. Assimetria'],
        nome: [
            media,
            media_aparada,
            mediana,
            q1,
            q3,
            iqr,
            lim_sup,
            lim_inf,
            desvio,
            variancia,
            coef_var,
            n,
            curtose,
            assimetria
        ]    
    }
    
    return pd.DataFrame(resultados)

print(50*'_')
print("\n --- Estatísticas Básicas | Tempo de Espera ---")
estatisticas_espera = estatisticas_basicas(dados_clear['Tempo_Espera'], 'Tempo_Espera')
print(estatisticas_espera)

print("\n --- Estatísticas Básicas | Tempo de Operação ---")
estatisticas_operacao = estatisticas_basicas(dados_clear['Tempo_Operacao'], 'Tempo_Operacao')
print(estatisticas_operacao)

# Visualização dos dados


# Testes de Aderência


# Testes de distribuições estatísticas