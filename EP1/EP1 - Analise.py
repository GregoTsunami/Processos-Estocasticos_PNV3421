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
from scipy.stats import norm, expon, lognorm, gamma, weibull_min
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


# Estatísticas básicas (média, variância, desvio padrão...)


# Visualização dos dados


# Testes de Aderência


# Testes de distribuições estatísticas