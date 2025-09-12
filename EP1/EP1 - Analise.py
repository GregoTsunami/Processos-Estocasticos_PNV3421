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
import numpy as np
import matplotlib.pyplot as plt
import os 
import time
from datetime import datetime

#%% Importando os dados
df = pd.read_excel('Base de Dados.xlsx', sheet_name='Planilha3', skiprows=1)
