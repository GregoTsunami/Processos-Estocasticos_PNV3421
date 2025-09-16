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
from scipy.stats import norm, expon, uniform, gamma, lognorm, weibull_min, beta, skew, kurtosis
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import time
from datetime import datetime

#%% Importando os dados
df = pd.read_excel('Base de Dados.xlsx', sheet_name='Tubarao', skiprows=1)

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

print(50*'_')
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


# Visualização dos dados
print(50*'_')
print("\n --- Estatísticas Básicas | Tempo de Espera ---")
estatisticas_espera = estatisticas_basicas(dados_clear['Tempo_Espera'], 'Tempo_Espera')
print(estatisticas_espera)

print("\n --- Estatísticas Básicas | Tempo de Operação ---")
estatisticas_operacao = estatisticas_basicas(dados_clear['Tempo_Operacao'], 'Tempo_Operacao')
print(estatisticas_operacao)


# Box Plot e Histogramas
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

sns.boxplot(data=df[['Tempo_Espera', 'Tempo_Operacao']], ax=axes[0,0])
axes[0,0].set_title('Boxplot dos Dados Originais')
axes[0,0].legend()

sns.boxplot(data=dados_clear, ax=axes[0,1])
axes[0,1].set_title('Boxplot dos Dados sem Outliers')
axes[0,1].legend()

dados_clear['Tempo_Espera'].hist(bins=30, ax=axes[1,0], alpha = 0.7, color = 'blue')
axes[1,0].axvline(lim_espera[0], color='r', linestyle='--', label='Limite Inferior')
axes[1,0].axvline(lim_espera[1], color='r', linestyle='--', label='Limite Superior')
axes[1,0].set_title("Histograma do Tempo de Espera (s/ outlier)")
axes[1,0].legend()

dados_clear['Tempo_Operacao'].hist(bins=30, ax=axes[1,1], alpha = 0.7, color = 'orange')
axes[1,1].axvline(lim_operacao[0], color='r', linestyle='--', label='Limite Inferior')
axes[1,1].axvline(lim_operacao[1], color='r', linestyle='--', label='Limite Superior')
axes[1,1].set_title("Histograma do Tempo de Operação (s/ outlier)")
axes[1,1].legend()

plt.tight_layout()
plt.show()


# Testes de Aderência e Distribuições
def teste_distribuicoes(dados, coluna):
    values = dados[coluna].dropna()
    
    distribuicoes = [
        ('Normal', norm),
        ('Exponencial', expon),
        ('Uniforme', uniform),
        ('Gama', gamma),
        ('Log-Normal', lognorm),
        ('Weibull', weibull_min),
        ('Beta', beta)
    ]
    
    results = []
    
    for nome, dist in distribuicoes:
        try:
            if nome == 'Beta':
                min_val = values.min()
                max_val = values.max()
                values_norm = (values - min_val) / (max_val - min_val)
                params = dist.fit(values_norm, floc = 0, fscale = 1)
                
                # Kolmogorov - Smirnov c normalização
                D, p_value = stats.kstest(values_norm, dist.name, args=params)
                
            else:
                params = dist.fit(values)
                
                # Kolmogorov - Smirnov
                D, p_value = stats.kstest(values, dist.name, args= params)
            
            results.append(
                {
                    'Distribuição': nome,
                    'Estatísticas KS': D,
                    'p_value': p_value
                }
            )
                
        except Exception as e:
            print(f'Erro no teste de {nome}: {e}')
            continue
        
    return pd.DataFrame(results)

print(50*'_')

print("\n Testes de Aderência | Tempo de Espera")
results_espera = teste_distribuicoes(dados_clear, 'Tempo_Espera')
print(results_espera)

print("\n Testes de Aderência | Tempo de Operação")
results_operacao = teste_distribuicoes(dados_clear, 'Tempo_Operacao')
print(results_operacao)


# Histograma dos testes
def plot_distribuicoes(dados, coluna):
    values = dados[coluna].dropna()
    plt.figure(figsize=(16,10))
    
    plt.hist(values, bins = 30, density = True, alpha = 0.6, color='g', label='Dados')
    
    x = np.linspace(values.min(), values.max(), 100)
    
    params_norm = norm.fit(values)
    plt.plot(x, norm.pdf(x, *params_norm), 'r-', label='Normal', linewidth=2)
    
    params_expon = expon.fit(values)
    plt.plot(x, expon.pdf(x, *params_expon), 'b-', label='Exponencial', linewidth=2)
    
    params_uniform = uniform.fit(values)
    plt.plot(x, uniform.pdf(x, *params_uniform), 'y-', label='Uniforme', linewidth=2)
    
    params_gamma = gamma.fit(values)
    plt.plot(x, gamma.pdf(x, *params_gamma), 'm-', label='Gama', linewidth=2)
    
    params_lognorm = lognorm.fit(values)
    plt.plot(x, lognorm.pdf(x, *params_lognorm), 'c-', label='Log-Normal', linewidth=2)
    
    params_weibull = weibull_min.fit(values)
    plt.plot(x, weibull_min.pdf(x, *params_weibull), 'k-', label='Weibull', linewidth=2)
    
    plt.title(f'Ajuste de Distribuições - {coluna}')
    plt.xlabel('Tempo (h)')
    plt.ylabel('Densidade')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
plot_distribuicoes(dados_clear, 'Tempo_Espera')
plot_distribuicoes(dados_clear, 'Tempo_Operacao')