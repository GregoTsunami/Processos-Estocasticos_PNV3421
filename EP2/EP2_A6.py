import os
import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy as scp
import scipy.stats as stats
from sklearn.utils import resample

###########################################################################################################################################################
# Exercício 1 - A6

def ex_1_A6():
    
    # Dados
    cidades = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    vendas = [25, 16, 23, 15, 32, 25, 18, 18, 35, 34, 15, 32]
    aparicoes = [6, 2, 5, 1, 10, 7, 15, 3, 11, 13, 2, 12]

    # a) Cálculo da correlação de Pearson
    correlacao, p_valor = stats.pearsonr(vendas, aparicoes)

    print("="*50)
    print("a) ")
    print(f'Correlação de Pearson: {correlacao}')
    print(f'P-valor: {p_valor}')
    print("="*50)

    # b) Curva de regressão linear
    slope, intercept, r_value, p_value, std_err = stats.linregress(aparicoes, vendas)
    print("="*50)
    print("b) ")
    print(f"Equação: Vendas = {intercept:.4f} + {slope:.4f} * Aparições")
    print("="*50)

    # c) Erro padrão da estimativa
    vendas_previstas = [intercept + slope * x for x in aparicoes]
    sse = sum((y - y_prev) ** 2 for y, y_prev in zip(vendas, vendas_previstas))
    erro_padrao = np.sqrt(sse / (len(vendas) - 2))
    print("="*50)
    print("c) ")
    print(f"Erro padrão da estimativa: {erro_padrao:.4f}")
    print("="*50)

    # d) Coeficiente de determinação (R²)
    r_quadrado = r_value ** 2
    print("="*50)
    print("d) ")
    print(f"Coeficiente de determinação (R²): {r_quadrado:.4f} ou {r_quadrado * 100:.2f}%")
    print("="*50)

    # e) Significância do coeficiente angular
    n = len(vendas)
    t_calculado = slope / std_err
    t_critico = stats.t.ppf(1 - 0.01/2, n-2)
    print("="*50)
    print("e) ")
    print(f"Estatística t: {t_calculado:.4f}")
    print(f"Valor crítico t (α=1%): ±{t_critico:.4f}")
    print("Conclusão: Rejeita H0" if abs(t_calculado) > t_critico else "Não rejeita H0")
    print("="*50)
    
    # f) Previsão
    previsao_10 = intercept + slope * 10
    print("="*50)
    print("f) ")
    print(f"Previsão para 10 aparições: {previsao_10:.4f} (mil unidades)")
    print("="*50)

###########################################################################################################################################################
# Exercício 2 - A6

def ex_2_A6():

    # Dados
    data = {
        'X': [10, 10, 10, 20, 20, 30, 30, 30, 40, 40, 40, 50, 50, 50, 50, 60, 60, 
          70, 70, 70, 80, 80, 80, 90, 100, 100, 100, 110, 110, 110, 110, 120, 
          120, 120, 130, 130, 140, 150, 150, 150, 160, 160, 160, 160, 170, 170, 
          170, 180, 180, 180, 190, 190, 190, 200, 200, 200, 220, 220, 220, 220, 
          230, 230, 230, 230, 230, 240, 240, 240, 250, 250, 250, 260, 260, 260, 
          270, 270, 270, 280, 290, 290, 290, 290, 290, 300, 300, 300, 300, 310, 
          310, 310, 320, 320, 320, 320, 330, 340, 340, 340, 350, 350, 350, 350, 
          360, 360, 370, 370, 370, 380, 380, 390, 390, 390, 400, 400, 400, 400, 
          420, 420, 420, 420, 430, 430, 440, 440, 450, 450, 450, 460, 460, 470, 
          470, 470, 480, 480, 480, 480, 490, 490, 490, 500],

    'Y': [1.0, 0.9, 0.8, 1.3, 0.9, 0.6, 1.1, 1.0, 1.4, 1.4, 1.2, 1.7, 0.9, 1.2, 
          1.3, 0.7, 1.0, 1.3, 1.5, 2.0, 0.8, 0.6, 1.8, 1.0, 2.0, 0.5, 1.5, 1.3, 
          1.7, 1.2, 0.8, 1.0, 1.8, 2.1, 1.5, 1.9, 1.7, 1.2, 1.4, 2.1, 0.9, 1.1, 
          1.7, 2.0, 1.6, 1.9, 1.7, 2.2, 2.4, 1.6, 1.8, 4.1, 2.0, 1.5, 2.1, 2.5, 
          1.7, 2.0, 2.3, 1.8, 1.3, 1.6, 2.8, 2.2, 2.6, 1.4, 1.6, 1.7, 1.5, 2.2, 
          2.5, 2.4, 2.0, 2.7, 2.0, 2.2, 2.4, 1.8, 2.8, 2.2, 2.4, 2.1, 1.9, 2.4, 
          2.5, 2.9, 2.0, 1.9, 2.5, 2.6, 3.2, 2.8, 2.4, 2.5, 2.0, 2.4, 2.2, 2.0, 
          2.5, 2.8, 2.3, 2.7, 2.8, 3.1, 2.5, 2.9, 2.6, 3.0, 3.2, 2.9, 2.6, 2.5, 
          2.7, 3.1, 2.4, 3.0, 3.4, 3.5, 3.1, 2.9, 2.8, 3.3, 2.5, 2.8, 2.4, 2.6, 
          3.0, 3.4, 3.0, 3.3, 3.4, 3.1, 3.6, 3.0, 2.9, 3.2, 2.6, 3.8, 3.3, 2.9]
    }

    df = pd.DataFrame(data)

    def calcular_intervalo_previsao(X, Y, x0=305, alpha=0.05):

        n = len(X)
        
        # Regressão linear
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
        
        # Previsão para x0
        y_pred = intercept + slope * x0
        
        # Cálculos para intervalo de confiança
        y_previstos = intercept + slope * X
        sse = np.sum((Y - y_previstos) ** 2)
        se = np.sqrt(sse / (n - 2))
        
        x_mean = np.mean(X)
        s_xx = np.sum((X - x_mean) ** 2)
        
        # Erro padrão da previsão
        se_pred = se * np.sqrt(1 + 1/n + (x0 - x_mean)**2 / s_xx)
        
        # Valor t crítico
        t_critico = stats.t.ppf(1 - alpha/2, n-2)
        
        # Intervalo de confiança
        ic_inferior = y_pred - t_critico * se_pred
        ic_superior = y_pred + t_critico * se_pred
        amplitude = ic_superior - ic_inferior
        
        return {
            'tamanho_amostra': n,
            'previsao': y_pred,
            'ic_inferior': ic_inferior,
            'ic_superior': ic_superior,
            'amplitude': amplitude,
            'slope': slope,
            'intercept': intercept,
            'r_quadrado': r_value**2,
            'X_amostra': X.copy(),
            'Y_amostra': Y.copy()
        }

    def experimento_amostras(df, tamanhos_amostra, n_repeticoes=2, x0=305):
        resultados = []
        
        for tamanho in tamanhos_amostra:
            print(f"\n{'='*50}")
            print(f"TAMANHO DA AMOSTRA: {tamanho}")
            print(f"{'='*50}")
            
            for rep in range(n_repeticoes):
                # Amostra aleatória sem reposição
                amostra = df.sample(n=tamanho, random_state=rep*42)
                X_amostra = amostra['X'].values
                Y_amostra = amostra['Y'].values
                
                # Calcular intervalo de previsão
                resultado = calcular_intervalo_previsao(X_amostra, Y_amostra, x0)
                resultado['repeticao'] = rep + 1
                
                resultados.append(resultado)
                
                print(f"\nAmostra {rep + 1}:")
                print(f"  Previsão para X={x0}: {resultado['previsao']:.3f}")
                print(f"  Intervalo: [{resultado['ic_inferior']:.3f}, {resultado['ic_superior']:.3f}]")
                print(f"  Amplitude: {resultado['amplitude']:.3f}")
                print(f"  R²: {resultado['r_quadrado']:.4f}")
                print(f"  Equação: Y = {resultado['intercept']:.3f} + {resultado['slope']:.4f}*X")
        
        return resultados
    
    def visualizar_resultados(resultados, x0=305):
        df_resultados = pd.DataFrame(resultados)
        
        # Gráfico 1: Intervalos de previsão por tamanho de amostra
        plt.figure(figsize=(12, 8))
        cores = {20: 'red', 40: 'blue', 60: 'green'}
        
        for idx, row in df_resultados.iterrows():
            cor = cores[row['tamanho_amostra']]
            plt.plot([row['ic_inferior'], row['ic_superior']], 
                    [f"{row['tamanho_amostra']}-{row['repeticao']}"]*2, 
                    color=cor, linewidth=3, alpha=0.7, 
                    label=f"n={row['tamanho_amostra']}" if idx % 2 == 0 else "")
            plt.plot(row['previsao'], f"{row['tamanho_amostra']}-{row['repeticao']}", 
                    'o', color=cor, markersize=8)
        
        plt.axvline(x=df_resultados['previsao'].mean(), color='black', linestyle='--', 
                    label='Média das previsões')
        plt.xlabel('Valor de Y')
        plt.ylabel('Amostra (Tamanho-Repetição)')
        plt.title(f'Intervalos de Previsão para X={x0} por Tamanho de Amostra')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Gráfico 2: Amplitude dos intervalos
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_resultados, x='tamanho_amostra', y='amplitude')
        plt.title('Amplitude dos Intervalos de Previsão por Tamanho de Amostra')
        plt.xlabel('Tamanho da Amostra')
        plt.ylabel('Amplitude do Intervalo')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Gráfico 3: Dispersão e retas de regressão das amostras
        plt.figure(figsize=(12, 8))
        plt.scatter(df['X'], df['Y'], alpha=0.2, color='gray', s=20, 
                label='População completa', zorder=1)
        
        cores_linhas = {
            20: ['red', 'darkred'],
            40: ['blue', 'darkblue'], 
            60: ['green', 'darkgreen']
        }
        estilos = ['-', '--']

        for idx, row in df_resultados.iterrows():
            tamanho = row['tamanho_amostra']
            rep = row['repeticao']
            cor = cores_linhas[tamanho][rep-1]
            
            plt.scatter(row['X_amostra'], row['Y_amostra'], alpha=0.6, 
                    color=cor, s=40, zorder=2,
                    label=f'n={tamanho}, rep{rep}' if idx % 2 == 0 else "")
            
            x_min = min(row['X_amostra'])
            x_max = max(row['X_amostra'])
            x_range = np.linspace(x_min, x_max, 100)
            y_range = row['intercept'] + row['slope'] * x_range
            
            plt.plot(x_range, y_range, color=cor, linestyle=estilos[rep-1],
                    linewidth=2, alpha=0.8, zorder=3)
            
            previsao = row['intercept'] + row['slope'] * x0
            plt.plot(x0, previsao, 'o', color=cor, markersize=8, 
                    markeredgecolor='white', markeredgewidth=1, zorder=4)
        
        plt.axvline(x=x0, color='purple', linestyle=':', alpha=0.7,
                    label=f'X = {x0} (ponto de previsão)')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Dispersão dos Dados e Retas de Regressão das Amostras')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Resumo
        print("\n" + "="*60)
        print("RESUMO ESTATÍSTICO")
        print("="*60)
        
        resumo = df_resultados.groupby('tamanho_amostra').agg({
            'previsao': ['mean', 'std'],
            'amplitude': ['mean', 'std'],
            'r_quadrado': 'mean'
        }).round(4)
        
        print(resumo)
        
        return df_resultados

    print("="*60)
    print("COMPARAÇÃO DE INTERVALOS DE PREVISÃO")
    print("="*60)
    # Tamanhos de amostra a serem testados
    tamanhos_amostra = [20, 40, 60]

    # Executar experimento
    resultados = experimento_amostras(df, tamanhos_amostra, n_repeticoes=2, x0=305)

    # Visualizar resultados
    df_resultados = visualizar_resultados(resultados, x0=305)

    # Regressão com todos os dados
    print("\n" + "="*60)
    print("REGRESSÃO COM TODOS OS DADOS (População)")
    print("="*60)
    resultado_populacao = calcular_intervalo_previsao(df['X'].values, df['Y'].values, x0=305)
    print(f"Previsão populacional para X=305: {resultado_populacao['previsao']:.3f}")
    print(f"Intervalo populacional: [{resultado_populacao['ic_inferior']:.3f}, {resultado_populacao['ic_superior']:.3f}]")
    print(f"Amplitude populacional: {resultado_populacao['amplitude']:.3f}")
    print(f"R² populacional: {resultado_populacao['r_quadrado']:.4f}")


###########################################################################################################################################################
if __name__ == "__main__":
    ex_1_A6()
    ex_2_A6()