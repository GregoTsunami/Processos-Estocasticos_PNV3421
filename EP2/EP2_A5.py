import os
import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy as scp
from statsmodels.api import OLS
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

##########################################################################################################################################################
def suav_expon_simples(arq, pagina):

    # Preparação dos dados
    df = pd.read_excel(arq, sheet_name=pagina)

    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    df['valor'] = df['valor'].astype(str).str.replace(',', '.').astype(float)
    df = df.sort_values('data').set_index('data')

    # Suavização simples
    model = SimpleExpSmoothing(df['valor'])
    model_ajustado = model.fit(optimized=True)
    alpha_otimizado = model_ajustado.params['smoothing_level']

    df['suavizado'] = model_ajustado.fittedvalues

    # Gráficos
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['valor'], label='Original', linewidth=1, alpha=0.7)
    plt.plot(df.index, df['suavizado'], label=f'Suavizado (α={alpha_otimizado:.3f})', linewidth=2)
    plt.title('Valores Originais vs Suavizados')
    plt.xlabel('Data')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Resultados
    print("=" * 60)
    print("RESULTADOS DA SUAVIZAÇÃO EXPONENCIAL SIMPLES")
    print("=" * 60)
    print(f"Melhor alpha encontrado: {alpha_otimizado:.4f}")
    print(f"Número de observações: {len(df)}")
    print(f"Período analisado: {df.index[0].strftime('%d/%m/%Y')} a {df.index[-1].strftime('%d/%m/%Y')}")

    print("\n" + "=" * 60)
    print("VALORES ORIGINAIS vs SUAVIZADOS")
    print("=" * 60)
    resultados_print = df.reset_index()
    resultados_print['data'] = resultados_print['data'].dt.strftime('%d/%m/%Y')
    resultados_print['valor'] = resultados_print['valor'].round(2)
    resultados_print['suavizado'] = resultados_print['suavizado'].round(2)
    print("\nPrimeiras 10 observações:")
    print(resultados_print.head(10).to_string(index=False))
    print("\nÚltimas 10 observações:")
    print(resultados_print.tail(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("ESTATÍSTICAS RESUMIDAS")
    print("=" * 60)
    print(f"Valor médio original: {df['valor'].mean():.2f}")
    print(f"Valor médio suavizado: {df['suavizado'].mean():.2f}")
    print(f"Desvio padrão original: {df['valor'].std():.2f}")
    print(f"Desvio padrão suavizado: {df['suavizado'].std():.2f}")
    print(f"Variação original: {df['valor'].max() - df['valor'].min():.2f}")
    print(f"Variação suavizada: {df['suavizado'].max() - df['suavizado'].min():.2f}")

    previsao = model_ajustado.forecast(1)
    print(f"\nPrevisão para o próximo período: {previsao.iloc[0]:.2f}")

##########################################################################################################################################################
def suav_tendencia(arq, pagina):

    # Preparação dos dados
    df = pd.read_excel(arq, sheet_name=pagina)
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    df['valor'] = df['valor'].astype(str).str.replace(',', '.').astype(float)
    df = df.sort_values('data').set_index('data')

    # Suavização com tendência
    model_holt = Holt(df['valor'])
    model_holt_ajustado = model_holt.fit(optimized=True)
    alpha_otimizado = model_holt_ajustado.params['smoothing_level']
    beta_otimizado = model_holt_ajustado.params['smoothing_trend']

    df['suavizado_tendencia'] = model_holt_ajustado.fittedvalues

    # Gráficos
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['valor'], label='Original', linewidth=1, alpha=0.7)
    plt.plot(df.index, df['suavizado_tendencia'], label=f'Suavizado com Tendência (α={alpha_otimizado:.3f}, β={beta_otimizado:.3f})', linewidth=2)
    plt.title('Valores Originais vs Suavizados com Tendência')
    plt.xlabel('Data')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Resultados

    print("=" * 60)
    print("RESULTADOS DA SUAVIZAÇÃO EXPONENCIAL COM TENDÊNCIA")
    print("=" * 60)
    print(f"Melhor alpha encontrado: {alpha_otimizado:.4f}")
    print(f"Melhor beta encontrado: {beta_otimizado:.4f}")
    print(f"Número de observações: {len(df)}")
    print(f"Período analisado: {df.index[0].strftime('%d/%m/%Y')} a {df.index[-1].strftime('%d/%m/%Y')}")

    print("\n" + "=" * 60)
    print("VALORES ORIGINAIS vs SUAVIZADOS COM TENDÊNCIA")
    print("=" * 60)
    resultados_print = df.reset_index()
    resultados_print['data'] = resultados_print['data'].dt.strftime('%d/%m/%Y')
    resultados_print['valor'] = resultados_print['valor'].round(2)
    resultados_print['suavizado_tendencia'] = resultados_print['suavizado_tendencia'].round(2)
    print("\nPrimeiras 10 observações:")
    print(resultados_print.head(10).to_string(index=False))
    print("\nÚltimas 10 observações:")
    print(resultados_print.tail(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("ESTATÍSTICAS RESUMIDAS")
    print("=" * 60)
    print(f"Valor médio original: {df['valor'].mean():.2f}")
    print(f"Valor médio suavizado com tendência: {df['suavizado_tendencia'].mean():.2f}")
    print(f"Desvio padrão original: {df['valor'].std():.2f}")
    print(f"Desvio padrão suavizado com tendência: {df['suavizado_tendencia'].std():.2f}")
    print(f"Variação original: {df['valor'].max() - df['valor'].min():.2f}")
    print(f"Variação suavizada com tendência: {df['suavizado_tendencia'].max() - df['suavizado_tendencia'].min():.2f}")

    previsoes = 3
    previsao_holt = model_holt_ajustado.forecast(previsoes)
    print("\n" + "=" * 60)
    print(f"PREVISÃO PARA OS PRÓXIMOS {previsoes} PERÍODOS")
    print("=" * 60)
    ultima_data = df.index[-1]
    datas_futuras = pd.date_range(
        start=ultima_data + pd.DateOffset(months=1),
        periods=previsoes,
        freq='ME'
    )
    for i, (data, valor) in enumerate(zip(datas_futuras, previsao_holt), 1):
        print(f"{data.strftime('%d/%m/%Y')}: {valor:.4f}")
    
    print("\n" + "=" * 60)
    print("ANÁLISE DA TENDÊNCIA")
    print("=" * 60)
    tendencia_media = np.mean(np.diff(df['suavizado_tendencia'].dropna()))
    print(f"Tendência média identificada: {tendencia_media:.4f} por período")
    crescimento_total = ((df['valor'].iloc[-1] - df['valor'].iloc[0]) / df['valor'].iloc[0]) * 100
    print(f"Crescimento total no período: {crescimento_total:.2f}%")
    if len(df) >= 12:
        crescimento_12m = ((df['valor'].iloc[-1] - df['valor'].iloc[-12]) / df['valor'].iloc[-12]) * 100
        print(f"Crescimento nos últimos 12 meses: {crescimento_12m:.2f}%")

##########################################################################################################################################################
def suav_sazonalidade(arq, pagina):

    # Preparação dos dados
    df = pd.read_excel(arq, sheet_name=pagina)
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    df['valor'] = df['valor'].astype(str).str.replace(',', '.').astype(float)
    df = df.sort_values('data').set_index('data')

    # Suavização com tendência e sazonalidade
    model_holtwinters = ExponentialSmoothing(df['valor'], seasonal='add', seasonal_periods=12, trend='add')
    model_holtwinters_ajustado = model_holtwinters.fit(optimized=True)
    alpha_otimizado = model_holtwinters_ajustado.params['smoothing_level']
    beta_otimizado = model_holtwinters_ajustado.params['smoothing_trend']
    gamma_otimizado = model_holtwinters_ajustado.params['smoothing_seasonal']

    df['suavizado_sazonalidade'] = model_holtwinters_ajustado.fittedvalues

    # Gráficos
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['valor'], label='Original', linewidth=1, alpha=0.7)
    plt.plot(df.index, df['suavizado_sazonalidade'], label=f'Suavizado com Sazonalidade (α={alpha_otimizado:.3f}, β={beta_otimizado:.3f}, γ={gamma_otimizado:.3f})', linewidth=2)
    plt.title('Valores Originais vs Suavizados com Sazonalidade')
    plt.xlabel('Data')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Resultados
    print("=" * 60)
    print("RESULTADOS DA SUAVIZAÇÃO EXPONENCIAL COM SAZONALIDADE")
    print("=" * 60)
    print(f"Melhor alpha encontrado: {alpha_otimizado:.4f}")
    print(f"Melhor beta encontrado: {beta_otimizado:.4f}")
    print(f"Melhor gamma encontrado: {gamma_otimizado:.4f}")
    print(f"Número de observações: {len(df)}")
    print(f"Período analisado: {df.index[0].strftime('%d/%m/%Y')} a {df.index[-1].strftime('%d/%m/%Y')}")

    print("\n" + "=" * 60)
    print("VALORES ORIGINAIS vs SUAVIZADOS COM SAZONALIDADE")
    print("=" * 60)
    resultados_print = df.reset_index()
    resultados_print['data'] = resultados_print['data'].dt.strftime('%d/%m/%Y')
    resultados_print['valor'] = resultados_print['valor'].round(2)
    resultados_print['suavizado_sazonalidade'] = resultados_print['suavizado_sazonalidade'].round(2)
    print("\nPrimeiras 10 observações:")
    print(resultados_print.head(10).to_string(index=False))
    print("\nÚltimas 10 observações:")
    print(resultados_print.tail(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("ESTATÍSTICAS RESUMIDAS")
    print("=" * 60)
    print(f"Valor médio original: {df['valor'].mean():.2f}")
    print(f"Valor médio suavizado com sazonalidade: {df['suavizado_sazonalidade'].mean():.2f}")
    print(f"Desvio padrão original: {df['valor'].std():.2f}")
    print(f"Desvio padrão suavizado com sazonalidade: {df['suavizado_sazonalidade'].std():.2f}")
    print(f"Variação original: {df['valor'].max() - df['valor'].min():.2f}")
    print(f"Variação suavizada com sazonalidade: {df['suavizado_sazonalidade'].max() - df['suavizado_sazonalidade'].min():.2f}")

    previsoes = 12
    previsao_holtwinters = model_holtwinters_ajustado.forecast(previsoes)
    print("\n" + "=" * 60)
    print(f"PREVISÃO PARA OS PRÓXIMOS {previsoes} PERÍODOS")
    print("=" * 60)
    ultima_data = df.index[-1]
    datas_futuras = pd.date_range(
        start=ultima_data + pd.DateOffset(months=1),
        periods=previsoes,
        freq='ME'
    )
    for i, (data, valor) in enumerate(zip(datas_futuras, previsao_holtwinters), 1):
        print(f"{data.strftime('%d/%m/%Y')}: {valor:.4f}")

    print("\n" + "=" * 60)
    print("ANÁLISE DA SAZONALIDADE")
    print("=" * 60)
    tendencia_media = np.mean(np.diff(df['suavizado_sazonalidade'].dropna()))
    print(f"Tendência média identificada: {tendencia_media:.4f} por período")
    if hasattr(model_holtwinters_ajustado, 'season'):
        componente_sazonal = model_holtwinters_ajustado.season
        print("Componente sazonal médio por mês:")
        for i, valor in enumerate(componente_sazonal[:12]):
            print(f"  Mês {i+1}: {valor:.2f}")

##########################################################################################################################################################
if __name__ == "__main__":
    arq = "Aula 05 - Suavizamento.xlsx" 
    pagina1 = 'suav.simples'
    pagina2 = 'tendência'
    pagina3 = 'sazonalidade'

    suav_expon_simples(arq, pagina1)
    suav_tendencia(arq, pagina2)
    suav_sazonalidade(arq, pagina3)