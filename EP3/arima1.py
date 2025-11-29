import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

# Leitura do arquivo
df = pd.read_csv('Exercício1.txt', header=None, names=['valores'], decimal=',', skiprows=1, sep='\t')
data = df['valores'].values

# Análise da Série
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(data, marker='o', markersize=3)
plt.title('Série Temporal Original')
plt.grid(True)

# Estacionariedade
result = adfuller(data)
print(f'Teste ADF - Estatística: {result[0]:.4f}, p-valor: {result[1]:.4f}')

if result[1] > 0.05:
    print("Série NÃO estacionária - necessita diferenciação")
    d = 1
    diff_data = np.diff(data, n=1)
    result_diff = adfuller(diff_data)
    print(f'Teste ADF (1ª diferença) - Estatística: {result_diff[0]:.4f}, p-valor: {result_diff[1]:.4f}')
    
    if result_diff[1] > 0.05:
        print("Série ainda não estacionária - testando 2ª diferença")
        d = 2
        diff_data = np.diff(data, n=2)
        result_diff2 = adfuller(diff_data)
        print(f'Teste ADF (2ª diferença) - Estatística: {result_diff2[0]:.4f}, p-valor: {result_diff2[1]:.4f}')
    else:
        print("Série estacionária após 1ª diferença")
else:
    print("Série estacionária")
    d = 0
    diff_data = data

diff_data = np.diff(data, n=1)
result_diff = adfuller(diff_data)
print(f'Teste ADF (1ª diferença) - Estatística: {result_diff[0]:.4f}, p-valor: {result_diff[1]:.4f}')

# Parâmetros [p] e [q] por ACF e PACF
plt.subplot(3, 1, 2)
plot_acf(diff_data, ax=plt.gca(), lags=20, title=f'ACF - Série {"Original" if d == 0 else "Diferenciada"}')
plt.grid(True)

plt.subplot(3, 1, 3)
plot_pacf(diff_data, ax=plt.gca(), lags=20, title=f'PACF - Série {"Original" if d == 0 else "Diferenciada"}', method='ywm')
plt.grid(True)

plt.tight_layout()
plt.show()

# Testes de ARIMA por ACF e PACF
models = [(1, d, 1), (1, d, 0), (0, d, 1), (2, d, 1), (1, d, 2), (2, d, 2)]
best_aic = np.inf
best_model = None
best_order = None

print("Comparação de modelos ARIMA:")
print("Ordem (p,d,q) | AIC | BIC")

for order in models:
    try:
        model = ARIMA(data, order=order)
        model_fit = model.fit()
        aic_value = model_fit.aic
        bic_value = model_fit.bic
        print(f"{order} | {aic_value:.2f} | {bic_value:.2f}")
        
        if aic_value < best_aic:
            best_aic = aic_value
            best_model = model_fit
            best_order = order
    except Exception as e:
        print(f"{order} | Falha no ajuste: {e}")

print(f"\n Melhor modelo: ARIMA{best_order} com AIC = {best_aic:.2f}")

# Análise de Resíduos

residuals = best_model.resid

plt.figure(figsize=(15, 10))

# Residuo X Tempo
plt.subplot(2, 3, 1)
plt.plot(residuals)
plt.title('Resíduos do Modelo')
plt.axhline(y=0, color='r', linestyle='--')
plt.grid(True)

# Histograma
plt.subplot(2, 3, 2)
plt.hist(residuals, bins=15, density=True, alpha=0.7)
plt.title('Distribuição dos Resíduos')
plt.grid(True)

# ACF
plt.subplot(2, 3, 3)
plot_acf(residuals, ax=plt.gca(), lags=20, title='ACF dos Resíduos')
plt.grid(True)

# QQ-Plot
plt.subplot(2, 3, 4)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot dos Resíduos')

# PACF
plt.subplot(2, 3, 5)
plot_pacf(residuals, ax=plt.gca(), lags=20, title='PACF dos Resíduos', method='ywm')
plt.grid(True)

plt.tight_layout()
plt.show()

# Ljung-Box para Resíduos
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print("\nTeste de Ljung-Box para resíduos:")
print("H0: Não há autocorrelação nos resíduos (ruído branco)")
print(f"p-valor: {lb_test['lb_pvalue'].iloc[0]:.4f}")

if lb_test['lb_pvalue'].iloc[0] > 0.05:
    print("✓ Resíduos são ruído branco (não rejeitamos H0)")
else:
    print("✗ Resíduos mostram autocorrelação (rejeitamos H0)")

# Normalidade para Resíduos
_, pval_normality = stats.normaltest(residuals)
print(f"\nTeste de normalidade dos resíduos - p-valor: {pval_normality:.4f}")
if pval_normality > 0.05:
    print("✓ Resíduos seguem distribuição normal")
else:
    print("✗ Resíduos não seguem distribuição normal")

# Previsões ARIMA
forecast_steps = 7
forecast = best_model.forecast(steps=forecast_steps)
forecast_conf_int = best_model.get_forecast(steps=forecast_steps).conf_int()

print(f"Previsões ARIMA{best_order} para as próximas {forecast_steps} medições:")
for i in range(forecast_steps):
    lower = forecast_conf_int[i, 0]
    upper = forecast_conf_int[i, 1]
    print(f"  Mediçao {i+1}: {forecast[i]:.4f} (IC 95%: [{lower:.4f}, {upper:.4f}])")
  
plt.figure(figsize=(12, 6))

# Dados Historicos
plt.plot(np.arange(len(data)), data, 'b-', label='Dados Históricos', marker='o', markersize=3)

# Previsões
forecast_index = np.arange(len(data), len(data) + forecast_steps)
plt.plot(forecast_index, forecast, 'ro-', label='Previsões', markersize=5)

# Intervalo de confiança
plt.fill_between(forecast_index, 
                forecast_conf_int[:, 0], 
                forecast_conf_int[:, 1], 
                color='pink', alpha=0.3, label='Intervalo de Confiança 95%')

plt.axvline(x=len(data)-1, color='gray', linestyle='--', alpha=0.7)
plt.legend()
plt.title(f'Previsões ARIMA{best_order} - Processo de Controle de Qualidade')
plt.grid(True)
plt.xlabel('Tempo')
plt.ylabel('Valores')
plt.show()

print(f"Modelo selecionado: ARIMA{best_order}")
print(f"Critério de seleção: AIC = {best_aic:.2f}")
print(f"Estacionariedade: {d} diferenciações necessárias")
print("Diagnóstico dos resíduos:")
print(f"  - Teste Ljung-Box (ruído branco): {'✓ Aprovado' if lb_test['lb_pvalue'].iloc[0] > 0.05 else '✗ Reprovado'}")
print(f"  - Normalidade dos resíduos: {'✓ Aprovado' if pval_normality > 0.05 else '✗ Reprovado'}")
print("\nModelo válido para previsão!")