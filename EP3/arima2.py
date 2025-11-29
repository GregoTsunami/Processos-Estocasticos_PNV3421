import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.gofplots import qqplot
from scipy import stats

# Leitura do arquivo
df = pd.read_csv('Exercício2.txt', header=None, names=['valores'], decimal=',', skiprows=1, sep='\t')
data = df['valores'].values

series = pd.Series(data).astype(float)
series.index = pd.RangeIndex(start=0, stop=len(series), step=1)

# Análise da Série
plt.figure(figsize=(10,5))
plt.plot(data, marker='o')
plt.title("Série Temporal Original - Fechamento das Ações")
plt.xlabel("Dias")
plt.ylabel("Valor de Fechamento")
plt.grid(True)
plt.show()

# Estacionariedade
def make_stationary(series, max_d=3, signif=0.05):
    d = 0
    s = series.copy()
    adf_res = adfuller(s, autolag='AIC')
    print(f"ADF p-value (d={d}): {adf_res[1]:.6f}")
    while adf_res[1] > signif and d < max_d:
        s = s.diff().dropna()
        d += 1
        adf_res = adfuller(s, autolag='AIC')
        print(f"ADF p-value (d={d}): {adf_res[1]:.6f}")
    return s, d, adf_res

stationary_series, d, adf_res = make_stationary(series)

# p,q AIC
p_max = 5
q_max = 5
best_aic = np.inf
best_order = None
best_model = None

print("Buscando (p, q) com base no AIC...")
for p in range(p_max + 1):
    for q in range(q_max + 1):
        try:
            model = ARIMA(series, order=(p, d, q))
            result = model.fit()
            print(f"(p={p}, q={q}) → AIC = {result.aic:.2f}")

            if result.aic < best_aic:
                best_aic = result.aic
                best_order = (p, d, q)
                best_model = result

        except Exception as e:
            # Alguns valores de p,q podem falhar — ignore
            pass

print("\nMelhor modelo encontrado:")
print(f"ARIMA{best_order} com AIC = {best_aic:.2f}")

# Modelo
print("\nResumo do modelo ajustado:")
print(best_model.summary())

# Resíduos
resid = best_model.resid
plt.figure(figsize=(12,10))

# Resíduos no tempo
plt.subplot(4,1,1)
plt.plot(resid)
plt.title("Resíduos do Modelo ARIMA")
plt.xlabel("Tempo")

# ACF dos resíduos
plt.subplot(4,1,2)
plot_acf(resid, ax=plt.gca(), lags=20)
plt.title("ACF dos Resíduos")

# PACF dos resíduos
plt.subplot(4,1,3)
plot_pacf(resid, ax=plt.gca(), lags=20, method='ywm')
plt.title("PACF dos Resíduos")

# QQ-Plot
plt.subplot(4,1,4)
qqplot(resid, line='s', ax=plt.gca())
plt.title("QQ-Plot dos Resíduos")

plt.tight_layout()
plt.show()

# Histograma
plt.figure(figsize=(8,4))
plt.hist(resid, bins=20, density=True)
plt.title("Histograma dos Resíduos")
plt.xlabel("Resíduo")
plt.ylabel("Densidade")
plt.grid(alpha=0.3)
plt.show()

# Ljung-Box
lb_test = acorr_ljungbox(resid, lags=[10], return_df=True)
print("\nLjung-Box test (lag=10):")
print(lb_test)

# Normalidade
sh = stats.shapiro(resid)
print("\nTeste de Shapiro-Wilk (normalidade dos resíduos):")
print(f"estatística={sh.statistic:.4f}, p-value={sh.pvalue:.4f}")

# Previsões
steps = 5
forecast = best_model.get_forecast(steps=steps)

fc_mean = forecast.predicted_mean
fc_int = forecast.conf_int()

idx = np.arange(len(series), len(series) + steps)

resultado_forecast = pd.DataFrame({
    'forecast': fc_mean.values,
    'lower': fc_int.iloc[:,0].values,
    'upper': fc_int.iloc[:,1].values
}, index=idx)

print("\nPrevisão para os próximos 5 dias:")
print(resultado_forecast)

plt.figure(figsize=(10,5))
plt.plot(series.index, series, label='Observado')
plt.plot(idx, resultado_forecast['forecast'], marker='o', label='Previsão')
plt.fill_between(idx, resultado_forecast['lower'], resultado_forecast['upper'], alpha=0.3)
plt.axvline(len(series)-1, color='gray', linestyle='--')
plt.title(f"Previsão ARIMA{best_order} (5 passos)")
plt.legend()
plt.show()