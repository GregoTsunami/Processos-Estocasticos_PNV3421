import os
import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy as scp
from sklearn.linear_model import LinearRegression

###########################################################################################################################################################
def ex1_a7():
    data = {
        'Ano': [2001, 2001, 2001, 2001, 2002, 2002, 2002, 2002, 2003, 2003, 2003, 2003,
                2004, 2004, 2004, 2004, 2005, 2005, 2005, 2005, 2006, 2006, 2006, 2006,
                2007, 2007, 2007, 2007, 2008, 2008, 2008, 2008, 2009, 2009, 2009, 2009,
                2010, 2010, 2010, 2010, 2011, 2011, 2011, 2011, 2012, 2012, 2012, 2012,
                2013, 2013, 2013, 2013, 2014, 2014, 2014, 2014, 2015, 2015, 2015, 2015,
                2016, 2016, 2016, 2016, 2017, 2017],
        'Trimestre': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                    1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                    1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                    1, 2, 3, 4, 1, 2],
        'kWh': [1071, 648, 480, 746, 965, 661, 501, 768, 1065, 667, 486, 780,
                926, 618, 483, 757, 1047, 667, 495, 794, 1068, 625, 499, 850,
                975, 623, 496, 728, 933, 582, 490, 708, 953, 604, 508, 708,
                1036, 612, 503, 710, 952, 628, 534, 733, 1085, 692, 568, 783,
                928, 655, 590, 814, 1018, 670, 566, 811, 962, 647, 630, 803,
                1002, 887, 615, 828, 1003, 706]
    }

    # Criar DataFrame
    df = pd.DataFrame(data)

    # Criar variáveis dummy S2, S3, S4
    df['S2'] = (df['Trimestre'] == 2).astype(int)
    df['S3'] = (df['Trimestre'] == 3).astype(int)
    df['S4'] = (df['Trimestre'] == 4).astype(int)

    # Preparar dados para regressão
    X = df[['S2', 'S3', 'S4']]
    y = df['kWh']

    # Ajustar modelo de regressão linear
    model = LinearRegression()
    model.fit(X, y)

    # Coeficientes do modelo
    b0 = model.intercept_
    b2, b3, b4 = model.coef_

    print("=== MODELO DE REGRESSÃO ===")
    print(f"Equação: Y_hat = {b0:.6f} + {b2:.6f}*S2 + {b3:.6f}*S3 + {b4:.6f}*S4")
    print(f"\nCoeficientes:")
    print(f"b0 (intercepto) = {b0:.6f}")
    print(f"b2 = {b2:.6f}")
    print(f"b3 = {b3:.6f}")
    print(f"b4 = {b4:.6f}")

    # Previsões para 2017
    # 3º trimestre: S2=0, S3=1, S4=0
    pred_trim3 = b0 + b3
    # 4º trimestre: S2=0, S3=0, S4=0
    pred_trim4 = b0

    print("\n=== PREVISÕES PARA 2017 ===")
    print(f"3º Trimestre 2017: {pred_trim3:.1f} kWh")
    print(f"4º Trimestre 2017: {pred_trim4:.1f} kWh")

    # Calcular médias por trimestre para comparação
    print("\n=== MÉDIAS POR TRIMESTRE (VERIFICAÇÃO) ===")
    medias_trimestre = df.groupby('Trimestre')['kWh'].mean()
    print("Médias históricas por trimestre:")
    for trim, media in medias_trimestre.items():
        print(f"Trimestre {trim}: {media:.1f} kWh")

    print(f"\nComparação:")
    print(f"1º Trim (base): {b0:.1f}")
    print(f"2º Trim: {b0 + b2:.1f} (vs média histórica: {medias_trimestre[2]:.1f})")
    print(f"3º Trim: {b0 + b3:.1f} (vs média histórica: {medias_trimestre[3]:.1f})")
    print(f"4º Trim: {b0 + b4:.1f} (vs média histórica: {medias_trimestre[4]:.1f})")


###########################################################################################################################################################
if __name__ == "__main__":
    ex1_a7()
