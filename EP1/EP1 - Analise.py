# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 12:04:25 2025

@authors: taumanni ioannou
"""


import pandas as pd
import scipy as scp
import numpy as np
import matplotlib.pyplot as plt
import os 
import time

#%% Importando os dados
df = pd.read_excel('Base de Dados.xlsx', sheet_name='Planilha3', skiprows=1)
