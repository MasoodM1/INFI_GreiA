import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_excel('data/bev_meld.xlsx')

# A 2
# A 2.1
bezirk = df['Bezirk']
gemnr = df['Gemnr']
gemeinde = df['Gemeinde']
years = df.loc[:, '1993':'2021']