import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_excel('data/bev_meld.xlsx')

print(df.columns)

bezirk = df.iloc[:, 0]
gemnr = df.iloc[:, 1]
gemeinde = df.iloc[:, 2]
jahre = df.iloc[:, 3:]

# A 2
# A 2.1
jahre_sum = df.iloc[:, 3:].sum(axis=0)

plt.figure(figsize=(10, 6))
jahre_sum.plot(kind='line')
plt.title('Summe der Bevölkerung pro Jahr')
plt.xlabel('Jahre')
plt.ylabel('Bevölkerung')
plt.grid(True)
plt.show()