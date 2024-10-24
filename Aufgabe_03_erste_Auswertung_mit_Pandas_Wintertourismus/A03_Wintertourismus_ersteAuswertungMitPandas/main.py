import numpy as py
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import describe
from tabulate import tabulate

df =pd.read_excel('data/Zeitreihe-Winter-2024011810.xlsx')

# A1.3
base = ['Bez', 'Gemnr', 'Gemeinde']
years = df.columns[3:].astype(str)
base.extend('x' + years)
df.columns = base
# Alle Daten von 2000 werden ausgegeben
print("Daten von 2000:")
print(df.x2000)

'''
print(df.describe())

Die Ausgabe von describe() sieht wie folgt aus:
Es zählt die Anzahl der Einträge pro Column, den Mittelwert, die Standardabweichung, das Minimum, jeweils das Quantil und das Maximum
              Gemnr         x2000  ...         x2022         x2023
count    277.000000  2.770000e+02  ...  2.770000e+02  2.770000e+02
mean   70572.003610  8.086864e+04  ...  7.548605e+04  9.280590e+04
std      235.223059  1.681780e+05  ...  1.620111e+05  1.971253e+05
min    70101.000000  0.000000e+00  ...  0.000000e+00  0.000000e+00
25%    70351.000000  5.772000e+03  ...  4.950000e+03  6.967000e+03
50%    70601.000000  2.118500e+04  ...  1.528900e+04  2.037400e+04
75%    70807.000000  7.673700e+04  ...  6.272300e+04  8.540600e+04
max    70941.000000  1.644428e+06  ...  1.643703e+06  1.976378e+06
'''

# als Tabelle ausgeben
#print(tabulate(df, headers=df.columns))


# A2.1
# Hier wird die Spalte "I" ausgewählt und die Werte in ein Array "i2" gespeichert. Und als Punktediagramm dargestellt
b_i = df[df.Bez == 'I']
i2 = b_i.values[0,3:]
print("Alle Daten von Innsbruck:")
print(i2)
plt.scatter(years, i2)
plt.xticks(rotation=90)
plt.show()


# A2.2
# Gleiche Ausgabe wie oben nur als Linien Diagramm und die Daten sind aus dem Bezirk Landeck, alle Gemeinden zusammengezählt
b_la = df[df.Bez == 'LA']
la = b_la.values[:,3:]
yearly_sum = la.sum(axis=0)
print("Summe Touristen pro Jahr im Bezirk Landeck:")
print(yearly_sum)
plt.plot(years, yearly_sum)
plt.xticks(rotation=90)
plt.xlabel('Jahr')
plt.ylabel('Summe Touristen')
plt.title('Summe Touristen pro Jahr im Bezirk Landeck')
plt.show()


# A3.1
df['Min'] = df.values[:, 3:].min(axis=1)
df['Max'] = df.values[:, 3:].max(axis=1)
df['Range'] = df['Max'] - df['Min']
df['Mean'] = df.values[:, 3:].mean(axis=1)
print(df[['Gemeinde', 'Min', 'Max', 'Range', 'Mean']])
