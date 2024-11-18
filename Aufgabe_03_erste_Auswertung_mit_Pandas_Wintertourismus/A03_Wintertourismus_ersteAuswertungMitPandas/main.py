import numpy as py
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import describe
from tabulate import tabulate
import seaborn as sns

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
# 2020 bis 2022 war durch Corona ein starker Einsturz der Touristen in Innsbruck, bis es 2022 wieder gestiegen ist.
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
# Hier gleich wie in Innsbruck aufgrund von Corona ein sehr starker Absturz der Touristenanzahl 2020 bis 2022. Die Jahre davor sind die Touristenazahl jedoch stetig gewachsen
plt.show()


# A3.1
print("A3.1")
df['Min'] = df.values[:, 3:].min(axis=1)
df['Max'] = df.values[:, 3:].max(axis=1)
df['Range'] = df['Max'] - df['Min']
df['Mean'] = df.values[:, 3:].mean(axis=1)
# A3.1.1
# Hier wird der Range als Prozent von den Maximum angezeigt
df['Standardisierte Range'] = df['Range'] / df['Max'] *100
print(df[['Gemeinde', 'Min', 'Max', 'Range', 'Mean', 'Standardisierte Range']])


# A3.2
print("A3.2")
# Gesamtzahl an Touristen pro Jahr, die Touristenanzahl stieg stetig an bis 2020, wo es durch Corona einen starken Absturz gab und sich 2023 wieder erholte
print("Gesamtzahl pro Jahr")
tj = df.values[:,3:]
j_sum = tj.sum(axis=0)
for year, total in zip(years, j_sum):
    print(f"{year}: {total}")
# Gesamtzahl über alle Jahre
j2_sum = j_sum.sum()
print("Gesamtanzahl an Touristen über alle Jahre:")
print(j2_sum)
# Gesamtzahl nach Bezirken
bezirk_totals = df.groupby('Bez').sum().values[:, 2:]
total_tourists = bezirk_totals.sum(axis=1)
sorted_bezirke = sorted(zip(df['Bez'].unique(), total_tourists), key=lambda x: x[0])
for bezirk, total in sorted_bezirke:
    # Man sieht wie die Bezirke Landeck und Schwaz die meisten Touristen haben, da z.B. Landeck durch die viele Skigebiete wie Serfaus, Ischgel oder St. Anton sehr viele Touristen anzieht
    print(f"{bezirk}: {total}")

# A4
# A4.1
print("A4.1")
boxplot_data = df[['Bez', 'Standardisierte Range']]
plt.figure(figsize=(12, 8))
sns.boxplot(x='Bez', y='Standardisierte Range', data=boxplot_data, palette='Set3', hue='Bez', legend=False)
plt.xlabel('Bezirk')
plt.ylabel('Standardisierte Range (%)')
plt.title('Standardisierte Ranges der einzelnen Bezirke')
plt.xticks(rotation=90)
plt.show()

# A4.2
print("A4.2")
plt.figure(figsize=(12, 8))
sns.barplot(x=years, y=i2, palette='terrain', hue=years, legend=False)
plt.xticks(rotation=70)
plt.xlabel('Jahr')
plt.ylabel('Anzahl an Touristen')
plt.title('Anzahl an Touristen in Innsbruck pro Jahr')
# Es gab wie in ganz Tirol eine stetige Wachstum, bis 2020. 2019 waren es die meisten Touristen von 2000 bis 2023
#   Man kann also sagen, Wintertourismus in ganz Tirol wird immer beliebter
plt.show()


