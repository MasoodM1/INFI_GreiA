import numpy as np
from matplotlib import pyplot as plt

#source: https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data
d = np.genfromtxt('data/london_weather.csv', delimiter=",", skip_header=1)

dt =  d[:,0] #Datum mit folgendem Aufbau: 19790103 (3.Jänner 1979)
# Aufteilen in Tag, Monat, Jahr
day = (dt % 100).astype('i')
month = (dt % 10000 / 100).astype('i')
year = (dt % 100000000 / 10000).astype('i')


meantemp = d[:,5] # Durchschnittstemperatur in Grad Celsius

print('Temperaturdurchsnitt 1979, 1990, 2010, 2019',np.mean(meantemp[year == 1979]), np.mean(meantemp[year == 1990]), np.mean(meantemp[year == 2010]), np.mean(meantemp[year == 2019]))
plt.boxplot([meantemp[year == 1979], meantemp[year == 1990], meantemp[year == 2010], meantemp[year == 2019]])
plt.xticks([1,2,3,4], ["1979", "1990", "2010", "2019"]) # Für die X-Achse Beschriftung
plt.show()

plt.scatter(month[year == 1990], meantemp[year == 1990])
plt.show()

last10years = range(2010,2021)
for i in last10years:
    qlt = np.nanquantile(meantemp[year == i], [0.05, 0.95])
    print(f'{i} - 5% Quantil: {qlt[0]}, 95% Quantil: {qlt[1]}')

s = []
for i in range(2010,2021):
    s.append(np.nanmean(meantemp[year == i]))

plt.bar(last10years, s)
plt.xticks(last10years)
plt.show()

# Zusatzaufgabe: Es werden die Durchschnittstemperaturen der Jahre 2010-2020 für jeden Monat als Linien-Diagramm dargestellt.
years = range(2010,2021)
monthly_mean = {y: [np.nanmean(meantemp[(year == y) & (month == m)]) for m in range(1, 13)] for y in years}

for y in years:
    plt.plot(range(1, 13), monthly_mean[y], label=str(y))

plt.xlabel('Monat')
plt.ylabel('Durchschnittstemperatur (°C)')
plt.title('Monatliche Durchschnittstemperaturen (2010-2020)')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez'])
plt.show()