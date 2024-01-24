import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

df = pd.read_csv("combo_crime_data.csv")

df.dropna(); ## drop rows with missing data
# convert dates to pandas datetime format
#df.Date = pd.to_datetime(df.Date, format='%m/%d/%Y %I:%M:%S %p')
df.index = pd.DatetimeIndex(df.Date)

x = df.sample(30000) ##sampling a part of the dataset

x['Primary_Type'].value_counts().plot.bar()
plt.title("Crimes")
plt.show()

x_theft = x[x['Primary_Type'] == "THEFT"]
x_battery = x[x['Primary_Type'] == "BATTERY"]
x_cd = x[x['Primary_Type'] == "CRIMINAL DAMAGE"]
x_narc = x[(x['Primary_Type'] == "NARCOTICS")]

x_theft['Description'].value_counts(normalize=True).plot.bar()
plt.title("Theft Types")
plt.show()

x_battery['Description'].value_counts(normalize=True).plot.bar()
plt.title("Battery Types")
plt.show()

x_cd['Description'].value_counts(normalize=True).plot.bar()
plt.title("Criminal Damage Types")
plt.show()

x_narc['Description'].value_counts(normalize=True).plot.bar()
plt.title("Narcotics Types")
plt.show()

x['Arrest'].value_counts(normalize=True).plot.bar()
plt.title("Arrests")
plt.show()

x[x.Arrest == True]['Primary_Type'].value_counts(normalize=True).plot.bar()
plt.title("Arrests per crime type")
plt.show()

color = (0.2, 0.4, 0.6, 0.6)
x.groupby([x.index.month]).size().plot(kind='barh', color=color)
plt.ylabel('Month')
plt.xlabel('Number of crimes')
plt.title('Number of crimes by month of year')
plt.show()
