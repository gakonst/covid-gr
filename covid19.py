import pandas as pd
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
import numpy as np
import datetime

# load the data
names = ["date", "infections", "age", "gender", "area", "source"]
df = pd.read_csv(
    'greece.csv',
    names=names,
    sep='|',
    keep_date_col=True,
    index_col=[0]) # index by date

# convert indexes to dates
df.index = df.index.map(lambda d: datetime.datetime.strptime(d.strip(), "%d.%m.%Y"))

# Add elapsed days
day_0 = df.index[0]
elapsed = df.index.map(lambda date: (date- day_0).days)
df['elapsed-days'] = elapsed

# Calculate cumulative infections
infections = pd.Series(df['infections'])
cumulative = infections.cumsum()
df['cumulative-infections'] = cumulative # load it to the dataframe

## group all intra day infections together
infections_per_day = pd.Series(df.groupby(['date']).agg(sum)['infections'])

# calculate the growth rate of the infections
growth_rate = []
for i in range(1, len(infections_per_day)):
    diff = abs(infections_per_day[i] - infections_per_day[i-1])
    growth = diff / infections_per_day[i-1]
    growth_rate.append(growth)

cumulative_per_day = infections_per_day.cumsum()
elapsed = infections_per_day.index.map(lambda date: (date- day_0).days)

# plot the actual data
xdata = np.array(elapsed)
ydata = np.array(cumulative_per_day)
pylab.plot(xdata, ydata, 'o', label='data')

# f(x) = \frac{L}{1 + e^{-k(x-x0)}}
def sigmoid(x, L, x0, k):
    y = L / (1 + np.exp(-k*(x-x0)))
    return y

def sigmoid_with_growth(growth):
    return lambda x, L, x0: sigmoid(x, L, x0, growth)


# fit for various growth rates
print(cumulative_per_day)
hypotheses = []
growth_rates = [0.22, 0.23, 0.24, 0.25]
for g in growth_rates:
    popt, pcov = curve_fit(sigmoid_with_growth(g), xdata, ydata, maxfev=1000)
    hypotheses.append([*popt, g])

# plot each scenario
# generate x axis over 90 days since Feb 26th
x = np.linspace(0, 90, 100)
eps = 15
for params in hypotheses:
    L, x0, k = params
    # get the 10%, 90% values
    ten = L * 0.1
    ninety = L * 0.9

    # find the x values which produces them



    midpoint_time = day_0 + datetime.timedelta(days=x0)
    prediction = L / 2
    y = sigmoid(x, *params)
    print(f"{k}: Expecting to reach {L} by {midpoint_time} ({x0} after initial)")
    pylab.plot(x,y, label=f'L={L}, x0 = {x0}, k = {k}')

pylab.xlabel("Days since first incident") 
pylab.xlim([0, 100])
pylab.ylabel("Number tested positive")
pylab.legend(loc='best')
pylab.show()
