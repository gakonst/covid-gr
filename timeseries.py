import pandas as pd
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
import numpy as np
import datetime

from bokeh.plotting import figure
from bokeh.io import show, output_notebook

names = ["city", "latitude", "longitude", "infections"]
loc = pd.read_csv(
    'cities.csv',
    names=names,
)

# load the data
names = ["date", "infections"]
ts = pd.read_csv(
    'greece.csv',
    names=names,
)
ts['date'] = pd.to_datetime(ts['date'], dayfirst=True)

# cumulative infections
ts['cumulative-infections'] = ts['infections'].cumsum()

# infections
infections_graph = figure(title = "Μολύνσεις", x_axis_label = "Date", y_axis_label = "Number infected", x_axis_type="datetime", width = 1200)

# Add a quad glyph
infections_graph.line(ts["date"], ts["infections"])

show(infections_graph)

# cumulative infections
total_infected = figure(title = "Μολύνσεις", x_axis_label = "Date", y_axis_label = "Number infected", x_axis_type="datetime", width = 1200)

# Add a quad glyph
total_infected.line(ts["date"], ts["cumulative-infections"])

show(total_infected)

# # convert indexes to dates
# ts.index = ts.index.map(lambda d: datetime.datetime.strptime(d.strip(), "%d.%m.%Y"))

# # Add elapsed days
# day_0 = ts.index[0]
# elapsed = ts.index.map(lambda date: (date- day_0).days)
# ts['elapsed-days'] = elapsed

# # Calculate cumulative infections
# infections = pd.Series(ts['infections'])
# cumulative = infections.cumsum()
# ts['cumulative-infections'] = cumulative # load it to the dataframe

# ## group all intra day infections together
# infections_per_day = pd.Series(ts.groupby(['date']).agg(sum)['infections'])

# # calculate the growth rate of the infections
# growth_rate = []
# for i in range(1, len(infections_per_day)):
    # diff = abs(infections_per_day[i] - infections_per_day[i-1])
    # growth = diff / infections_per_day[i-1]
    # growth_rate.append(growth)

# cumulative_per_day = infections_per_day.cumsum()
# elapsed = infections_per_day.index.map(lambda date: (date- day_0).days)

# # plot the actual data
# samples = 15

# # N = 500
# # x = np.linspace(0, 10, N)
# # y = np.linspace(0, 10, N)
# # xx, yy = np.meshgrid(x, y)
# # d = np.sin(xx)*np.cos(yy)

# # p = figure(tooltips=[("day since first infection", "$xdata"), ("infections", "$ydata"), ("value", "@image")])
# # p.x_range.range_padding = p.y_range.range_padding = 0

# # # must give a vector of image data for image parameter
# # p.image(image=[d], x=0, y=0, dw=10, dh=10, palette="Spectral11", level="image")
# # p.grid.grid_line_width = 0.5

# # output_file("image.html", title="image.py example")

# xdata = np.array(elapsed[:samples])
# ydata = np.array(cumulative_per_day[:samples])
# pylab.plot(xdata, ydata, 'o', label='trained')

# new_xdata = np.array(elapsed[samples:])
# new_ydata = np.array(cumulative_per_day[samples:])
# pylab.plot(new_xdata, new_ydata, 'x', label='new-data')

# # f(x) = \frac{L}{1 + e^{-k(x-x0)}}
# def sigmoid(x, L, x0, k):
    # y = L / (1 + np.exp(-k*(x-x0)))
    # return y

# def sigmoid_with_growth(growth):
    # return lambda x, L, x0: sigmoid(x, L, x0, growth)


# # fit for various growth rates
# print(cumulative_per_day)
# hypotheses = []
# growth_rates = [0.23, 0.235, 0.24]
# for g in growth_rates:
    # popt, pcov = curve_fit(sigmoid_with_growth(g), xdata, ydata, maxfev=10000)
    # hypotheses.append([*popt, g])

# # plot each scenario
# # generate x axis over 90 days since Feb 26th
# x = np.linspace(0, 90, 100)
# eps = 15
# for params in hypotheses:
    # L, x0, k = params
    # # get the 10%, 90% values
    # ten = L * 0.1
    # ninety = L * 0.9

    # # find the x values which produces them



    # midpoint_time = day_0 + datetime.timedelta(days=x0)
    # prediction = L / 2
    # y = sigmoid(x, *params)
    # print(f"{k}: Expecting to reach {prediction} by {midpoint_time} ({x0} after initial)")
    # pylab.plot(x,y, label=f'L={L}, x0 = {x0}, k = {k}')

# pylab.xlabel("Days since first incident") 
# pylab.xlim([0, 100])
# pylab.ylabel("Number tested positive")
# pylab.legend(loc='best')
# pylab.show()
