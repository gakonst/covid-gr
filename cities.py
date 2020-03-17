import pandas as pd
import matplotlib.pyplot as plt
import pylab
from scipy.optimize import curve_fit
import numpy as np
import datetime
import folium
from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar
from bokeh.palettes import brewer

# load the data
names = ["city", "latitude", "longitude", "infections"]
df = pd.read_csv(
    'cities.csv',
    names=names,
)


def infections_map():
    MAGNIFICATION = 100

    # map centered at Athens
    map = folium.Map(
        location=[37.9838, 23.7275],
    )

    for i, city in df.dropna().iterrows():
        print(city)
        # add a tooltip showing the number of infections on top
        popup = folium.Popup(str(city["infections"]), sticky=True, show=True)
        folium.Circle(
            location=[
                city["latitude"],
                city["longitude"]
            ],
            radius=city["infections"] * MAGNIFICATION,
            color="red",
            sticky=True,
            show=True,
            fill=True,  # Set fill to True
            fill_opacity=0.7
        ).add_child(popup).add_to(map)

    sw = df[['latitude', 'longitude']].min().values.tolist()
    ne = df[['latitude', 'longitude']].max().values.tolist()
    map.fit_bounds([sw, ne])
    return map

def infections_barchart():
    # get the numbers
    data = df.loc[:, ['city', 'infections']]
    data = data.sort_values(ascending=True, by=['infections'])

    p = figure(x_range=data['city'], title="Μολύνσεις ανά πόλη",
               x_axis_label="City", y_axis_label="Αριθμός Μολυσμένων", width=1200)

    p.vbar(x=data['city'], top=data['infections'], width=0.9)
    p.tools = []
    p.add_tools(HoverTool(tooltips=[("City", "@x"), ("Infected", "@top")]))

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.toolbar.logo = None
    p.toolbar_location = None

    return p


map = infections_map()
map.save('map.html')

folium.output_notebook()


barchart = infections_barchart()
show(barchart)
