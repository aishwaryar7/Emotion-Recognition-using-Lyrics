# cd C:\Users\aishw\Anaconda3\Scripts
# bokeh serve --show C:\Users\aishw\PycharmProjects\Project\Pie_chart.py

import pandas as pd
from numpy import pi
import csv

from bokeh.io import show, output_file, curdoc
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.plotting import figure
from bokeh.layouts import column, widgetbox
from bokeh.models.widgets import TextInput


p = figure(x_range=(-1, 1), y_range=(-1, 1), tools="", toolbar_location=None, width=500, height=400)
p.axis.visible = False
p.grid.visible = False


def update_title(attrname, old, new):
    p = figure(x_range=(-1, 1), y_range=(-1, 1), tools="", toolbar_location=None, width=500, height=400)
    p.axis.visible = False
    p.grid.visible = False
    art = text.value.strip()

    artist = []
    mood = []
    with open("C:/Users/aishw/PycharmProjects/Project/ml_lyrics.csv", encoding='latin1') as csvfile:
        csv_read = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(csv_read)
        for row in csv_read:
            artist.append(row[1])
            mood.append(row[3])

    d = {'Artist': artist, 'Mood': mood}
    df = pd.DataFrame(d)

    dfc = df.groupby(['Artist', 'Mood'], sort=True).size().reset_index(name='Count')

    source = ColumnDataSource(dfc)

    art_filter = []
    mood_filter = []

    for i in range(len(dfc)):
        if (dfc.Artist[i] == art):
            art_filter.append(dfc.Count[i])
            mood_filter.append(dfc.Mood[i])

    print(art_filter)
    print(mood_filter)

    total_art = 0.0

    for i in range(len(art_filter)):
        total_art += art_filter[i]

    pct_filter = []
    pct_val_filter = []
    pct_val_filter.append(0)
    pct_val_filter.append(art_filter[0])
    for i in range(1, len(art_filter)):
        pct_val_filter.append(art_filter[i] + pct_val_filter[i])

    for i in range(len(pct_val_filter)):
        pct_filter.append(round((pct_val_filter[i] / total_art), 2))

    l = len(art_filter)

    starts = [p * 2 * pi for p in pct_filter[:-1]]
    ends = [p * 2 * pi for p in pct_filter[1:]]

    if (l == 1):
        colors = ['Green']
    else:
        if (l == 2):
            colors = ['Green', 'Red']
        else:
            if (l == 3):
                colors = ['Green', 'Red', 'Blue']
            else:
                colors = ['Green', 'Red', 'Blue', 'Yellow']

    p.wedge(x=0, y=0, radius=0.5, name="pie", alpha=0.3, start_angle=starts, end_angle=ends, color=colors)

    if (l == 1):
        hover = HoverTool(tooltips=[(mood_filter[0], str(art_filter[0] * 100 / total_art) + '%')], name="h")
    else:
        if (l == 2):
            hover = HoverTool(tooltips=[(mood_filter[0], str(art_filter[0] * 100 / total_art) + '%'),
                                        (mood_filter[1], str(art_filter[1] * 100 / total_art) + '%')], name="h")
        else:
            if (l == 3):
                hover = HoverTool(tooltips=[(mood_filter[0], str(art_filter[0] * 100 / total_art) + '%'),
                                            (mood_filter[1], str(art_filter[1] * 100 / total_art) + '%'),
                                            (mood_filter[2], str(art_filter[2] * 100 / total_art) + '%')], name="h")
            else:
                hover = HoverTool(tooltips=[(mood_filter[0], str(art_filter[0] * 100 / total_art) + '%'),
                                            (mood_filter[1], str(art_filter[1] * 100 / total_art) + '%'),
                                            (mood_filter[2], str(art_filter[2] * 100 / total_art) + '%'),
                                            (mood_filter[3], str(art_filter[3] * 100 / total_art) + '%')], name="h")

    p.add_tools(hover)


text = TextInput(title="Enter an artist name: ", value='')
text.on_change('value', update_title)

inputs = widgetbox(text)
main = column(inputs, p)

curdoc().add_root(main)
