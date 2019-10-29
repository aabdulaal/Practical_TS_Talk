# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:05:06 2019

@author: aabdulaal
"""

import numpy as np
import pandas as pd
from joblib import load
from bokeh.plotting import figure
from bokeh.palettes import Blues4
from bokeh.models import ColumnDataSource, DataRange1d, Select
from bokeh.layouts import row
from bokeh.io import curdoc

def prepare_data(results, test_ts):
    df = pd.concat([results.predicted_mean, results.conf_int(alpha=0.05)], axis=1) 
    df.columns = ['Forecasts', 'Lower 95% CI', 'Upper 95% CI']
    df['CPC'] = test_ts
    df['Anomaly'] = np.nan
    anomaly_mask = (df['CPC'] < df['Lower 95% CI']) | (df['CPC'] > df['Upper 95% CI'])
    df.loc[anomaly_mask, 'Anomaly'] = df.loc[anomaly_mask, 'CPC'].values
    df['date'] = df.index
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d %H:%M')
    return ColumnDataSource(data=df)
    
def create_interactive_plot(source, title):
    plot = figure(x_axis_type="datetime", plot_width=1200, plot_height=360, 
                  tooltips=[("Date", "@date_str"), ("CPC", "$y")],
                  tools="", toolbar_location=None)
    plot.line(x='date', y='CPC', source=source,
              color="blue", line_width=2, legend='Observed (actual)')
    plot.varea(x='date', y1='Upper 95% CI', y2='Lower 95% CI', source=source,
               color=Blues4[2], alpha=0.5, legend='95% confidence interval')
    plot.circle(x='date', y='Anomaly', source=source, 
                size=4, color='red', legend='Anomaly ')
    
    # fixed attributes
    plot.xaxis.axis_label = None
    plot.yaxis.axis_label = "CPC"
    plot.axis.axis_label_text_font_style = "bold"
    plot.x_range = DataRange1d(range_padding=0.0)
    plot.grid.grid_line_alpha = 0.3
    plot.legend.click_policy="hide"
    plot.legend.location = "bottom_left"
    
    return plot

def update_interactive_plot(attrname, old, new):
    params = params_select.value
    plot.title.text = "Forecast Results for SARIMA " + params
    src = prepare_data(results[params], test_ts)
    source.data.update(src.data)

# load data
file = load('data_clean.z')
test_ts = file['test']

# load models
models = load('Models/sarimas.z')

# create forecasts
f_steps = test_ts.shape[0]
results = {key: models[key].get_forecast(f_steps) for key in models.keys()}

# default plot varaibles
params = list(models.keys())[0]
params_select = Select(value=params, title='SARIMA Type', options=sorted(models.keys()))
source = prepare_data(results[params], test_ts)
plot = create_interactive_plot(source, "Forecast Results for SARIMA " + params)

# controls & dashboard setup
params_select.on_change('value', update_interactive_plot)
curdoc().add_root(row(plot, params_select))
curdoc().title = "SARIMA ANOMALY DETECTOR"



