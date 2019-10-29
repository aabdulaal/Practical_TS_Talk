# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 13:56:01 2019

@author: aabdulaal
"""

# =============================================================================
# Load Libraries
# =============================================================================
import pandas as pd
from joblib import load, dump
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pyflux as pf

# =============================================================================
# Load Data
# =============================================================================
file = load('data_clean.z')
train_ts = file['train']
val_ts = file['val']
test_ts = file['test']
is_ts = train_ts.append(val_ts)

# =============================================================================
# Build SARIMA models
# =============================================================================
params=[((1,0,0),(0,1,3,24)),
        ((3,0,0),(0,1,3,24)),
        ((2,0,0),(0,1,3,24)),
        ((3,0,0),(0,1,2,24))]

# reduce model size
def reduce_size(model):
    # reduce
    reduced_list = [
                'model.get_forecast.__self__.cov_params_approx = None',
                'model.get_forecast.__self__.filtered_state = None',
                'model.get_forecast.__self__.filtered_state_cov = None',
                'model.get_forecast.__self__.fittedvalues = None',
                'model.get_forecast.__self__.forecasts = None',
                'model.get_forecast.__self__.forecasts_error = None',
                'model.get_forecast.__self__.forecasts_error_cov = None',
                'model.get_forecast.__self__.predicted_state = None',
                'model.get_forecast.__self__.predicted_state_cov = None',
                'model.get_forecast.__self__.scaled_smoothed_estimator = None',
                'model.get_forecast.__self__.scaled_smoothed_estimator_cov = None',
                'model.get_forecast.__self__.smoothed_measurement_disturbance = None',
                'model.get_forecast.__self__.smoothed_measurement_disturbance_cov = None',
                'model.get_forecast.__self__.smoothed_state = None',
                'model.get_forecast.__self__.smoothed_state_autocov = None',
                'model.get_forecast.__self__.smoothed_state_cov = None',
                'model.get_forecast.__self__.smoothed_state_disturbance = None',
                'model.get_forecast.__self__.smoothed_state_disturbance_cov = None',
                'model.get_forecast.__self__.filter_results._kalman_gain = None',
                'model.get_forecast.__self__.filter_results.filtered_state = None',
                'model.get_forecast.__self__.filter_results.filtered_state_cov = None',
                'model.get_forecast.__self__.filter_results.nmissing = None',
                'model.get_forecast.__self__.filter_results.predicted_state = None',
                'model.get_forecast.__self__.filter_results.predicted_state_cov = None',
                'model.get_forecast.__self__.filter_results.scaled_smoothed_estimator = None',
                'model.get_forecast.__self__.filter_results.scaled_smoothed_estimator_cov = None',
                'model.get_forecast.__self__.filter_results.smoothed_measurement_disturbance = None',
                'model.get_forecast.__self__.filter_results.smoothed_measurement_disturbance_cov = None',
                'model.get_forecast.__self__.filter_results.smoothed_state = None',
                'model.get_forecast.__self__.filter_results.smoothed_state_autocov = None',
                'model.get_forecast.__self__.filter_results.smoothed_state_cov = None',
                'model.get_forecast.__self__.filter_results.smoothed_state_disturbance = None',
                'model.get_forecast.__self__.filter_results.smoothed_state_disturbance_cov = None',
                'model.get_forecast.__self__.filter_results.smoothing_error = None',
                'model.get_forecast.__self__.filter_results.tmp1 = None',
                'model.get_forecast.__self__.filter_results.tmp2 = None',
                'model.get_forecast.__self__.filter_results.tmp3 = None',
                'model.get_forecast.__self__.filter_results.tmp4 = None',
                'model.get_forecast.__self__.filter_results.model._kalman_filters = None',
                'model.get_forecast.__self__.filter_results.model._kalman_smoothers = None']
    for rule in reduced_list:
        try:
             exec(rule)
        except:
            continue
    return model

output = {}
for param in params:
    model = SARIMAX(is_ts, order=param[0], seasonal_order=param[1])
    sarima = model.fit(dis=0)
    sarima = reduce_size(sarima)
    output[str(param)]=sarima
    
dump(output , 'Models/sarimas.z', compress=True)

# =============================================================================
# Build Bayesian ARIMA model
# =============================================================================
model = pf.ARIMA(data=is_ts.values, ar=72, integ=0, ma=0, family=pf.Normal())
barima = model.fit(dis=0)
dump(barima, 'Models/barima.z', compress=True)
