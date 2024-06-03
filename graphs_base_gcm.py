import xarray as xr
import torch
from lib import utils, models, data
import sys, time
import numpy as np

DATA_PATH_PREDICTORS = '/lustre/gmeteo/PTICLIMA/DATA/PROJECTIONS/CMIP6_PNACC/CMIP6_models/'
FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/GCM_Base/'
#GCM_NAME = sys.argv[2]
GCM_NAME = 'EC-Earth3-Veg'

scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
main_scenerio = 'ssp370'
hist_reference = ('1980-01-01', '2014-12-31')
hist_baseline = ('1995-01-01', '2014-12-31') #95-14
future_1 = ('2021-01-01', '2040-12-31')
future_2 = ('2041-01-01', '2060-12-31')
future_3 = ('2081-01-01', '2100-12-31') 
periods = [future_1, future_2, future_3]
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']


gcm_hist = xr.open_dataset(f'{DATA_PATH_PREDICTORS}tas_EC-Earth3-Veg_historical_r1i1p1f1_19500101-20141231.nc')
hist_tas = gcm_hist.sel(time=slice(*(hist_baseline[0], hist_baseline[1])))
hist_tas = hist_tas.drop_vars('height')#.sel(dim=~(hist_tas.dims.get('height')))
hist_tas = utils.checkCorrectData(hist_tas)
hist_tas = utils.checkUnitsTempt(hist_tas, 'tas')
hist_tas = hist_tas.drop_dims('bnds')
hist_tas = hist_tas.reindex(lat=list(reversed(hist_tas.lat)))
hist_tas = hist_tas.assign_coords({'time': hist_tas.indexes['time'].normalize()})

hist_metrics = utils.getMetricsTemp(hist_tas, 'tas')


for scenario in scenarios: 
    gcm_futu = xr.open_dataset(f'{DATA_PATH_PREDICTORS}tas_EC-Earth3-Veg_{scenario}_r1i1p1f1_20150101-21001231.nc')
    print("GCM")
    print(gcm_futu)
    for future in periods:
        #predictor = utils.loadGcm(GCM_NAME, scenario, (hist_reference[0], future_3[1]), DATA_PATH_PREDICTORS)
        #gcm_futu = xr.open_dataset(f'{DATA_PATH_PREDICTORS}tas_EC-Earth3-Veg_{scenario}_r1i1p1f1_20150101-21001231.nc')
        future_tas = gcm_futu.sel(time=slice(*(future[0], future[1])))
        future_tas = future_tas.drop_vars('height')#.sel(dim=~(future_tas.dims.get('height')))
        future_tas = utils.checkCorrectData(future_tas)
        future_tas = utils.checkUnitsTempt(future_tas, 'tas')
        future_tas = future_tas.drop_dims('bnds')
        future_tas = future_tas.reindex(lat=list(reversed(future_tas.lat)))
        future_tas = future_tas.assign_coords({'time': future_tas.indexes['time'].normalize()})
        future_metrics = utils.getMetricsTemp(future_tas, 'tas')
        print("FUTURE")
        print(future_tas)
        
        utils.graphsBaseGCM(future_metrics, hist_metrics, f'{FIGS_PATH}metricsGCM_{scenario}_{future[0]}-{future[1]}.pdf')