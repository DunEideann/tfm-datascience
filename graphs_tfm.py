import xarray as xr
import torch
from lib import utils, models, data
import sys, time
import numpy as np

DATA_PATH_PREDICTORS = '/lustre/gmeteo/PTICLIMA/DATA/PROJECTIONS/CMIP6_PNACC/CMIP6_models/'
DATA_PATH_PREDICTANDS_READ = '/lustre/gmeteo/PTICLIMA/DATA/AUX/GRID_INTERCOMP/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/GCM/TFM/'
MODELS_PATH = '/oceano/gmeteo/users/reyess/tfm/official-code/models'
DATA_PREDICTORS_TRANSFORMED = '/lustre/gmeteo/WORK/reyess/data/NorthAtlanticRegion_1.5degree/'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/'
VARIABLES_TO_DROP = ['lon_bnds', 'lat_bnds', 'crs']
GCM_NAME = 'EC-Earth3-Veg'
#TODO SACAR DE ESTE CODIGO TODO LO INECESARIO
#TODO Ciclos for no mas de 1??
scenarios = ['ssp585'] #'ssp126', 'ssp245', 'ssp370', 
hist_reference = ('1980-01-01', '2014-12-31')
hist_baseline = ('1995-01-01', '2014-12-31') #95-14
future_1 = ('2021-01-01', '2040-12-31')
future_2 = ('2041-01-01', '2060-12-31')
future_3 = ('2081-01-01', '2100-12-31') 
future = future_3
scenario = 'ssp370'
values_extended = True if scenario == 'ssp585' else False


predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'pti-grid', 'CHELSA']
metrics = ['mean', 'std', '99quantile', '1quantile', 'over30']
seasons = {'spring': 'MAM', 'summer': 'JJA', 'autumn': 'SON', 'winter': 'DJF'}


# FIGURA 3: Predicciones de TRAIN y metricas
#Load predictions and metrics
train_preds = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
train_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
for predictand_name in predictands:
    modelName = f'DeepESD_tas_{predictand_name}' 
    train_preds['annual'][predictand_name] = xr.open_dataset(f'{PREDS_PATH}predTrain_{modelName}.nc')
    train_metrics['annual'][predictand_name] = utils.getMetricsTemp(train_preds['annual'][predictand_name], short = True)
    for season_name, months in seasons.items():
        train_preds[season_name][predictand_name] = train_preds['annual'][predictand_name].isel(time = (train_preds['annual'][predictand_name].time.dt.season == months))
        train_metrics[season_name][predictand_name] = utils.getMetricsTemp(train_preds[season_name][predictand_name], short = True)
        
utils.metricsGraph(train_metrics, figs_path=FIGS_PATH, vmin=[0, 0, -10, 0, 1], vmax=[35, 40, 10, 10, 501], pred_type='train')




# FIGURA 5: Predicciones de TEST y metricas
#Load predictions and metrics
test_preds = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
test_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
for predictand_name in predictands:
    modelName = f'DeepESD_tas_{predictand_name}' 
    test_preds['annual'][predictand_name] = xr.open_dataset(f'{PREDS_PATH}predTest_{modelName}.nc')
    test_metrics['annual'][predictand_name] = utils.getMetricsTemp(test_preds['annual'][predictand_name], short = True)
    for season_name, months in seasons.items():
        test_preds[season_name][predictand_name] = test_preds['annual'][predictand_name].isel(time = (test_preds['annual'][predictand_name].time.dt.season == months))
        test_metrics[season_name][predictand_name] = utils.getMetricsTemp(test_preds[season_name][predictand_name], short = True)
        
utils.metricsGraph(test_metrics, figs_path=FIGS_PATH, vmin=[0, 0, -10, 0, 1], vmax=[35, 40, 10, 10, 501], pred_type='test')


# FIGURA 4: 
#diff_preds = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
diff_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
for predictand_name in predictands:
    test_metrics['annual'][predictand_name]['over30'] = test_metrics['annual'][predictand_name]['over30']/(12*12)
    train_metrics['annual'][predictand_name]['over30'] = train_metrics['annual'][predictand_name]['over30']/(24*12)
    diff_metrics['annual'][predictand_name] = {
        key: test_metrics['annual'][predictand_name][key]-train_metrics['annual'][predictand_name][key] 
            if key != 'std' 
            else test_metrics['annual'][predictand_name][key]/train_metrics['annual'][predictand_name][key] 
            for key in metrics}
    for season_name, months in seasons.items():
        test_metrics[season_name][predictand_name]['over30'] = test_metrics[season_name][predictand_name]['over30']/(12*3)
        train_metrics[season_name][predictand_name]['over30'] = train_metrics[season_name][predictand_name]['over30']/(24*3)
        diff_metrics[season_name][predictand_name] = {
            key: test_metrics[season_name][predictand_name][key]-train_metrics[season_name][predictand_name][key] 
                if key != 'std' 
                else test_metrics[season_name][predictand_name][key]/train_metrics[season_name][predictand_name][key] 
                for key in metrics}
        
utils.metricsGraph(diff_metrics, figs_path=FIGS_PATH, vmin=[-2, 0, -2, -2, 0], vmax=[2, 2, 2, 2, 10], pred_type='diff')

# FIGURA 6 y 7 a 10:
# long_preds = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
# long_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
# for predictand_name in predictands:
#     modelName = f'DeepESD_tas_{predictand_name}_EC-Earth3-Veg_ssp585_2081-01-01-2100-12-31' 
#     long_preds['annual'][predictand_name] = xr.open_dataset(f'{PREDS_PATH}GCM/predGCM_{modelName}.nc')
#     long_metrics['annual'][predictand_name] = utils.getMetricsTemp(long_preds['annual'][predictand_name], short = True)
#     for season_name, months in seasons.items():
#         long_preds[season_name][predictand_name] = long_preds['annual'][predictand_name].isel(time = (long_preds['annual'][predictand_name].time.dt.season == months))
#         long_metrics[season_name][predictand_name] = utils.getMetricsTemp(long_preds[season_name][predictand_name], short = True)
        
# utils.metricsGraph(long_metrics, figs_path=FIGS_PATH, vmin=[0, 0, -10, 0, 1], vmax=[35, 50, 20, 15, 3001], pred_type='ssp585')


# FIGURA 11: EFEMERIDE TEST OBS
yearsTrain = ('1980-01-01', '2003-12-31') #24 años
yearsTest = ('2004-01-01', '2015-12-31') #12 años

obs_efemeride = {}
obs_test_efemeride = {}
efemeride_days = {'cold': '2012-02-12' ,'hot': '2012-08-10'}

for key, efemeride_day in efemeride_days.items():
    for predictand_name in predictands:
        obs_efemeride[predictand_name] = utils.getPredictand(DATA_PATH_PREDICTANDS_SAVE, predictand_name, 'tasmean')
        obs_efemeride[predictand_name] = obs_efemeride[predictand_name].sel(time=slice(*(yearsTrain[0], yearsTest[1])))
        obs_efemeride[predictand_name] = utils.maskData(
                path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                var='tasmean',
                to_slice=(yearsTrain[0], yearsTest[1]),
                objective = obs_efemeride[predictand_name],
                secondGrid = obs_efemeride[predictand_name])
        obs_test_efemeride[predictand_name] = obs_efemeride[predictand_name].sel(time=efemeride_day)
        
    vmin = 10 if key == 'hot' else -10
    vmax = 35 if key == 'hot' else 20
    utils.efemerideGraph(obs_test_efemeride, figs_path=FIGS_PATH, vmin=vmin, vmax=vmax, pred_type=f'efemeride_test_obs_{key}')


# FIGURA 12: EFEMERIDE TEST PREDICTION
test_efemeride = {}

for key, efemeride_day in efemeride_days.items():
    for predictand_name in predictands:
        modelName = f'DeepESD_tas_{predictand_name}' 
        test_efemeride[predictand_name] = xr.open_dataset(f'{PREDS_PATH}predTest_{modelName}.nc')
        test_efemeride[predictand_name] = test_efemeride[predictand_name].sel(time=efemeride_day)
            
    vmin = 10 if key == 'hot' else -10
    vmax = 35 if key == 'hot' else 20
    utils.efemerideGraph(test_efemeride, figs_path=FIGS_PATH, vmin=vmin, vmax=vmax, pred_type=f'efemeride_test_pred_{key}')

# FIGURA 13: EFEMERIDE TRAIN OBS
obs_train_efemeride = {}
efemeride_days = {'cold': '2001-12-25' ,'hot': '2003-08-12'}

for key, efemeride_day in efemeride_days.items():
    for predictand_name in predictands:
        obs_train_efemeride[predictand_name] = obs_efemeride[predictand_name].sel(time=efemeride_day)
            
    vmin = 10 if key == 'hot' else -10
    vmax = 35 if key == 'hot' else 20
    utils.efemerideGraph(obs_train_efemeride, figs_path=FIGS_PATH, vmin=vmin, vmax=vmax, pred_type=f'efemeride_train_obs_{key}')


# FIGURA 14: EFEMERIDE TRAIN PREDICTION
train_efemeride = {}

for key, efemeride_day in efemeride_days.items():
    for predictand_name in predictands:
        modelName = f'DeepESD_tas_{predictand_name}' 
        train_efemeride[predictand_name] = xr.open_dataset(f'{PREDS_PATH}predTrain_{modelName}.nc')
        train_efemeride[predictand_name] = train_efemeride[predictand_name].sel(time=efemeride_day)
            
    vmin = 10 if key == 'hot' else -10
    vmax = 35 if key == 'hot' else 20
    utils.efemerideGraph(train_efemeride, figs_path=FIGS_PATH, vmin=vmin, vmax=vmax, pred_type=f'efemeride_train_pred_{key}')
