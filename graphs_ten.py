import xarray as xr
import torch
from lib import utils, models, data
import sys, time
import numpy as np

DATA_PATH_PREDICTORS = '/lustre/gmeteo/PTICLIMA/DATA/PROJECTIONS/CMIP6_PNACC/CMIP6_models/'
DATA_PATH_PREDICTANDS_READ = '/lustre/gmeteo/PTICLIMA/DATA/AUX/GRID_INTERCOMP/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/GCM/AEMET/'
MODELS_PATH = '/oceano/gmeteo/users/reyess/tfm/official-code/models'
DATA_PREDICTORS_TRANSFORMED = '/lustre/gmeteo/WORK/reyess/data/NorthAtlanticRegion_1.5degree/'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/GCM/AEMET/'
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


model_name = f'AEMET_0.25deg_'
predictands = [f"{model_name}{i}" for i in range(1, 11)]


# Cargamos Predicciones a escenarios
print(f"Current period: {future}")
yPredLoaded = {}
yPredMetrics = {}
yPredLoaded[scenario] = {}
yPredMetrics[scenario] = {}
for predictand_name in predictands:
    modelName = f'DeepESD_tas_{predictand_name}' 
    yPredLoaded[scenario][predictand_name] = xr.open_dataset(f'{PREDS_PATH}predGCM_{modelName}_{GCM_NAME}_{scenario}_{future[0]}-{future[1]}.nc')
    yPredMetrics[scenario][predictand_name] = utils.getMetricsTemp(yPredLoaded[scenario][predictand_name]) # CARGAMOS METRICAS
        
yHisto = {}
yHistoMetrics = {}
yHisto[scenario] = {}
yHistoMetrics[scenario] = {}
for predictand_name in predictands:
    modelName = f'DeepESD_tas_{predictand_name}' 
    yHisto[scenario][predictand_name] = xr.open_dataset(f'{PREDS_PATH}predGCM_{modelName}_{GCM_NAME}_{scenario}_{hist_baseline[0]}-{hist_baseline[1]}.nc')
    yHistoMetrics[scenario][predictand_name] = utils.getMetricsTemp(yHisto[scenario][predictand_name]) # CARGAMOS METRICAS
        

metrics = ['mean', 'std', '99quantile', 'over30', 'over40', 'mean_max_mean']
seasons = {'spring': 'MAM', 'summer': 'JJA', 'autumn': 'SON', 'winter': 'DJF'}
plot_metrics = ['pred', 'hist', 'diff']


data_to_plot = {scenario: {metric: {season_name: {predictand_name: None for predictand_name in predictands} for season_name in seasons.keys()} for metric in plot_metrics}}

for season_name, months in seasons.items():
    for predictand_name in predictands:
        y_pred_season = yPredLoaded[scenario][predictand_name].isel(time = (yPredLoaded[scenario][predictand_name].time.dt.season == months))
        y_hist_season = yHisto[scenario][predictand_name].isel(time = (yHisto[scenario][predictand_name].time.dt.season == months))
        y_pred_metrics = utils.getMetricsTemp(y_pred_season)#, mask=maskToUse)
        y_hist_metrics = utils.getMetricsTemp(y_hist_season)#, mask=maskToUse)

        data_to_plot[scenario]['pred'][season_name][predictand_name] = y_pred_metrics
        data_to_plot[scenario]['hist'][season_name][predictand_name] = y_hist_metrics
        data_to_plot[scenario]['diff'][season_name][predictand_name] = {key: y_pred_metrics[key]-y_hist_metrics if key != 'std' else y_pred_metrics[key]/y_hist_metrics[key] for key in metrics}
        #names.append(predictand_name)

print(f"{season_name} metricas cargadas!")

years = f"{future[0].split('-')[0]}-{future[1].split('-')[0]}"


utils.multiMapPerSeason(data_to_plot[scenario], metrics, plot_metrics, f'{FIGS_PATH}Ten', extra_path=f"{years}-{scenario}", values_extended=values_extended)

print("10 GRAFICOS TERMINADOs CON EXITO!")


# SELECCIONAR MIN, MAX y MEAN
new_predictands = ['Min', 'Mean', 'Max']

data_to_plot_2 = {scenario: {metric: {season_name: {predictand_name: None for predictand_name in new_predictands} for season_name in seasons.keys()} for metric in plot_metrics}}
yThree = {'Min': {}, 'Max': {}, 'Mean': {}}

for season_name, months in seasons.items():
    selected_data = {predictand_name: yPredLoaded[scenario][predictand_name].isel(time=(yPredLoaded[scenario][predictand_name].time.dt.season == months)) for predictand_name in yPredLoaded[scenario]}
    for predictand in new_predictands:
        yThree[predictand][season_name] = utils.getDataset(selected_data, predictand, 'tasmean')
        current_metric = yThree[predictand][season_name]
        
        y_hist_season = yHisto[scenario][predictand_name].isel(time = (yHisto[scenario][predictand_name].time.dt.season == months))
        y_hist_metrics = utils.getMetricsTemp(y_hist_season)

        data_to_plot_2[scenario]['pred'][season_name][predictand] = {'mean': current_metric}
        data_to_plot_2[scenario]['hist'][season_name][predictand] = y_hist_metrics
        data_to_plot_2[scenario]['diff'][season_name][predictand] = {'mean': current_metric-y_hist_metrics['mean']}

utils.multiMapPerSeason(data_to_plot_2[scenario], ['mean'], plot_metrics, f'{FIGS_PATH}Three', extra_path=f"{years}-{scenario}", values_extended=values_extended)
