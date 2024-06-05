import xarray as xr
import torch
from lib import utils, models, data
import sys, time
import numpy as np

DATA_PATH_PREDICTORS = '/lustre/gmeteo/PTICLIMA/DATA/PROJECTIONS/CMIP6_PNACC/CMIP6_models/'
DATA_PATH_PREDICTANDS_READ = '/lustre/gmeteo/PTICLIMA/DATA/AUX/GRID_INTERCOMP/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/GCM/'
MODELS_PATH = '/oceano/gmeteo/users/reyess/tfm/official-code/models'
DATA_PREDICTORS_TRANSFORMED = '/lustre/gmeteo/WORK/reyess/data/NorthAtlanticRegion_1.5degree/'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/GCM/'#GCM
VARIABLES_TO_DROP = ['lon_bnds', 'lat_bnds', 'crs']
GCM_NAME = 'EC-Earth3-Veg'

scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
main_scenerio = 'ssp370'
hist_reference = ('1980-01-01', '2014-12-31')
hist_baseline = ('1995-01-01', '2014-12-31') #95-14
future_1 = ('2021-01-01', '2040-12-31')
future_2 = ('2041-01-01', '2060-12-31')
future_3 = ('2081-01-01', '2100-12-31') 

future = future_1
predictands = ['E-OBS', 'AEMET_0.25deg', 'Iberia01_v1.0', 'pti-grid', 'CHELSA', 'ERA5-Land0.25deg']

for future in [hist_baseline, future_1, future_2, future_3]:
    # Cargamos Predicciones a escenarios
    yPredLoaded = {}
    yPredMetrics = {}
    for scenario in scenarios:
        yPredLoaded[scenario] = {}
        yPredMetrics[scenario] = {}
        for predictand_name in predictands:
            modelName = f'DeepESD_tas_{predictand_name}' 
            yPredLoaded[scenario][predictand_name] = xr.open_dataset(f'{PREDS_PATH}predGCM_{modelName}_{GCM_NAME}_{scenario}_{future[0]}-{future[1]}.nc')
            yPredMetrics[scenario][predictand_name] = utils.getMetricsTemp(yPredLoaded[scenario][predictand_name]) # CARGAMOS METRICAS
            utils.getGraphsTempGCM(yPredMetrics[scenario][predictand_name], scenario, FIGS_PATH, GCM_NAME, predictand_name)# REALIZAMOS GRAFICOS COMPARATIVOS

    yHisto = {}
    yHistoMetrics = {}
    for scenario in scenarios:
        yHisto[scenario] = xr.open_dataset(f'{PREDS_PATH}predGCM_{modelName}_{GCM_NAME}_{scenario}_{hist_baseline[0]}-{hist_baseline[1]}.nc')
        yHistoMetrics[scenario] = utils.getMetricsTemp(yHisto[scenario]) # CARGAMOS METRICAS
        utils.getGraphsTempGCM(yHistoMetrics[scenario], scenario, FIGS_PATH, GCM_NAME, predictand_name)# REALIZAMOS GRAFICOS COMPARATIVOS

    print("Terminado con exito!")

    metrics = ['mean', 'std', '99quantile', 'over30', 'over40', 'mean_max_mean']
    seasons = {'spring': 'MAM', 'summer': 'JJA', 'autumn': 'SON', 'winter': 'DJF'}
    plot_metrics = ['pred', 'hist', 'diff']


    data_to_plot = {scenario: {metric: {season_name: {predictand_name: None for predictand_name in predictands} for season_name in seasons.keys()} for metric in plot_metrics} for scenario in scenarios}

    for scenario in scenarios:
        for season_name, months in seasons.items():
            for predictand_name in predictands:
                y_pred_season = yPredLoaded[scenario][predictand_name].isel(time = (yPredLoaded[scenario][predictand_name].time.dt.season == months))
                y_hist_season = yHisto[scenario].isel(time = (yHisto[scenario].time.dt.season == months))
                y_pred_metrics = utils.getMetricsTemp(y_pred_season)#, mask=maskToUse)
                y_hist_metrics = utils.getMetricsTemp(y_hist_season)#, mask=maskToUse)

                data_to_plot[scenario]['pred'][season_name][predictand_name] = y_pred_metrics
                data_to_plot[scenario]['hist'][season_name][predictand_name] = y_hist_metrics
                data_to_plot[scenario]['diff'][season_name][predictand_name] = {key: y_pred_metrics[key]-y_hist_metrics if key != 'std' else y_pred_metrics[key]/y_hist_metrics[key] for key in metrics}
                #names.append(predictand_name)

        print(f"{season_name} metricas cargadas!")


    values = {'diff': {'over30': (-50, 50), 'over40': (-10, 10), 'std': (0, 10), 'else': (-5, 5)},
            'noDiff': {'over30': (0, 500), 'over40': (0, 30), 'std': (0, 2.5), 'else': (-5, 45)}}


    years = f"{future[0].split('-')[0]}-{future[1].split('-')[0]}"
    for scenario in scenarios:
        utils.multiMapPerSeason(data_to_plot[scenario], metrics, plot_metrics, f'{FIGS_PATH}predictions', extra_path=f"{years}-{scenario}", values = values, color_extended=True)