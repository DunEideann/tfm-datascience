import xarray as xr
import torch
from lib import utils, models, data
import sys, time
import numpy as np

DATA_PATH_PREDICTORS = '(.....)/PTICLIMA/DATA/PROJECTIONS/CMIP6_PNACC/CMIP6_models/'
DATA_PATH_PREDICTANDS_READ = '(.....)/PTICLIMA/DATA/AUX/GRID_INTERCOMP/'
DATA_PATH_PREDICTANDS_SAVE = '(.....)/data/predictand/'
FIGS_PATH = '(.....)/figs/GCM/'
MODELS_PATH = '(.....)/models'
DATA_PREDICTORS_TRANSFORMED = '(.....)/data/NorthAtlanticRegion_1.5degree/'
PREDS_PATH = '(.....)/preds/GCM/'#GCM
VARIABLES_TO_DROP = ['lon_bnds', 'lat_bnds', 'crs']
GCM_NAME = 'EC-Earth3-Veg'

scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
main_scenerio = 'ssp370'
hist_reference = ('1980-01-01', '2014-12-31')
hist_baseline = ('1995-01-01', '2014-12-31') #95-14
future_1 = ('2021-01-01', '2040-12-31')
future_2 = ('2041-01-01', '2060-12-31')
future_3 = ('2081-01-01', '2100-12-31') 
futures = [hist_baseline, future_1, future_2, future_3]


predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'pti-grid', 'CHELSA']
for future in futures:
    # Cargamos Predicciones a escenarios
    print(f"Current period: {future}")
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
        yHisto[scenario] = {}
        yHistoMetrics[scenario] = {}
        for predictand_name in predictands:
            modelName = f'DeepESD_tas_{predictand_name}' 
            yHisto[scenario][predictand_name] = xr.open_dataset(f'{PREDS_PATH}predGCM_{modelName}_{GCM_NAME}_{scenario}_{hist_baseline[0]}-{hist_baseline[1]}.nc')
            yHistoMetrics[scenario][predictand_name] = utils.getMetricsTemp(yHisto[scenario][predictand_name]) # CARGAMOS METRICAS
            utils.getGraphsTempGCM(yHistoMetrics[scenario][predictand_name], scenario, FIGS_PATH, GCM_NAME, predictand_name)# REALIZAMOS GRAFICOS COMPARATIVOS


    metrics = ['mean', 'std', '99quantile', 'over30', 'over40', 'mean_max_mean']
    seasons = {'spring': 'MAM', 'summer': 'JJA', 'autumn': 'SON', 'winter': 'DJF'}
    plot_metrics = ['pred', 'hist', 'diff']


    data_to_plot = {scenario: {metric: {season_name: {predictand_name: None for predictand_name in predictands} for season_name in seasons.keys()} for metric in plot_metrics} for scenario in scenarios}

    for scenario in scenarios:
        for season_name, months in seasons.items():
            for predictand_name in predictands:
                y_pred_season = yPredLoaded[scenario][predictand_name].isel(time = (yPredLoaded[scenario][predictand_name].time.dt.season == months))
                y_hist_season = yHisto[scenario][predictand_name].isel(time = (yHisto[scenario][predictand_name].time.dt.season == months))
                y_pred_metrics = utils.getMetricsTemp(y_pred_season)#, mask=maskToUse)
                y_hist_metrics = utils.getMetricsTemp(y_hist_season)#, mask=maskToUse)

                data_to_plot[scenario]['pred'][season_name][predictand_name] = y_pred_metrics
                data_to_plot[scenario]['hist'][season_name][predictand_name] = y_hist_metrics
                data_to_plot[scenario]['diff'][season_name][predictand_name] = {key: y_pred_metrics[key]-y_hist_metrics if key != 'std' else y_pred_metrics[key]/y_hist_metrics[key] for key in metrics}

        print(f"{season_name} metricas cargadas!")


    values = {'diff': {'over30': (-50, 50), 'over40': (-10, 10), 'std': (0, 2), 'else': (-5, 5)},
            'noDiff': {'over30': (0, 500), 'over40': (0, 30), 'std': (0, 10), 'else': (-5, 45)}}   


    years = f"{future[0].split('-')[0]}-{future[1].split('-')[0]}"


    for scenario in scenarios:
        if scenario == scenarios[0] and future == future_1:
            values['diff']['else'] = (-4, 4)
        elif scenario == scenarios[0] and future == future_2:
            values['diff']['else'] = (-3, 7)
        elif scenario == scenarios[0] and future == future_3:
            values['diff']['else'] = (-2, 9)
        elif scenario == scenarios[2] and future == future_1:
            values['diff']['else'] = (-3, 4)
        elif scenario == scenarios[2] and future == future_2:
            values['diff']['else'] = (-2, 7)
        elif scenario == scenarios[2] and future == future_3:
            values['diff']['else'] = (-1, 9)
        elif scenario == scenarios[3] and future == future_1:
            values['diff']['else'] = (-3, 4)
        elif scenario == scenarios[3] and future == future_2:
            values['diff']['else'] = (-2, 7)
        elif scenario == scenarios[3] and future == future_3:
            values['diff']['else'] = (-1, 9)
        
        values_extended = True if scenario == scenarios[3] else False

        utils.multiMapPerSeason(data_to_plot[scenario], metrics, plot_metrics, f'{FIGS_PATH}predictions', extra_path=f"{years}-{scenario}", values = values, values_extended=values_extended)

print("TERMINADO CON EXITO!")