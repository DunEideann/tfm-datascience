import xarray as xr
import torch
from lib import utils, models, data
import sys, time
import numpy as np

DATA_PATH_PREDICTORS = '/lustre/gmeteo/PTICLIMA/DATA/PROJECTIONS/CMIP6_PNACC/CMIP6_models/'
DATA_PATH_PREDICTANDS_READ = '/lustre/gmeteo/PTICLIMA/DATA/AUX/GRID_INTERCOMP/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/'
MODELS_PATH = '/oceano/gmeteo/users/reyess/tfm/official-code/models'
DATA_PREDICTORS_TRANSFORMED = '/lustre/gmeteo/WORK/reyess/data/NorthAtlanticRegion_1.5degree/'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/'
VARIABLES_TO_DROP = ['lon_bnds', 'lat_bnds', 'crs']
LAT_SLICE = slice(33.5, 48.6)
LON_SLICE = slice(-10.5, 4.6)
#MODEL_NAME = sys.argv[1]
#GCM_NAME = sys.argv[2]
MODEL_NAME = 'E-OBS'
GCM_NAME = 'EC-Earth3-Veg'

scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
main_scenerio = 'ssp370'
hist_reference = ('1980-01-01', '2014-12-31')
hist_baseline = ('1995-01-01', '2014-12-31') #95-14
future_1 = ('2021-01-01', '2040-12-31')
future_2 = ('2041-01-01', '2060-12-31')
future_3 = ('2081-01-01', '2100-12-31') 


# Cargamos Predicciones a escenarios
yPredLoaded = {}
yPredMetrics = {}
for scenario in scenarios:
    yPredLoaded[scenario] = xr.open_dataset(f'{PREDS_PATH}predTest_{MODEL_NAME}_{GCM_NAME}_{scenario}.nc')
    yPredMetrics[scenario] = utils.getMetricsTemp(yPredLoaded[scenario]) # CARGAMOS METRICAS
    utils.getGraphsTempGCM(yPredMetrics[scenario], scenario, FIGS_PATH, GCM_NAME, MODEL_NAME)# REALIZAMOS GRAFICOS COMPARATIVOS


print("Terminado con exito!")

metrics = ['mean', 'std', '99quantile', 'over30', 'over40', 'mean_max_mean']
seasons = {'spring': 'MAM', 'summer': 'JJA', 'autumn': 'SON', 'winter': 'DJF'}
plot_metrics = ['pred', 'hist', 'diff']
predictands = [MODEL_NAME]

data_to_plot = {metric: {season_name: {predictand_name: None for predictand_name in predictands} for season_name in seasons.keys()} for metric in plot_metrics}
yHisto = utils.loadGcm(GCM_NAME, scenario, (hist_baseline[0], hist_baseline[1]), DATA_PATH_PREDICTORS)

for scenario in scenarios:
    for season_name, months in seasons.items():
        #months = 'MAM'
        #names = []
        #predictand_name = 'E-OBS'
        for predictand_name in predictands:
            y_pred_season = yPredLoaded[scenario].isel(time = (yPredLoaded[scenario].time.dt.season == months))
            y_hist_season = yHisto.isel(time = (yHisto.time.dt.season == months))
            y_pred_metrics = utils.getMetricsTemp(y_pred_season)#, mask=maskToUse)
            y_hist_metrics = utils.getMetricsTemp(y_hist_season)#, mask=maskToUse)

            data_to_plot['pred'][season_name][predictand_name] = y_pred_metrics
            data_to_plot['hist'][season_name][predictand_name] = y_hist_metrics
            data_to_plot['diff'][season_name][predictand_name] = {key: y_pred_metrics[key]-y_hist_metrics if key != 'std' else y_pred_metrics[key]/y_hist_metrics[key] for key in metrics}
            #names.append(predictand_name)

    print(f"{season_name} metricas cargadas!")


values = {'diff': {'over30': (-50, 50), 'over40': (-10, 10), 'std': (0, 10), 'else': (-5, 5)},
          'noDiff': {'over30': (0, 500), 'over40': (0, 30), 'std': (0, 2.5), 'else': (-5, 45)}}





for scenario in scenarios:
    utils.multiMapPerSeason(data_to_plot[scenario], metrics, FIGS_PATH, values)