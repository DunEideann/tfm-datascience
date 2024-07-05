import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from lib import utils, models, data
import time
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap

FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/climatology'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
DATA_PREDICTORS_TRANSFORMED = '/lustre/gmeteo/WORK/reyess/data/NorthAtlanticRegion_1.5degree/'
MODELS_PATH = '/oceano/gmeteo/users/reyess/tfm/official-code/models'

predictands = ['E-OBS', 'AEMET_0.25deg', 'Iberia01_v1.0', 'pti-grid', 'CHELSA', 'ERA5-Land0.25deg']
seasons = {'spring': 'MAM', 'summer': 'JJA', 'autumn': 'SON', 'winter': 'DJF'}
plot_metrics = ['train', 'test', 'whole']
metrics = ['mean', 'std', '99quantile', 'over30', 'over40', 'mean_max_mean']
yearsTrain = ('1980-01-01', '2003-12-31')
yearsTest = ('2004-01-01', '2015-12-31')

predictors = utils.getPredictors(DATA_PREDICTORS_TRANSFORMED)
data_to_plot = {metric: {season_name: {predictand_name: None for predictand_name in predictands} for season_name in seasons.keys()} for metric in plot_metrics}


for predictand_name in predictands:
    predictand = utils.getPredictand(DATA_PATH_PREDICTANDS_SAVE, predictand_name, 'tasmean')
    predictand = predictand.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
    predictand = utils.maskData(
        path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
        var='tasmean',
        to_slice=(yearsTrain[0], yearsTest[1]),
        objective = predictand,
        secondGrid = predictand)

    # X and Y Train/Test
    xTrain, xTest, yTrain, yTest = utils.getTrainTest(predictors, predictand, (yearsTrain, yearsTest))

    for season_name, months in seasons.items():
        y_train_season = yTrain.isel(time= (yTrain.time.dt.season == months))
        y_test_season = yTest.isel(time= (yTest.time.dt.season == months))
        y_train_metrics = utils.getMetricsTemp(y_train_season)
        y_test_metrics = utils.getMetricsTemp(y_test_season)
        y_metrics = utils.getMetricsTemp(predictand.sel(time=slice(*(yearsTrain[0], yearsTrain[1]))).load())
        data_to_plot['train'][season_name][predictand_name] = y_train_metrics
        data_to_plot['test'][season_name][predictand_name] = y_test_metrics
        data_to_plot['whole'][season_name][predictand_name] = y_metrics


start_time = time.time()
utils.multiMapPerSeason(data_to_plot, metrics, plot_metrics, f'{FIGS_PATH}')

total_time = time.time() - start_time
print(f"El código de graficos de test se ejecutó en {total_time:.2f} segundos.")