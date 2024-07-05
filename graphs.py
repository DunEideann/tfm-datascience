import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from lib import utils, models, data
import time
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap

FIGS_PATH = '(.....)/figs/'
PREDS_PATH = '(.....)preds/'
DATA_PATH_PREDICTANDS_SAVE = '(.....)/data/predictand/'

predictands = ['E-OBS', 'AEMET_0.25deg', 'Iberia01_v1.0', 'pti-grid', 'CHELSA', 'ERA5-Land0.25deg']

yPred = {}
yRealTest = {}
yTrain_data = {}
yRealTrain = {}
# TODO Pasar a archivo unico cosas repetidas
yearsTrain = ('1980-01-01', '2003-12-31')
yearsTest = ('2004-01-01', '2015-12-31')

for predictand_name in predictands:
    # Load Pred
    modelName = f'DeepESD_tas_{predictand_name}'
    print(modelName)
    yPred[predictand_name] = xr.open_dataset(f'{PREDS_PATH}predTest_{modelName}.nc')

    #Load Pred Train
    yTrain_data[predictand_name] = xr.open_dataset(f'{PREDS_PATH}predTrain_{modelName}.nc')

    # Load Real Data
    file_name = utils.getFileName(DATA_PATH_PREDICTANDS_SAVE, predictand_name, keyword = 'tasmean')
    predictand_path = f'{DATA_PATH_PREDICTANDS_SAVE}{predictand_name}/{file_name}'
    predictand = xr.open_dataset(predictand_path, chunks=-1)
    predictand = utils.checkCorrectData(predictand) # Transform coordinates and dimensions if necessary
    predictand = utils.checkIndex(predictand)
    predictand = utils.checkUnitsTempt(predictand, 'tasmean')
    predictand=predictand.assign_coords({'time': predictand.indexes['time'].normalize()})
    yTrain = predictand.sel(time=slice(*yearsTrain)).load()
    yTest = predictand.sel(time=slice(*yearsTest)).load()
    baseMask = utils.obtainMask(
        path=f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
        var='tasmean',
        to_slice=(yearsTrain[0], yearsTest[1]))
    yTrainFlat = baseMask.flatten(grid=yTrain, var='tasmean')

    # Extract the raw data from the xarray Dataset
    yTrainFlat_array = utils.toArray(yTrainFlat)
    yTrainFlat['tasmean'].values = yTrainFlat_array
    yTrainUnflatten = baseMask.unFlatten(grid=yTrainFlat, var='tasmean')
    # Same 
    yTestFlat = baseMask.flatten(grid=yTest, var='tasmean')
    yTestFlat_array = utils.toArray(yTestFlat)
    yTestFlat['tasmean'].values = yTestFlat_array
    yTestUnflatten = baseMask.unFlatten(grid=yTestFlat, var='tasmean')
    maskToUse = baseMask


    if np.isnan(yTrainFlat_array).sum() > 0:
        # Second security mask
        secondMask = utils.obtainMask(grid = yTrainUnflatten, var = 'tasmean')
        ySecondTrainFlat = secondMask.flatten(grid=yTrainUnflatten, var='tasmean')
        yTrainFlat_array = utils.toArray(ySecondTrainFlat)
        yTestFlat2 = secondMask.flatten(grid=yTestUnflatten, var='tasmean')
        yTestFlat_array2 = utils.toArray(yTestFlat2)
        yTestFlat2['tasmean'].values = yTestFlat_array2
        yTestUnflatten = secondMask.unFlatten(grid=yTestFlat2, var='tasmean')
        maskToUse = secondMask
    # Add to dictionary
    yRealTest[predictand_name] = yTestUnflatten
    yRealTrain[predictand_name] = yTrainUnflatten

print("Predictandos cargados")

metrics = ['mean', 'std', '99quantile', 'over30', 'over40', 'mean_max_mean']
box_metrics = ['mean', 'std', '99quantile']
plot_metrics = ['pred', 'real', 'diff']
seasons = {'spring': 'MAM', 'summer': 'JJA', 'autumn': 'SON', 'winter': 'DJF'}

data_to_box = {metric: {season_name: {predictand_name: None for predictand_name in predictands} for season_name in seasons.keys()} for metric in box_metrics}
data_to_plot = {metric: {season_name: {predictand_name: None for predictand_name in predictands} for season_name in seasons.keys()} for metric in plot_metrics}
train_to_box = {metric: {season_name: {predictand_name: None for predictand_name in predictands} for season_name in seasons.keys()} for metric in box_metrics}
train_to_plot = {metric: {season_name: {predictand_name: None for predictand_name in predictands} for season_name in seasons.keys()} for metric in plot_metrics}

for season_name, months in seasons.items():
    names = []
    for predictand_name in predictands:
        y_test_season = yRealTest[predictand_name].isel(time = (yPred[predictand_name].time.dt.season == months))
        y_pred_season = yPred[predictand_name].isel(time = (yPred[predictand_name].time.dt.season == months))
        y_train_season = yRealTrain[predictand_name].isel(time = (yRealTrain[predictand_name].time.dt.season == months))
        y_pred_train_season = yTrain_data[predictand_name].isel(time= (yRealTrain[predictand_name].time.dt.season == months))
        y_test_metrics = utils.getMetricsTemp(y_test_season)
        y_pred_metrics = utils.getMetricsTemp(y_pred_season)
        y_train_metrics = utils.getMetricsTemp(y_train_season)
        y_pred_train_metrics = utils.getMetricsTemp(y_pred_train_season)
        mean = (y_pred_metrics['mean']['tasmean'] - y_test_metrics['mean']['tasmean']).data.flatten()
        std = (y_pred_metrics['std']['tasmean']/y_test_metrics['std']['tasmean']).data.flatten()
        quantile = (y_pred_metrics['99quantile']['tasmean'] - y_test_metrics['99quantile']['tasmean']).data.flatten()
        mean_train = (y_pred_train_metrics['mean']['tasmean'] - y_train_metrics['mean']['tasmean']).data.flatten()
        std_train = (y_pred_train_metrics['std']['tasmean']/y_train_metrics['std']['tasmean']).data.flatten()
        quantile_train = (y_pred_train_metrics['99quantile']['tasmean'] - y_train_metrics['99quantile']['tasmean']).data.flatten()
        data_to_plot['pred'][season_name][predictand_name] = y_pred_metrics
        data_to_plot['real'][season_name][predictand_name] = y_test_metrics
        data_to_plot['diff'][season_name][predictand_name] = {key: y_pred_metrics[key]-y_test_metrics if key != 'std' else y_pred_metrics[key]/y_test_metrics[key] for key in metrics}
        train_to_plot['pred'][season_name][predictand_name] = y_pred_train_metrics
        train_to_plot['real'][season_name][predictand_name] = y_train_metrics
        train_to_plot['diff'][season_name][predictand_name] = {key: y_pred_train_metrics[key]-y_train_metrics if key != 'std' else y_pred_train_metrics[key]/y_train_metrics[key] for key in metrics}
        data_to_box['mean'][season_name][predictand_name] = mean[~np.isnan(mean)]
        data_to_box['std'][season_name][predictand_name] = std[~np.isnan(std)]
        data_to_box['99quantile'][season_name][predictand_name] = quantile[~np.isnan(quantile)]
        train_to_box['mean'][season_name][predictand_name] = mean_train[~np.isnan(mean_train)]
        train_to_box['std'][season_name][predictand_name] = std_train[~np.isnan(std_train)]
        train_to_box['99quantile'][season_name][predictand_name] = quantile_train[~np.isnan(quantile_train)]
        names.append(predictand_name)

    print(f"{season_name} metricas cargadas!")
    

    for key, value in data_to_box.items():
        to_plot = [value[season_name][name] for name in names]
        plt.boxplot(to_plot)
        plt.xticks([i for i in range(1, len(predictands)+1)], names, rotation=45)
        
        # Agregar título y etiquetas de los ejes
        plt.title(f'Boxplots Test - Metrica:{key} Estacion:{season_name}')
        plt.xlabel('Predictandos')
        plt.ylabel('Valores', rotation= 90)
        plt.subplots_adjust(bottom=0.24)
        plt.savefig(f'{FIGS_PATH}boxplots/boxplot_{season_name}_{key}.pdf')
        plt.close()

    print(f"{season_name} boxplot de test hecho!")

    for key, value in train_to_box.items():
        to_plot = [value[season_name][name] for name in names]
        plt.boxplot(to_plot)
        plt.xticks([i for i in range(1, len(predictands)+1)], names, rotation=45)
        
        # Agregar título y etiquetas de los ejes
        plt.title(f'Boxplots Train - Metrica:{key} Estacion:{season_name}')
        plt.xlabel('Predictandos')
        plt.ylabel('Valores', rotation= 90)
        plt.subplots_adjust(bottom=0.24)
        plt.savefig(f'{FIGS_PATH}boxplots/boxplot_{season_name}_{key}.pdf')
        plt.close()

    print(f"{season_name} boxplot de train hecho!")



# Guardado de mapas de varias season comparadas
utils.multiMapPerSeason(data_to_plot, metrics, plot_metrics, f'{FIGS_PATH}predictions')
utils.multiMapPerSeason(train_to_plot, metrics, plot_metrics, f'{FIGS_PATH}predictions')


