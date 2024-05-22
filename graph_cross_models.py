import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from lib import utils, models, data, settings
import time
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap

FIGS_PATH_OBS = '/lustre/gmeteo/WORK/reyess/figs/cross_observations'
FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/cross_models'
FIGS_PATH_HISTO = '/lustre/gmeteo/WORK/reyess/figs/hist'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
DATA_PREDICTORS_TRANSFORMED = '/lustre/gmeteo/WORK/reyess/data/NorthAtlanticRegion_1.5degree/'
MODELS_PATH = '/oceano/gmeteo/users/reyess/tfm/official-code/models'

predictands = ['E-OBS', 'AEMET_0.25deg', 'Iberia01_v1.0', 'pti-grid', 'CHELSA', 'ERA5-Land0.25deg']
seasons = {'spring': 'MAM', 'summer': 'JJA', 'autumn': 'SON', 'winter': 'DJF'}
metrics = ['mean', 'std', '99quantile', 'over30', 'over40']
plot_metrics = ['mean_train', 'real_train', 'diff_train', 'mean_test', 'real_test', 'diff_test']
yearsTrain = ('1980-01-01', '2003-12-31')
yearsTest = ('2004-01-01', '2015-12-31')

#aemet_pred = xr.open_dataset(f'{PREDS_PATH}predTest_AEMET_0.25deg.nc')
# TODO Pasar a archivo unico cosas repetidas
# PREDICTORS
predictors = utils.getPredictors(DATA_PREDICTORS_TRANSFORMED)


data_to_plot = {metric: {season_name: {predictand_name: None for predictand_name in predictands} for season_name in seasons.keys()} for metric in plot_metrics}
pred_to_plot = {metric: {season_name: {predictand_name: None for predictand_name in predictands} for season_name in seasons.keys()} for metric in plot_metrics}

start_time = time.time()

predictand_list = {}
for predictand_name in predictands:
    # PREDICTAND
    predictand = utils.getPredictand(DATA_PATH_PREDICTANDS_SAVE, predictand_name, 'tasmean').sel(time=slice(*(yearsTrain[0], yearsTest[1])))
    predictand_list[predictand_name] = utils.maskData(
        path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
        var='tasmean',
        to_slice=(yearsTrain[0], yearsTest[1]),
        objective = predictand,
        secondGrid = predictand)
predictand_train_mps = {}
predictand_test_mps = {}
for season_name, months in seasons.items():
    datasets_combined = xr.concat(
        [ds.isel(time=(ds.time.dt.season == months)) for ds in predictand_list.values()],
        dim='dataset').mean(dim='dataset')
    #Predictand mean per seasons
    predictand_train_mps[season_name] = utils.getMetricsTemp(datasets_combined.sel(time=slice(*('1980-01-01', '2003-12-31'))))
    predictand_test_mps[season_name] = utils.getMetricsTemp(datasets_combined.sel(time=slice(*('2004-01-01', '2015-12-31'))))



for predictand_name in predictands:
    # X and Y Train/Test
    xTrain, xTest, yTrain, yTest = utils.getTrainTest(predictors, predictand_list[predictand_name], (('1980-01-01', '2003-12-31'), ('2004-01-01', '2015-12-31')))
    for season_name, months in seasons.items():
        y_train_season = yTrain.isel(time= (yTrain.time.dt.season == months))
        y_train_metrics = utils.getMetricsTemp(y_train_season)
        y_test_season = yTest.isel(time= (yTest.time.dt.season == months))
        y_test_metrics = utils.getMetricsTemp(y_test_season)

        data_to_plot['mean_train'][season_name][predictand_name]  = predictand_train_mps[season_name]
        data_to_plot['real_train'][season_name][predictand_name]  = y_train_metrics
        data_to_plot['diff_train'][season_name][predictand_name]  = {key: predictand_train_mps[season_name][key]-y_train_metrics[key] if key != 'std' else predictand_train_mps[season_name][key]/y_train_metrics[key] for key in metrics}

        data_to_plot['mean_test'][season_name][predictand_name]  = predictand_test_mps[season_name]
        data_to_plot['real_test'][season_name][predictand_name]  = y_test_metrics
        data_to_plot['diff_test'][season_name][predictand_name]  = {key: predictand_test_mps[season_name][key]-y_test_metrics[key] if key != 'std' else predictand_test_mps[season_name][key]/y_test_metrics[key] for key in metrics}


cmap = plt.cm.bwr  # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)  
for graph_type, seasons_value in data_to_plot.items():
    for metric in metrics: 
        #Cambiar a un diccionario TODO
        if 'diff' not in graph_type:
            if metric == 'over30':
                v_min = 0
                v_max = 500
            elif metric == 'over40':
                v_min = 0
                v_max = 30
            elif metric == 'std':
                v_min = 0
                v_max = 10
            else:
                v_min = -5
                v_max = 35
        else:
            if metric == 'over30' or metric == 'over40':     
                v_min = -50
                v_max = 50
            elif metric == 'std':
                v_min = 0
                v_max = 1.5
            else:
                v_min = -3
                v_max = 3
        bounds = np.linspace(v_min, v_max, 21)
        norm = BoundaryNorm(bounds, cmap.N)
        #intervalos = np.arange(v_min, v_max+0.1, 10)
        nRows, nCols = 6, 4
        #fig = plt.figure(figsize=(20, 18))#, sharex = True, sharey = True)
        fig, axes = plt.subplots(nRows, nCols, figsize=(20, 18), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
        #gs = fig.add_gridspec(4, 6)
        i = 0
        for season_name, predictand_value in seasons_value.items():
            j = 0

            for predictand_name, value in predictand_value.items():     

                ax = axes[j, i]
                if j == 0:
                    ax.set_title(f'{season_name.capitalize()}', fontsize=14)
                if i == 0:
                    ax.text(-0.07, 0.55, predictand_name, va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize=14)
        
                ax.coastlines(resolution='10m')
                if graph_type != 'diff':
                    dataToPlot = value[metric]['tasmean']
                elif metric != 'std':
                    dataToPlot = (data_to_plot['pred'][season_name][predictand_name][metric] - data_to_plot['real'][season_name][predictand_name][metric])['tasmean']
                else:
                    dataToPlot = (data_to_plot['pred'][season_name][predictand_name][metric]/data_to_plot['real'][season_name][predictand_name][metric])['tasmean']

                im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                                    dataToPlot,
                                    transform=ccrs.PlateCarree(),
                                    cmap=cmap,
                                    norm=BoundaryNorm(bounds, cmap.N))
                #ax.grid(True)
                j += 1
            i += 1

        cax = fig.add_axes([0.91, 0.058, 0.04, 0.88])
        cbar = fig.colorbar(im, cax)#, cmap=cmap_discreto, norm=norm, boundaries=intervalos)#, pad = 0.02, shrink=0.8)
        #|fig.supylabel("HAIWHROIWR")

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        #plt.setp(axes[0, 0].get_ylabel, visible=True)
        plt.savefig(f'{FIGS_PATH_OBS}/intercomparisson_totalMean_{metric}_{graph_type}.pdf')
        plt.close()

total_time = time.time() - start_time
print(f"Graficos de comparacion cruzada (Observaciones) se ejecutaron en {total_time:.2f} segundos.")


# PREDICCIONES
start_time = time.time()

pred_test = {}
pred_train = {}
for predictand_name in predictands:
    modelName = f'DeepESD_tas_{predictand_name}'
    pred_test[predictand_name] = xr.open_dataset(f'{PREDS_PATH}predTest_{modelName}.nc')
    pred_train[predictand_name] = xr.open_dataset(f'{PREDS_PATH}predTrain_{modelName}.nc')

for predictand_name in predictands:
  
    for season_name, months in seasons.items():
        y_train_season = pred_train[predictand_name].isel(time= (pred_train[predictand_name].time.dt.season == months))
        y_train_metrics = utils.getMetricsTemp(y_train_season)
        y_test_season = pred_test[predictand_name].isel(time= (pred_test[predictand_name].time.dt.season == months))
        y_test_metrics = utils.getMetricsTemp(y_test_season)

        pred_to_plot['mean_train'][season_name][predictand_name]  = predictand_train_mps[season_name]
        pred_to_plot['real_train'][season_name][predictand_name]  = y_train_metrics
        pred_to_plot['diff_train'][season_name][predictand_name]  = {key: predictand_train_mps[season_name][key]-y_train_metrics[key] if key != 'std' else predictand_train_mps[season_name][key]/y_train_metrics[key] for key in metrics}

        pred_to_plot['mean_test'][season_name][predictand_name]  = predictand_test_mps[season_name]
        pred_to_plot['real_test'][season_name][predictand_name]  = y_test_metrics
        pred_to_plot['diff_test'][season_name][predictand_name]  = {key: predictand_test_mps[season_name][key]-y_test_metrics[key] if key != 'std' else predictand_test_mps[season_name][key]/y_test_metrics[key] for key in metrics}

for graph_type, seasons_value in pred_to_plot.items():
    for metric in metrics: 
        #Cambiar a un diccionario TODO
        if 'diff' not in graph_type:
            if metric == 'over30':
                v_min = 0
                v_max = 500
            elif metric == 'over40':
                v_min = 0
                v_max = 30
            elif metric == 'std':
                v_min = 0
                v_max = 10
            else:
                v_min = -5
                v_max = 35
        else:
            if metric == 'over30' or metric == 'over40':     
                v_min = -50
                v_max = 50
            elif metric == 'std':
                v_min = 0
                v_max = 1.5
            else:
                v_min = -3
                v_max = 3
        bounds = np.linspace(v_min, v_max, 21)
        norm = BoundaryNorm(bounds, cmap.N)
        #intervalos = np.arange(v_min, v_max+0.1, 10)
        nRows, nCols = 6, 4
        #fig = plt.figure(figsize=(20, 18))#, sharex = True, sharey = True)
        fig, axes = plt.subplots(nRows, nCols, figsize=(20, 18), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
        #gs = fig.add_gridspec(4, 6)
        i = 0
        for season_name, predictand_value in seasons_value.items():
            j = 0

            for predictand_name, value in predictand_value.items():     

                ax = axes[j, i]
                if j == 0:
                    ax.set_title(f'{season_name.capitalize()}', fontsize=14)
                if i == 0:
                    ax.text(-0.07, 0.55, predictand_name, va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize=14)
        
                ax.coastlines(resolution='10m')
                if graph_type != 'diff':
                    dataToPlot = value[metric]['tasmean']
                elif metric != 'std':
                    dataToPlot = (data_to_plot['pred'][season_name][predictand_name][metric] - data_to_plot['real'][season_name][predictand_name][metric])['tasmean']
                else:
                    dataToPlot = (data_to_plot['pred'][season_name][predictand_name][metric]/data_to_plot['real'][season_name][predictand_name][metric])['tasmean']

                im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                                    dataToPlot,
                                    transform=ccrs.PlateCarree(),
                                    cmap=cmap,
                                    norm=BoundaryNorm(bounds, cmap.N))
                #ax.grid(True)
                j += 1
            i += 1

        cax = fig.add_axes([0.91, 0.058, 0.04, 0.88])
        cbar = fig.colorbar(im, cax)#, cmap=cmap_discreto, norm=norm, boundaries=intervalos)#, pad = 0.02, shrink=0.8)
        #|fig.supylabel("HAIWHROIWR")

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        #plt.setp(axes[0, 0].get_ylabel, visible=True)
        plt.savefig(f'{FIGS_PATH}/intercomparisson_totalMean_{metric}_{graph_type}.pdf')
        plt.close()

total_time = time.time() - start_time
print(f"Graficos de comparacion cruzada (Predicciones) se ejecutaron en {total_time:.2f} segundos.")



# OVERALL CROSS MODELS

chelsa_whole = predictand_list['CHELSA']



chelsa_whole_flat = chelsa_whole.tasmean.values.ravel()
chelsa_whole_flat = chelsa_whole_flat[~np.isnan(chelsa_whole_flat)]

chelsa_train = chelsa_whole.sel(time=slice(*(yearsTrain[0], yearsTrain[1])))
chelsa_train_flat = chelsa_train.tasmean.values.ravel()
chelsa_train_flat = chelsa_train_flat[~np.isnan(chelsa_train_flat)]

chelsa_test = chelsa_whole.sel(time=slice(*(yearsTest[0], yearsTest[1])))
chelsa_test_flat = chelsa_test.tasmean.values.ravel()
chelsa_test_flat = chelsa_test_flat[~np.isnan(chelsa_test_flat)]

for season_name, months in seasons.items():
    # Compute figure
    nRows, nCols = 1, 1
    fig = plt.figure(figsize=(10, 6))
    generalIdx = 0

    chelsa_test = chelsa_whole.sel(time=slice(*(yearsTest[0], yearsTest[1])))
    chelsa_test = chelsa_test.isel(time= (chelsa_test.time.dt.season == months))
    chelsa_test_flat = chelsa_test.tasmean.values.ravel()
    chelsa_test_flat = chelsa_test_flat[~np.isnan(chelsa_test_flat)]

    plt.hist(chelsa_test_flat,
        histtype='step',
        bins=np.arange(-10, 40, 0.5),
        color=settings.models_colors['CHELSA(OBS)'],
        linestyle=settings.models_linestyle['CHELSA(OBS)'],
        label='CHELSA(OBS)')

    for predictand_name in predictands:
        currentData = pred_test[predictand_name].isel(time= (pred_test[predictand_name].time.dt.season == months)).sel(time=slice(*(yearsTest[0], yearsTest[1])))
        currentData = currentData.tasmean.values.ravel()
        currentData = currentData[~np.isnan(currentData)]

        plt.hist(currentData,
            histtype='step',
            bins=np.arange(-10, 40, 0.5),
            color=settings.models_colors[predictand_name],
            linestyle=settings.models_linestyle[predictand_name],
            label=predictand_name)
        
    plt.legend()

    # Save figure
    figName = f'histogramFull_{season_name}_chelsa_test.pdf'
    plt.savefig(f'{FIGS_PATH_HISTO}/{figName}', bbox_inches='tight')
    plt.close()

