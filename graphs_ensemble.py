import xarray as xr
import torch
from lib import utils, models, data, settings
import sys, time
import numpy as np
from decimal import Decimal
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import pickle
import os

FIG_NUM = str(sys.argv[1])

DATA_PATH_PREDICTORS = '/lustre/gmeteo/PTICLIMA/DATA/PROJECTIONS/CMIP6_PNACC/CMIP6_models/'
DATA_PATH_PREDICTANDS_READ = '/lustre/gmeteo/PTICLIMA/DATA/AUX/GRID_INTERCOMP/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
DATA_PATH_SHAPE = '/lustre/gmeteo/WORK/reyess/shapes/'
FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs-ensemble/climatology/'
MODELS_PATH = '/oceano/gmeteo/users/reyess/tfm/official-code/models-ensemble'
DATA_PREDICTORS_TRANSFORMED = '/lustre/gmeteo/WORK/reyess/data/NorthAtlanticRegion_1.5degree/'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/'

GCM_NAME = 'EC-Earth3-Veg'


ENSEMBLE_QUANTITY = 10
predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'CHELSA', 'Ensemble']
predictands_map = {'ERA5-Land0.25deg': 'ERA5-Land', 'E-OBS': 'E-OBS','AEMET_0.25deg':'ROCIO-IBEB', 'Iberia01_v1.0':'Iberia01', 'CHELSA': 'CHELSA', 'Ensemble': 'Ensemble'}

hist_baseline = ('1995-01-01', '2014-12-31') #95-14
yearsTrain = ('1980-01-01', '2003-12-31')
yearsTest = ('2004-01-01', '2015-12-31')
yearsTrainTest = ('1980-01-01','2015-12-31')
future_1 = ('2021-01-01', '2040-12-31')
future_2 = ('2041-01-01', '2060-12-31')
future_3 = ('2081-01-01', '2100-12-31') 
future_4 = ('2061-01-01', '2080-12-31')
days_selected = ('2000-07-31', '2010-07-31')
main_scenario = 'ssp585'

fechas_a_eliminar = ["1982-02-28", "1986-02-28", "1990-02-28", "1994-02-28", "1998-02-28", "2002-02-28", "2006-02-28", "2010-02-28", "2014-02-28"]
fechas_a_eliminar = np.array(fechas_a_eliminar, dtype="datetime64")

file_path = f'/oceano/gmeteo/users/reyess/tfm/official-code/models-ensemble/generalMask{yearsTrainTest[0]}-{yearsTrainTest[1]}.pkl'
if os.path.exists(file_path):
    print(f"Existe path: {file_path}")
    with open(file_path, 'rb') as f:
        newMask = pickle.load(f)
else:
    print("No existe path")
    newMask = None

# FIGURA TEMPERATURA AGREGADA EN TRAIN Y TEST DE TODOS PARA 1 REALIZACION
# CARGO DATOS Y ME QUEDO SOLO CON METRICAS
if FIG_NUM == '0':
    metrics = ['Mean', '99th', '1st', 'Day']
    statistics = {'Train': {}, 'Test': {}}
    
    for predictand_name in predictands:
        # Comienza figura
        modelName = f'DeepESD_tas_{predictand_name}' 
        pathPreds = PREDS_PATH if predictand_name != 'Ensemble' else f'{PREDS_PATH}ensemble/'
        loaded_data = xr.open_dataset(f'{PREDS_PATH}/predTrain_{modelName}_1.nc')
        current_data_mean = loaded_data.mean(dim='time')
        current_data_99 = (loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')).mean(dim='time')
        current_data_1 = (loaded_data.resample(time = 'YE').quantile(0.01, dim = 'time')).mean(dim='time')
        current_data_day = loaded_data.sel(time=days_selected[0])
        statistics['Train'][predictand_name] = {metrics[0]: current_data_mean, metrics[1]: current_data_99, metrics[2]: current_data_1, metrics[3]: current_data_day}


        loaded_data = xr.open_dataset(f'{PREDS_PATH}/predTest_{modelName}_1.nc')
        current_data_mean = loaded_data.mean(dim='time')
        current_data_99 = (loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')).mean(dim='time')
        current_data_1 = (loaded_data.resample(time = 'YE').quantile(0.01, dim = 'time')).mean(dim='time')
        current_data_day = loaded_data.sel(time=days_selected[1])
        statistics['Test'][predictand_name] = {metrics[0]: current_data_mean, metrics[1]: current_data_99, metrics[2]: current_data_1, metrics[3]: current_data_day}





    for period in ['Train', 'Test']:
        figName = f'fig0_Statistics_{period}'
        vminMetric = {'Mean': (5, 0.05, 1, 1), '1st': (-8, 0, 1, 1), '99th': (17, 0.25, 1, 2), 'Day': (17, 0.05, 1, 1)}
        vmaxMetric = {'Mean': (25, 0.85, 11, 11), '1st': (13, 0.5, 11, 14), '99th': (37, 1.25, 11, 12), 'Day': (37, 0.05, 11, 11)}
        # Crear la figura y los ejes
        fig, axes = plt.subplots(4, 6, figsize=(20, 12), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
        for i, (predictand_name, predictand_data) in enumerate(statistics[period].items()):
            for j, (metric, metric_data) in enumerate(predictand_data.items()):

                ax = axes[j, i]
                if j == 0:
                    ax.set_title(f'{predictands_map[predictand_name]}', fontsize=16)
                if i == 0:
                    ax.text(-0.07, 0.55, f'{metric}', va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize=16)

                #pos = 2 if metric=='mean' else 3
                num_ticks = vmaxMetric[metric][2] - vminMetric[metric][2]
                continuousCMAP = plt.get_cmap('hot_r') #if metric == 'mean' else plt.get_cmap('cool')
                discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, num_ticks+1)[vminMetric[metric][2]:vmaxMetric[metric][2]]))

                ax.coastlines(resolution='10m')
                

                dataToPlot = metric_data['tasmean']

                im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                                    dataToPlot,
                                    transform=ccrs.PlateCarree(),
                                    cmap=discreteCMAPnoWhite,
                                    vmin=vminMetric[metric][0], vmax=vmaxMetric[metric][0])
                                    #norm=BoundaryNorm(bounds, cmap.N))

                if i == 0:
                    cax = fig.add_axes([0.125, 0.741 - (j * 0.226), 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
                    cbar = plt.colorbar(im, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
                    ticks = np.linspace(vminMetric[metric][0], vmaxMetric[metric][0], int(np.floor(vmaxMetric[metric][2]+1-vminMetric[metric][2])))# if metric=='mean' else int(np.floor(vmaxMetric[metric][3]+1-vminMetric[metric][3])))
                    cbar.set_ticks(ticks)
                    cbar.ax.tick_params(labelsize=16)
                    tick_labels = [tick.get_text() for tick in cbar.ax.get_xticklabels()]
                    tick_labels[-1] += '+'
                    cbar.ax.set_xticklabels(tick_labels) 

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()

    print(f"Figura {FIG_NUM} Completada!")

# FIGURA CLIMATOLOGIA PARA 10 REALIAZCIONES
if FIG_NUM == '1':
    metrics = ['Mean', '1st', '99th', 'Day']
    statistics = {'Train': {f'{predictand_name}': {} for predictand_name in predictands}, 'Test': {f'{predictand_name}': {} for predictand_name in predictands}}
    for predictand_name in predictands:
        # Comienza figura 
        predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
        ensemble_train = {f'{metric}': [] for metric in metrics}
        ensemble_test = {f'{metric}': [] for metric in metrics}

        for predictand_number in predictand_numbered:
            modelName = f'DeepESD_tas_{predictand_number}' 
            pathPreds = PREDS_PATH if predictand_name != 'Ensemble' else f'{PREDS_PATH}ensemble/'
            loaded_data = xr.open_dataset(f'{pathPreds}/predTrain_{modelName}.nc')
            current_data_mean = loaded_data.mean(dim='time')
            current_data_99 = (loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')).mean(dim='time')
            current_data_1 = (loaded_data.resample(time = 'YE').quantile(0.01, dim = 'time')).mean(dim='time')
            current_data_day = loaded_data.sel(time=days_selected[0])
            ensemble_train[metrics[0]].append(current_data_mean)
            ensemble_train[metrics[1]].append(current_data_1)
            ensemble_train[metrics[2]].append(current_data_99)
            ensemble_train[metrics[3]].append(current_data_day)



            loaded_data = xr.open_dataset(f'{PREDS_PATH}/predTest_{modelName}.nc')
            current_data_mean = loaded_data.mean(dim='time')
            current_data_99 = (loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')).mean(dim='time')
            current_data_1 = (loaded_data.resample(time = 'YE').quantile(0.01, dim = 'time')).mean(dim='time')
            current_data_day = loaded_data.sel(time=days_selected[1])
            ensemble_test[metrics[0]].append(current_data_mean)
            ensemble_test[metrics[1]].append(current_data_1)
            ensemble_test[metrics[2]].append(current_data_99)
            ensemble_test[metrics[3]].append(current_data_day)

        for i, metric in enumerate(metrics):
            ensemble_metric_train = (xr.concat(ensemble_train[metric], dim='member')).mean(dim='member')
            ensemble_metric_test = (xr.concat(ensemble_test[metric], dim='member')).mean(dim='member')

            statistics['Train'][predictand_name][metrics[i]] = ensemble_metric_train
            statistics['Test'][predictand_name][metrics[i]] = ensemble_metric_test

        ensemble_train = {f'{metric}': [] for metric in metrics}
        ensemble_test = {f'{metric}': [] for metric in metrics}





    for period in ['Train', 'Test']:
        figName = f'fig1_Statistics_{ENSEMBLE_QUANTITY}_{period}'
        vminMetric = {'Mean': (5, 0.05, 1, 1), '1st': (-10, 0, 1, 1), '99th': (15, 0.25, 1, 2), 'Day': (15, 0.05, 1, 1)}
        vmaxMetric = {'Mean': (25, 0.85, 11, 11), '1st': (10, 0.5, 11, 14), '99th': (35, 1.25, 11, 12), 'Day': (35, 0.05, 11, 11)}
        # Crear la figura y los ejes
        fig, axes = plt.subplots(4, 6, figsize=(20, 12), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
        for i, (predictand_name, predictand_data) in enumerate(statistics[period].items()):
            for j, (metric, metric_data) in enumerate(predictand_data.items()):

                ax = axes[j, i]
                if j == 0:
                    ax.set_title(f'{predictands_map[predictand_name]}', fontsize=16)
                if i == 0:
                    ax.text(-0.07, 0.55, f'{metric}', va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize=16)

                #pos = 2 if metric=='mean' else 3
                num_ticks = vmaxMetric[metric][2] - vminMetric[metric][2]
                continuousCMAP = plt.get_cmap('hot_r') #if metric == 'mean' else plt.get_cmap('cool')
                discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, num_ticks+1)[vminMetric[metric][2]:vmaxMetric[metric][2]]))

                ax.coastlines(resolution='10m')
                

                dataToPlot = metric_data['tasmean']

                im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                                    dataToPlot,
                                    transform=ccrs.PlateCarree(),
                                    cmap=discreteCMAPnoWhite,
                                    vmin=vminMetric[metric][0], vmax=vmaxMetric[metric][0])
                                    #norm=BoundaryNorm(bounds, cmap.N))

                if i == 0:
                    cax = fig.add_axes([0.125, 0.741 - (j * 0.226), 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
                    cbar = plt.colorbar(im, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
                    ticks = np.linspace(vminMetric[metric][0], vmaxMetric[metric][0], int(np.floor(vmaxMetric[metric][2]+1-vminMetric[metric][2])))# if metric=='mean' else int(np.floor(vmaxMetric[metric][3]+1-vminMetric[metric][3])))
                    cbar.set_ticks(ticks)
                    cbar.ax.tick_params(labelsize=16)
                    tick_labels = [tick.get_text() for tick in cbar.ax.get_xticklabels()]
                    tick_labels[-1] += '+'
                    cbar.ax.set_xticklabels(tick_labels) 

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()
    print(f"Figura {FIG_NUM} Completada!")

# FIGURA SEÑAL DE CAMBIO PARA 10 REALIAZCIONES
if FIG_NUM == '2':
    predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'CHELSA' ]
    metrics = ['Mean', 'Std-Mean', 'Ensemble', 'Std-Ensemble']#, '99th', 'Std-99th']
    statistics = {'Train': {f'{predictand_name}': {} for predictand_name in predictands}, 'Test': {f'{predictand_name}': {} for predictand_name in predictands}}

    # Cargar datos de ensemble
    ensemble_data_train = {}
    ensemble_data_test = {}
    ensemble_numbered = [f"Ensemble_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]

    for ensemble_number in ensemble_numbered:
        modelName = f'DeepESD_tas_{ensemble_number}' 
        loaded_data = xr.open_dataset(f'{PREDS_PATH}ensemble/predTrain_{modelName}.nc')
        ensemble_data_train[ensemble_number] = loaded_data
        loaded_data = xr.open_dataset(f'{PREDS_PATH}ensemble/predTest_{modelName}.nc')
        ensemble_data_test[ensemble_number] = loaded_data


    for predictand_name in predictands:
        print(f"{predictand_name}")
        pathPreds = PREDS_PATH if predictand_name != 'Ensemble' else f'{PREDS_PATH}ensemble/'
        # Historical Data
        obs_predictand = utils.getPredictand(f'{DATA_PATH_PREDICTANDS_SAVE}', predictand_name, 'tasmean')

        obs_temp = obs_predictand.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
        obs_temp = obs_temp.sel(time=~obs_temp.time.isin(fechas_a_eliminar))
        obs_train = utils.maskData(
                    path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                    var='tasmean',
                    to_slice=(yearsTrain[0], yearsTest[1]),
                    objective = obs_predictand.sel(time=slice(*(yearsTrain[0], yearsTrain[1]))),
                    secondGrid = obs_temp)
        #obs_train_99 = obs_train.resample(time = 'YE').quantile(0.99, dim = 'time').mean(dim='time')
        obs_train_mean = obs_train.mean(dim='time')

        obs_test = utils.maskData(
                    path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                    var='tasmean',
                    to_slice=(yearsTrain[0], yearsTest[1]),
                    objective = obs_predictand.sel(time=slice(*(yearsTest[0], yearsTest[1]))),
                    secondGrid = obs_temp)
        #obs_test_99 = obs_test.resample(time = 'YE').quantile(0.99, dim = 'time').mean(dim='time')
        obs_test_mean = obs_test.mean(dim='time')

        predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
        ensemble_train = {f'{metric}': [] for metric in metrics}
        ensemble_test = {f'{metric}': [] for metric in metrics}

        for predictand_number in predictand_numbered:
            modelName = f'DeepESD_tas_{predictand_number}' 

            loaded_train = xr.open_dataset(f'{pathPreds}/predTrain_{modelName}.nc')
            current_data_mean = loaded_train.mean(dim='time') - obs_train_mean
            #current_data_99 = (loaded_train.resample(time = 'YE').quantile(0.99, dim = 'time')).mean(dim='time') - obs_train_99
            ensemble_train[metrics[0]].append(current_data_mean)
            #ensemble_train[metrics[2]].append(current_data_99)


            loaded_test = xr.open_dataset(f'{pathPreds}/predTest_{modelName}.nc')
            current_data_mean = loaded_test.mean(dim='time') - obs_test_mean
            #current_data_99 = (loaded_test.resample(time = 'YE').quantile(0.99, dim = 'time')).mean(dim='time') - obs_test_99
            ensemble_test[metrics[0]].append(current_data_mean)
            #ensemble_test[metrics[2]].append(current_data_99)

        for ensemble_number in ensemble_numbered:
            current_data_mean = ensemble_data_train[ensemble_number].mean(dim='time') - obs_train_mean
            ensemble_train[metrics[2]].append(current_data_mean)
            current_data_mean = ensemble_data_test[ensemble_number].mean(dim='time') - obs_test_mean
            ensemble_test[metrics[2]].append(current_data_mean)

        for i, metric in enumerate(['Mean', 'Ensemble']):
            ensemble_metric_train = xr.concat(ensemble_train[metric], dim='member')
            ensemble_metric_test = xr.concat(ensemble_test[metric], dim='member')
            print(f"ENSEMBLE METRIC TRAIN {metric}")
            print(ensemble_metric_train)

            statistics['Train'][predictand_name][metric] = ensemble_metric_train.mean(dim='member')
            statistics['Train'][predictand_name][f'Std-{metric}'] = ensemble_metric_train.std(dim='member')

            statistics['Test'][predictand_name][metric] = ensemble_metric_test.mean(dim='member')
            statistics['Test'][predictand_name][f'Std-{metric}'] = ensemble_metric_test.std(dim='member')

        ensemble_train = {f'{metric}': [] for metric in metrics}
        ensemble_test = {f'{metric}': [] for metric in metrics}

        #statistics['Train'][predictand_name] = {metrics[0]: current_data_mean, metrics[1]: current_data_99, metrics[2]: current_data_1, metrics[3]: current_data_day}
        #statistics['Test'][predictand_name] = {metrics[0]: current_data_mean, metrics[1]: current_data_99, metrics[2]: current_data_1, metrics[3]: current_data_day}





    for period in ['Train', 'Test']:
        figName = f'fig{FIG_NUM}_CCSignal_Mean_{ENSEMBLE_QUANTITY}_{period}'
        vminMetric = {'Mean': (-2.0, 0.05, 1, 1), 'Std-Mean': (0, 0, 0, 1), 'Ensemble': (-2, 0.25, 1, 2), 'Std-Ensemble': (0, 0.05, 0, 1)}
        vmaxMetric = {'Mean': (3.0, 0.85, 21, 11), 'Std-Mean': (0.8, 0.5, 20, 14), 'Ensemble': (3, 1.25, 21, 12), 'Std-Ensemble': (0.8, 0.05, 20, 11)}
        # Crear la figura y los ejes
        fig, axes = plt.subplots(4, 5, figsize=(17, 12), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
        for i, (predictand_name, predictand_data) in enumerate(statistics[period].items()):
            for j, (metric, metric_data) in enumerate(predictand_data.items()):

                ax = axes[j, i]
                if j == 0:
                    ax.set_title(f'{predictands_map[predictand_name]}', fontsize=16)
                if i == 0:
                    ax.text(-0.07, 0.55, f'{metric}', va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize=16)

                #pos = 2 if metric=='mean' else 3
                num_ticks = vmaxMetric[metric][2] - vminMetric[metric][2]
                continuousCMAP = plt.get_cmap('hot_r') if 'Std' not in metric else plt.get_cmap('cool')
                discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, num_ticks+1)[vminMetric[metric][2]:vmaxMetric[metric][2]]))

                ax.coastlines(resolution='10m')
                

                dataToPlot = metric_data['tasmean']
                print(metric)
                print(dataToPlot)

                im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                                    dataToPlot,
                                    transform=ccrs.PlateCarree(),
                                    cmap=discreteCMAPnoWhite,
                                    vmin=vminMetric[metric][0], vmax=vmaxMetric[metric][0])
                                    #norm=BoundaryNorm(bounds, cmap.N))

                if i == 0:
                    cax = fig.add_axes([0.125, 0.741 - (j * 0.226), 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
                    cbar = plt.colorbar(im, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
                    ticks = np.linspace(vminMetric[metric][0], vmaxMetric[metric][0], int(np.floor(vmaxMetric[metric][2]+1-vminMetric[metric][2])))# if metric=='mean' else int(np.floor(vmaxMetric[metric][3]+1-vminMetric[metric][3])))
                    cbar.set_ticks(ticks)
                    cbar.ax.tick_params(labelsize=16)
                    tick_labels = [tick.get_text() for tick in cbar.ax.get_xticklabels()]
                    tick_labels[-1] += '+'
                    cbar.ax.set_xticklabels(tick_labels) 

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()
    print(f"Figura {FIG_NUM} Completada!")


# FIGURA SEÑAL DE CAMBIO PARA 10 REALIAZCIONES
if FIG_NUM == '3':
    predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'CHELSA' ]
    metrics = ['99th', 'Std-99th', 'Ensemble', 'Std-Ensemble']#, '99th', 'Std-99th']
    statistics = {'Train': {f'{predictand_name}': {} for predictand_name in predictands}, 'Test': {f'{predictand_name}': {} for predictand_name in predictands}}

    # Cargar datos de ensemble
    ensemble_data_train = {}
    ensemble_data_test = {}
    ensemble_numbered = [f"Ensemble_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]

    for ensemble_number in ensemble_numbered:
        modelName = f'DeepESD_tas_{ensemble_number}' 
        loaded_data = xr.open_dataset(f'{PREDS_PATH}ensemble/predTrain_{modelName}.nc')
        ensemble_data_train[ensemble_number] = loaded_data
        loaded_data = xr.open_dataset(f'{PREDS_PATH}ensemble/predTest_{modelName}.nc')
        ensemble_data_test[ensemble_number] = loaded_data


    for predictand_name in predictands:
        print(f"{predictand_name}")
        pathPreds = PREDS_PATH if predictand_name != 'Ensemble' else f'{PREDS_PATH}ensemble/'
        # Historical Data
        obs_predictand = utils.getPredictand(f'{DATA_PATH_PREDICTANDS_SAVE}', predictand_name, 'tasmean')

        obs_temp = obs_predictand.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
        obs_temp = obs_temp.sel(time=~obs_temp.time.isin(fechas_a_eliminar))
        obs_train = utils.maskData(
                    path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                    var='tasmean',
                    to_slice=(yearsTrain[0], yearsTest[1]),
                    objective = obs_predictand.sel(time=slice(*(yearsTrain[0], yearsTrain[1]))),
                    secondGrid = obs_temp)
        obs_train_99 = obs_train.resample(time = 'YE').quantile(0.99, dim = 'time').mean(dim='time')
        #obs_train_mean = obs_train.mean(dim='time')

        obs_test = utils.maskData(
                    path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                    var='tasmean',
                    to_slice=(yearsTrain[0], yearsTest[1]),
                    objective = obs_predictand.sel(time=slice(*(yearsTest[0], yearsTest[1]))),
                    secondGrid = obs_temp)
        obs_test_99 = obs_test.resample(time = 'YE').quantile(0.99, dim = 'time').mean(dim='time')
        #obs_test_mean = obs_test.mean(dim='time')

        predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
        ensemble_train = {f'{metric}': [] for metric in metrics}
        ensemble_test = {f'{metric}': [] for metric in metrics}

        for predictand_number in predictand_numbered:
            modelName = f'DeepESD_tas_{predictand_number}' 

            loaded_train = xr.open_dataset(f'{pathPreds}/predTrain_{modelName}.nc')
            #current_data_mean = loaded_train.mean(dim='time') - obs_train_mean
            current_data_99 = (loaded_train.resample(time = 'YE').quantile(0.99, dim = 'time')).mean(dim='time') - obs_train_99
            ensemble_train[metrics[0]].append(current_data_99)
            #ensemble_train[metrics[2]].append(current_data_99)


            loaded_test = xr.open_dataset(f'{pathPreds}/predTest_{modelName}.nc')
            #current_data_mean = loaded_test.mean(dim='time') - obs_test_mean
            current_data_99 = (loaded_test.resample(time = 'YE').quantile(0.99, dim = 'time')).mean(dim='time') - obs_test_99
            ensemble_test[metrics[0]].append(current_data_99)
            #ensemble_test[metrics[2]].append(current_data_99)

        for ensemble_number in ensemble_numbered:
            current_data_99 = ensemble_data_train[ensemble_number].resample(time = 'YE').quantile(0.99, dim = 'time').mean(dim='time') - obs_train_99
            ensemble_train[metrics[2]].append(current_data_99)
            current_data_99 = ensemble_data_test[ensemble_number].resample(time = 'YE').quantile(0.99, dim = 'time').mean(dim='time') - obs_test_99
            ensemble_test[metrics[2]].append(current_data_99)

        for i, metric in enumerate(['99th', 'Ensemble']):
            ensemble_metric_train = xr.concat(ensemble_train[metric], dim='member')
            ensemble_metric_test = xr.concat(ensemble_test[metric], dim='member')
            print(f"ENSEMBLE METRIC TRAIN {metric}")
            print(ensemble_metric_train)

            statistics['Train'][predictand_name][metric] = ensemble_metric_train.mean(dim='member')
            statistics['Train'][predictand_name][f'Std-{metric}'] = ensemble_metric_train.std(dim='member')

            statistics['Test'][predictand_name][metric] = ensemble_metric_test.mean(dim='member')
            statistics['Test'][predictand_name][f'Std-{metric}'] = ensemble_metric_test.std(dim='member')

        ensemble_train = {f'{metric}': [] for metric in metrics}
        ensemble_test = {f'{metric}': [] for metric in metrics}

        #statistics['Train'][predictand_name] = {metrics[0]: current_data_mean, metrics[1]: current_data_99, metrics[2]: current_data_1, metrics[3]: current_data_day}
        #statistics['Test'][predictand_name] = {metrics[0]: current_data_mean, metrics[1]: current_data_99, metrics[2]: current_data_1, metrics[3]: current_data_day}





    for period in ['Train', 'Test']:
        figName = f'fig{FIG_NUM}_CCSignal_99th_{ENSEMBLE_QUANTITY}_{period}'
        vminMetric = {'99th': (-2.0, 0.05, 1, 1), 'Std-99th': (0, 0, 0, 1), 'Ensemble': (-2, 0.25, 1, 2), 'Std-Ensemble': (0, 0.05, 0, 1)}
        vmaxMetric = {'99th': (3.0, 0.85, 21, 11), 'Std-99th': (0.8, 0.5, 20, 14), 'Ensemble': (3, 1.25, 21, 12), 'Std-Ensemble': (0.8, 0.05, 20, 11)}
        # Crear la figura y los ejes
        fig, axes = plt.subplots(4, 5, figsize=(17, 12), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
        for i, (predictand_name, predictand_data) in enumerate(statistics[period].items()):
            for j, (metric, metric_data) in enumerate(predictand_data.items()):

                ax = axes[j, i]
                if j == 0:
                    ax.set_title(f'{predictands_map[predictand_name]}', fontsize=16)
                if i == 0:
                    ax.text(-0.07, 0.55, f'{metric}', va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize=16)

                #pos = 2 if metric=='mean' else 3
                num_ticks = vmaxMetric[metric][2] - vminMetric[metric][2]
                continuousCMAP = plt.get_cmap('hot_r') if 'Std' not in metric else plt.get_cmap('cool')
                discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, num_ticks+1)[vminMetric[metric][2]:vmaxMetric[metric][2]]))

                ax.coastlines(resolution='10m')
                

                dataToPlot = metric_data['tasmean']
                print(metric)
                print(dataToPlot)

                im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                                    dataToPlot,
                                    transform=ccrs.PlateCarree(),
                                    cmap=discreteCMAPnoWhite,
                                    vmin=vminMetric[metric][0], vmax=vmaxMetric[metric][0])
                                    #norm=BoundaryNorm(bounds, cmap.N))

                if i == 0:
                    cax = fig.add_axes([0.125, 0.741 - (j * 0.226), 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
                    cbar = plt.colorbar(im, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
                    ticks = np.linspace(vminMetric[metric][0], vmaxMetric[metric][0], int(np.floor(vmaxMetric[metric][2]+1-vminMetric[metric][2])))# if metric=='mean' else int(np.floor(vmaxMetric[metric][3]+1-vminMetric[metric][3])))
                    cbar.set_ticks(ticks)
                    cbar.ax.tick_params(labelsize=16)
                    tick_labels = [tick.get_text() for tick in cbar.ax.get_xticklabels()]
                    tick_labels[-1] += '+'
                    cbar.ax.set_xticklabels(tick_labels) 

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()

    print(f"Figura {FIG_NUM} Completada!")
# FIGURA RMSE Y BIAS PARA 10 REALIZACIONES
def compare_datasets(ds1, ds2, var='tasmean'):
    # Alinear dimensiones
    ds1_a = ds1.transpose('time', 'lat', 'lon')
    ds2_a = ds2.transpose('time', 'lat', 'lon')

    report = []

    # 1) Coordenadas time
    t1, t2 = ds1_a.time.values, ds2_a.time.values
    if np.array_equal(t1, t2):
        report.append(f"✔️ Coordenada 'time' coincide ({t1.size} pasos).")
    else:
        diff1 = np.setdiff1d(t1, t2)
        diff2 = np.setdiff1d(t2, t1)
        report.append(f"❌ Coordenada 'time' difiere:")
        report.append(f"   - En ds1 y no en ds2: {diff1}")
        report.append(f"   - En ds2 y no en ds1: {diff2}")

    # 2) Coordenada lat
    lat1, lat2 = ds1_a.lat.values, ds2_a.lat.values
    if np.array_equal(lat1, lat2):
        report.append(f"✔️ Coordenada 'lat' coincide ({lat1.size} valores).")
    else:
        report.append(f"❌ Coordenada 'lat' difiere.")

    # 3) Coordenada lon
    lon1, lon2 = ds1_a.lon.values, ds2_a.lon.values
    if np.array_equal(lon1, lon2):
        report.append(f"✔️ Coordenada 'lon' coincide ({lon1.size} valores).")
    else:
        report.append(f"❌ Coordenada 'lon' difiere.")

    # 4) Valores de la variable
    arr1 = ds1_a[var].values
    arr2 = ds2_a[var].values
    # Calcula la máscara de desigualdad (tratando NaN como iguales)
    neq = (arr1 != arr2) & ~(np.isnan(arr1) & np.isnan(arr2))
    total = arr1.size
    n_diff = np.count_nonzero(neq)

    if n_diff == 0:
        report.append(f"✔️ Todos los valores de '{var}' coinciden (incluyendo NaN).")
    else:
        # Extrae algunos ejemplos de posiciones diferentes
        coords = np.argwhere(neq)
        sample = coords[:5]
        report.append(f"❌ Hay {n_diff} diferencias en '{var}' de un total de {total} elementos.")
        report.append(f"   Primeras 5 posiciones diferentes (time, lat, lon indices):\n   {sample}")

    # Imprime el reporte
    print("\n".join(report))

if FIG_NUM == '4':
    predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'CHELSA' ]
    metrics = ['RMSE', 'RMSE-Ensemble', 'Bias', 'Bias-Ensemble', 'Bias99', 'Bias99-Ensemble']#, 'Bias-99th']
    vminMetric = {'RMSE': (0, 0, 20), 'RMSE-Ensemble': (0, 0, 20), 'Bias': (0, 0, 20), 'Bias-Ensemble': (0, 0, 20), 'Bias99': (0, 0, 20), 'Bias99-Ensemble': (0, 0, 20)}#, 'Bias-99th': (0.0, 0, 20)}
    vmaxMetric = {'RMSE': (2.0, 20, 20), 'RMSE-Ensemble': (2, 20, 20), 'Bias': (2, 20, 20), 'Bias-Ensemble': (2, 20, 20), 'Bias99': (2, 20, 20), 'Bias99-Ensemble': (2, 20, 20)}#, 'Bias-99th': (2.0, 20, 20)}
    statistics = {'Test': {}}
    stat_realization_mean = {f'{metric}': {} for metric in metrics}

    # Cargar datos de ensemble
    ensemble_data = {}
    ensemble_numbered = [f"Ensemble_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]

    for ensemble_number in ensemble_numbered:
        modelName = f'DeepESD_tas_{ensemble_number}' 
        loaded_data = xr.open_dataset(f'{PREDS_PATH}ensemble/predTest_{modelName}.nc')
        ensemble_data[ensemble_number] = loaded_data

    # Cargar datos de cada predictando y comprar a ensemble y entre ellos y guardar
    for predictand_name in predictands:
        # Historical Data
        obs_predictand = utils.getPredictand(f'{DATA_PATH_PREDICTANDS_SAVE}', predictand_name, 'tasmean')
        obs_temp = obs_predictand.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
        obs_temp = obs_temp.sel(time=~obs_temp.time.isin(fechas_a_eliminar))
        # TEST
        # yTrainFlat = newMask.flatten(grid=obs_temp, var='tasmean')
        # yTrainFlat_array = utils.toArray(yTrainFlat)
        # yTrainFlat['tasmean'].values = yTrainFlat_array
        # yTrainUnflatten = newMask.unFlatten(grid=yTrainFlat, var='tasmean')
        # print("TEST DE MASCARA")
        # print(newMask)
        # print("Observaciones normales")
        # print(obs_temp)
        # print("Observaciones enmascaradas")
        # print(yTrainUnflatten)
        # #TEST VALORES Y COORDENADAS
        # compare_datasets(obs_temp, yTrainUnflatten, var='tasmean')
        # print("Segundo")
        # FIN DEL TEST
        obs_test = utils.maskData(
                    path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                    var='tasmean',
                    to_slice=(yearsTrain[0], yearsTest[1]),
                    objective = obs_predictand.sel(time=slice(*(yearsTest[0], yearsTest[1]))),
                    secondGrid = obs_temp)
        obs_test_99 = obs_test.resample(time = 'YE').quantile(0.99, dim = 'time').mean(dim='time')
        obs_test_mean = obs_test.mean(dim='time')

        predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]

        
        stat_realizations = {f'{stat}': [] for stat in metrics}
        for predictand_number in predictand_numbered:
            modelName = f'DeepESD_tas_{predictand_number}' 

            loaded_data = xr.open_dataset(f'{PREDS_PATH}/predTest_{modelName}.nc')

            rmse = np.sqrt((((loaded_data - obs_test)**2).mean(dim=['time'], skipna=True))['tasmean'])
            bias = np.abs(loaded_data.mean(['time'])['tasmean'] - obs_test.mean(['time'])['tasmean'])
            loaded_test_99 = loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')
            obs_test_99 = obs_test.resample(time = 'YE').quantile(0.99, dim = 'time')
            bias_99 = np.abs(loaded_test_99.mean(['time'])['tasmean'] - obs_test_99.mean(['time'])['tasmean'])

            stat_realizations['RMSE'].append(rmse)
            stat_realizations['Bias'].append(bias)
            stat_realizations['Bias99'].append(bias_99)

        for ensemble_number in ensemble_numbered:
            
            rmse = np.sqrt((((ensemble_data[ensemble_number] - obs_test)**2).mean(dim=['time']))['tasmean'])
            bias = np.abs(ensemble_data[ensemble_number].mean(['time'])['tasmean'] - obs_test.mean(['time'])['tasmean'])
            loaded_test_99 = ensemble_data[ensemble_number].resample(time = 'YE').quantile(0.99, dim = 'time')
            obs_test_99 = obs_test.resample(time = 'YE').quantile(0.99, dim = 'time')
            bias_99 = np.abs(loaded_test_99.mean(['time'])['tasmean'] - obs_test_99.mean(['time'])['tasmean'])

            stat_realizations['RMSE-Ensemble'].append(rmse)
            stat_realizations['Bias-Ensemble'].append(bias)
            stat_realizations['Bias99-Ensemble'].append(bias_99)

        for stat_name, realization_list in stat_realizations.items():
            stat_concat = xr.concat(realization_list, dim='member')
            stat_realization_mean[stat_name][predictand_name] = stat_concat.mean(dim='member')
    # Borrar datos no utiles
    # Hacer grafico


    

    figName = f'fig{FIG_NUM}_RMSE_{ENSEMBLE_QUANTITY}'
    fig, axes = plt.subplots(6, 5, figsize=(17, 18), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

    for i, (predictand_name) in enumerate(predictands):
        # INTERQUARTIL A MANO
        for j, (metric, stat_data) in enumerate(stat_realization_mean.items()):


            ax = axes[j, i]
            if j == 0:
                ax.set_title(f'{predictands_map[predictand_name]}', fontsize=16)
            if i == 0:
                ax.text(-0.07, 0.55, f'{metric}', va='bottom', ha='center',
                    rotation='vertical', rotation_mode='anchor',
                    transform=ax.transAxes, fontsize=16)
                
            continuousCMAP = plt.get_cmap('viridis_r')
            discreteCMAP = ListedColormap(continuousCMAP(np.linspace(0, 1, vmaxMetric[stat_name][2])[vminMetric[stat_name][1]:vmaxMetric[stat_name][1]]))

            print("Colores max general, min temporal, max temporal")
            print(vmaxMetric[stat_name][2], vminMetric[stat_name][1], vmaxMetric[stat_name][1])
            print(discreteCMAP.colors)
            print(len(discreteCMAP.colors))
            ax.coastlines(resolution='10m')
            

            dataToPlot = stat_data[predictand_name]

            im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                                dataToPlot,
                                transform=ccrs.PlateCarree(),
                                cmap=discreteCMAP,
                                vmin=vminMetric[stat_name][0], vmax=vmaxMetric[stat_name][0])
                                #norm=BoundaryNorm(bounds, cmap.N))

            if i == 0:
                print(f"{j}-{metric}")
                cax = fig.add_axes([0.125, 0.805 - (j * 0.150), 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
                cbar = plt.colorbar(im, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
                ticks = np.linspace(vminMetric[metric][0], vmaxMetric[metric][0], int(np.floor(vmaxMetric[metric][1]+1-vminMetric[metric][1])))# if metric=='mean' else int(np.floor(vmaxMetric[metric][3]+1-vminMetric[metric][3])))
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize=16)
                tick_labels = [tick.get_text() for tick in cbar.ax.get_xticklabels()]
                tick_labels[-1] += '+'
                cbar.ax.set_xticklabels(tick_labels) 

    

    plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
    plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
    plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
    plt.close()

    print(f"Figura {FIG_NUM} Completada!")


# FIGURA CLIMATOLOGIA PARA 10 REALIAZCIONES
if FIG_NUM == '5':
    metrics = ['Mean', 'Std-Mean', '99th', 'Std-99th']
    statistics = {'Train': {f'{predictand_name}': {} for predictand_name in predictands}, 'Test': {f'{predictand_name}': {} for predictand_name in predictands}}
    for predictand_name in predictands:      
        pathPreds = PREDS_PATH if predictand_name != 'Ensemble' else f'{PREDS_PATH}ensemble/'
        predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
        ensemble_train = {f'{metric}': [] for metric in metrics}
        ensemble_test = {f'{metric}': [] for metric in metrics}

        for predictand_number in predictand_numbered:
            modelName = f'DeepESD_tas_{predictand_number}' 

            loaded_train = xr.open_dataset(f'{pathPreds}/predTrain_{modelName}.nc')
            current_data_mean = loaded_train.mean(dim='time')
            current_data_99 = (loaded_train.resample(time = 'YE').quantile(0.99, dim = 'time')).mean(dim='time')
            ensemble_train[metrics[0]].append(current_data_mean)
            ensemble_train[metrics[2]].append(current_data_99)


            loaded_test = xr.open_dataset(f'{pathPreds}/predTest_{modelName}.nc')
            current_data_mean = loaded_test.mean(dim='time')
            current_data_99 = (loaded_test.resample(time = 'YE').quantile(0.99, dim = 'time')).mean(dim='time')
            ensemble_test[metrics[0]].append(current_data_mean)
            ensemble_test[metrics[2]].append(current_data_99)


        for i, metric in enumerate(['Mean', '99th']):
            ensemble_metric_train = xr.concat(ensemble_train[metric], dim='member')
            ensemble_metric_test = xr.concat(ensemble_test[metric], dim='member')
            print(f"ENSEMBLE METRIC TRAIN {metric}")
            print(ensemble_metric_train)

            statistics['Train'][predictand_name][metric] = ensemble_metric_train.mean(dim='member')
            statistics['Train'][predictand_name][f'Std-{metric}'] = ensemble_metric_train.std(dim='member')

            statistics['Test'][predictand_name][metric] = ensemble_metric_test.mean(dim='member')
            statistics['Test'][predictand_name][f'Std-{metric}'] = ensemble_metric_test.std(dim='member')

        ensemble_train = {f'{metric}': [] for metric in metrics}
        ensemble_test = {f'{metric}': [] for metric in metrics}

        #statistics['Train'][predictand_name] = {metrics[0]: current_data_mean, metrics[1]: current_data_99, metrics[2]: current_data_1, metrics[3]: current_data_day}
        #statistics['Test'][predictand_name] = {metrics[0]: current_data_mean, metrics[1]: current_data_99, metrics[2]: current_data_1, metrics[3]: current_data_day}





    for period in ['Train', 'Test']:
        figName = f'fig{FIG_NUM}_Climatology_{ENSEMBLE_QUANTITY}_{period}'
        vminMetric = {'Mean': (5, 0.05, 1, 1), 'Std-Mean': (0, 0, 0, 1), '99th': (17, 0.25, 1, 2), 'Std-99th': (0, 0.05, 0, 1)}
        vmaxMetric = {'Mean': (25.0, 0.85, 21, 11), 'Std-Mean': (0.7, 0.5, 14, 14), '99th': (37, 1.25, 21, 12), 'Std-99th': (0.7, 0.05, 14, 11)}
        # Crear la figura y los ejes
        fig, axes = plt.subplots(4, 6, figsize=(20, 12), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
        for i, (predictand_name, predictand_data) in enumerate(statistics[period].items()):
            for j, (metric, metric_data) in enumerate(predictand_data.items()):

                ax = axes[j, i]
                if j == 0:
                    ax.set_title(f'{predictands_map[predictand_name]}', fontsize=16)
                if i == 0:
                    ax.text(-0.07, 0.55, f'{metric}', va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize=16)

                #pos = 2 if metric=='mean' else 3
                num_ticks = vmaxMetric[metric][2] - vminMetric[metric][2]
                continuousCMAP = plt.get_cmap('hot_r') if 'Std' not in metric else plt.get_cmap('cool')
                discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, num_ticks+1)[vminMetric[metric][2]:vmaxMetric[metric][2]]))

                ax.coastlines(resolution='10m')
                

                dataToPlot = metric_data['tasmean']
                print(metric)
                print(dataToPlot)

                im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                                    dataToPlot,
                                    transform=ccrs.PlateCarree(),
                                    cmap=discreteCMAPnoWhite,
                                    vmin=vminMetric[metric][0], vmax=vmaxMetric[metric][0])
                                    #norm=BoundaryNorm(bounds, cmap.N))

                if i == 0:
                    cax = fig.add_axes([0.125, 0.741 - (j * 0.226), 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
                    cbar = plt.colorbar(im, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
                    ticks = np.linspace(vminMetric[metric][0], vmaxMetric[metric][0], int(np.floor(vmaxMetric[metric][2]+1-vminMetric[metric][2])))# if metric=='mean' else int(np.floor(vmaxMetric[metric][3]+1-vminMetric[metric][3])))
                    cbar.set_ticks(ticks)
                    cbar.ax.tick_params(labelsize=16)
                    tick_labels = [tick.get_text() for tick in cbar.ax.get_xticklabels()]
                    tick_labels[-1] += '+'
                    cbar.ax.set_xticklabels(tick_labels) 

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()

    print(f"Figura {FIG_NUM} Completada!")

# Histograma
if FIG_NUM == '6':
    predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'CHELSA', 'Ensemble']
    #predictands = ['ERA5-Land0.25deg', 'CHELSA', 'Ensemble']
    past_timeline = ('1970-01-01', '2020-12-31')
    hist_baseline = ('1995-01-01', '2014-12-31') #95-14
    future_1 = ('2021-01-01', '2040-12-31')
    future_2 = ('2041-01-01', '2060-12-31')
    future_3 = ('2081-01-01', '2100-12-31') 
    future_4 = ('2061-01-01', '2080-12-31')
    futures = [future_1, future_2, future_3, future_4]


    obs2 = {}
    obs_temp = {}
    gcm_preds = {}
    hist_gcm = {}
    hist_gcm_mean = {}
    hist_gcm_mean_flat = {}

    for predictand_name in predictands:        

        pathModel = f'{PREDS_PATH}GCM/AEMET/' if predictand_name != 'Ensemble' else f'{PREDS_PATH}ensemble/'
        hist_gcm_mean[predictand_name] = {}
        if predictand_name != 'Ensemble':
            obs2[predictand_name] = utils.getPredictand(f'{DATA_PATH_PREDICTANDS_SAVE}', predictand_name, 'tasmean')
            obs_temp[predictand_name] = obs2[predictand_name].sel(time=slice(*(yearsTrain[0], yearsTest[1])))
            obs2[predictand_name] = utils.maskData(
                        path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                        var='tasmean',
                        to_slice=(yearsTrain[0], yearsTest[1]),
                        objective = obs2[predictand_name].sel(time=slice(*(past_timeline[0], past_timeline[1]))),
                        secondGrid = obs_temp[predictand_name])
        
        predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
        print(predictand_name)
        print(pathModel)
        for predictand_number in predictand_numbered:
            gcms_futures = []
            modelName = f'DeepESD_tas_{predictand_number}' 
            
            for future in futures:
                print(f'{pathModel}predGCM_{modelName}_{GCM_NAME}_{main_scenario}_{future[0]}-{future[1]}.nc')
                archivo = xr.open_dataset(f'{pathModel}predGCM_{modelName}_{GCM_NAME}_{main_scenario}_{future[0]}-{future[1]}.nc')
                gcms_futures.append(archivo)
            gcm_preds[predictand_number] = xr.merge(gcms_futures)
            if predictand_name != 'Ensemble':
                hist_gcm[predictand_number] = xr.merge([obs2[predictand_name], gcm_preds[predictand_number]])
            else:
                hist_gcm[predictand_number] = xr.merge([gcm_preds[predictand_number]])
            hist_gcm_mean[predictand_name][predictand_number] = hist_gcm[predictand_number].resample(time = 'YE').mean()
            hist_gcm_mean[predictand_name][predictand_number] = hist_gcm_mean[predictand_name][predictand_number].mean(dim=['lat', 'lon'])#resample(time='1Y')

        hist_gcm_mean[predictand_name]['Mean'] = xr.concat(list(hist_gcm_mean[predictand_name].values()), dim='ensemble').mean(dim='ensemble')

    # Crear una figura y un conjunto de ejes
    figName = f'figdd{FIG_NUM}_histogramFull_{ENSEMBLE_QUANTITY}'
    plt.figure(figsize=(12, 8))

    # Generador de colores
    colors = cm.get_cmap('tab10', len(hist_gcm_mean))

    for idx, (predictand_name, predictand_data) in enumerate(hist_gcm_mean.items()):
        color = colors(idx)

        # Obtener la serie 'Mean'
        mean_data = predictand_data.get('Mean')
        if mean_data is None:
            continue  # Saltar si no hay 'Mean'

        years = mean_data['time'].dt.year
        tasmean_mean = mean_data['tasmean'].values

        # Graficar la línea de la media
        plt.plot(years, tasmean_mean, label=f"{predictand_name} - Mean", color=color)

        # Acumular datos de los miembros (excepto 'Mean')
        members = [v['tasmean'].values for k, v in predictand_data.items() if k != 'Mean']
        if members:
            members = np.array(members)  # shape: (n_members, n_years)

            tasmin = np.nanmin(members, axis=0)
            tasmax = np.nanmax(members, axis=0)

            # Rellenar el rango entre el mínimo y el máximo
            plt.fill_between(years, tasmin, tasmax, color=color, alpha=0.2, label=f"{predictand_name} - Range")
        if predictand_name == 'Ensemble':
            members = np.array(members)  # shape: (n_members, n_years)

            tasmin = np.nanmin(members, axis=0)
            tasmax = np.nanmax(members, axis=0)
            print("Members")
            print(members)
            print("MIN")
            print(tasmin)
            print("MAX")
            print(tasmax)

    # Línea vertical del año 2021
    plt.axvline(x=2021, color='r', linestyle='--', linewidth=1)

    # Configurar etiquetas y título
    plt.xlabel('Year')
    plt.ylabel('Tasmean')
    plt.legend(loc='best')

    # Guardar y cerrar
    plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
    plt.savefig(f'{FIGS_PATH}/{figName}.pdf', bbox_inches='tight')
    plt.close()

    print(f"Figura {FIG_NUM} Completada!")

# Std entre 5 datasets e interna de ensemble
if FIG_NUM == '7':
    predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'CHELSA']
    #predictands = ['ERA5-Land0.25deg', 'CHELSA', 'Ensemble']
    past_timeline = ('1970-01-01', '2020-12-31')
    hist_baseline = ('1995-01-01', '2014-12-31') #95-14
    future_1 = ('2021-01-01', '2040-12-31')
    future_2 = ('2041-01-01', '2060-12-31')
    future_3 = ('2081-01-01', '2100-12-31') 
    future_4 = ('2061-01-01', '2080-12-31')
    futures = [future_1, future_2, future_3, future_4]
    main_scenerio = 'ssp585'
    metrics = ['Mean', '99th', '1st']

    ensemble_numbered = [f"Ensemble_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
    for metric in metrics:
        gcm_data_list = []
        for ensemble_number in ensemble_numbered:
            modelName = f'DeepESD_tas_{ensemble_number}' 
            gcm_data_list_numbered = []
            # GCM
            for future in futures:
                loaded_pred = xr.open_dataset(f'/lustre/gmeteo/WORK/reyess/preds/ensemble/predGCM_{modelName}_{GCM_NAME}_ssp585_{future[0]}-{future[1]}.nc')
                
                if metric == '99th':
                    loaded_pred = loaded_pred.resample(time = 'YE').quantile(0.99, dim = 'time')
                elif metric == '1st':
                    loaded_pred = loaded_pred.resample(time = 'YE').quantile(0.01, dim = 'time')
                gcm_data_list_numbered.append(loaded_pred)
            concat_data = xr.merge(gcm_data_list_numbered)
            #concat_data = xr.concat(gcm_data_list_numbered, dim='time')
            gcm_data_list.append((concat_data.resample(time = 'YE').mean()).mean(dim=['lat', 'lon']))
        gcm_data_mean = xr.concat(gcm_data_list, dim='ensemble').mean(dim='ensemble')

        # DATASETS 
        #gcms_futures = []
        dataset_list = {}
        for predictand_name in predictands:
            dataset_list[predictand_name] = []
            predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
            for predictand_number in predictand_numbered:
                gcms_futures = []
                modelName = f'DeepESD_tas_{predictand_number}' 
                
                for future in futures:
                    archivo = xr.open_dataset(f'{PREDS_PATH}/GCM/AEMET/predGCM_{modelName}_{GCM_NAME}_{main_scenario}_{future[0]}-{future[1]}.nc')
                    if metric == '99th':
                        archivo = archivo.resample(time = 'YE').quantile(0.99, dim = 'time')
                    elif metric == '1st':
                        archivo = archivo.resample(time = 'YE').quantile(0.01, dim = 'time')
                    gcms_futures.append(archivo)
                concat_data = xr.merge(gcms_futures)
                #concat_data = xr.concat(gcms_futures, dim='time')
                dataset_list[predictand_name].append((concat_data.resample(time = 'YE').mean()).mean(dim=['lat', 'lon']))
            #dataset_list[predictand_name]['Mean'] = xr.concat(dataset_list[predictand_name].values(), dim='ensemble').mean(dim='ensemble')

            #hist_gcm_mean[predictand_name]['Mean'] = xr.concat(list(hist_gcm_mean[predictand_name].values()), dim='ensemble').mean(dim='ensemble')
        for predictand_name in predictands:
            figName = f'fig{FIG_NUM}_comparisson_{ENSEMBLE_QUANTITY}_{metric}_{predictand_name}'
            # Plotting
            fig, ax = plt.subplots(figsize=(12, 9))

            years = gcm_data_mean['time'].dt.year
            #print(gcm_data_mean) # 351973 REVISAR ESTO
            tasmean_mean = gcm_data_mean['tasmean'].values
            # print("TASMEAN")
            # print(tasmean_mean)
            # print("gcm data list [0]")
            # print(gcm_data_list[0]['tasmean'].values)
            # print("datasetlist")
            # print(dataset_list[predictand_name][0]['tasmean'].values)
            # Acumular datos de los miembros (excepto 'Mean')
            ensemble_members = [v['tasmean'].values - tasmean_mean for v in gcm_data_list]
            
            if ensemble_members:
                ensemble_members = np.array(ensemble_members)  # shape: (n_members, n_years)

                tasmin = np.nanmin(ensemble_members, axis=0)
                tasmax = np.nanmax(ensemble_members, axis=0)
                # print("tasmin")
                # print(tasmin)
                # print("tasmax")
                # print(tasmax)
                # Rellenar el rango entre el mínimo y el máximo
                plt.fill_between(years, tasmin, tasmax, color='Red', alpha=0.2, label=f"Ensemble - Range")

            dataset_members = [v['tasmean'].values - tasmean_mean for v in dataset_list[predictand_name]]
            print(dataset_members)
            if dataset_members:
                dataset_members = np.array(dataset_members)  # shape: (n_members, n_years)

                tasmin = np.nanmin(dataset_members, axis=0)
                tasmax = np.nanmax(dataset_members, axis=0)

                # print("tasmin")
                # print(tasmin)
                # print("tasmax")
                # print(tasmax)
                plt.fill_between(years, tasmin, tasmax, color='Blue', alpha=0.2, label=f"{predictand_name} - Range")

            # Configurar etiquetas y título
            plt.xlabel('Year')
            plt.ylabel('Tasmean')
            plt.legend(loc='best')

            # Guardar y cerrar
            plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
            plt.savefig(f'{FIGS_PATH}/{figName}.pdf', bbox_inches='tight')
            plt.close()

    print(f"Figura {FIG_NUM} Completada!")


### # LONG VS TEST TEMPERATURE SCATTER PLOT# ####
if FIG_NUM == '8':
    valuesMinMax = {'Mean': (13, 15, 19, 21), '99Percentile': (25, 28, 33, 36), '1Percentile': (0.75, 2.75, 5.75, 7.75)}
    predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'CHELSA']
    for metric in ['Mean', '99Percentile', '1Percentile']:
        observational_mean = {}
        test_pred_total = {}
        gcm_pred_total = {}
        color_list = ['crimson', 'forestgreen', 'royalblue', 'orchid', 'cadetblue', 'dimgray']
        color_num = 0


        ensemble_data_gcm = []
        ensemble_data_test = []
        ensemble_numbered = [f"Ensemble_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]

        for ensemble_number in ensemble_numbered:
            modelName = f'DeepESD_tas_{ensemble_number}' 
            # TEST
            loaded_pred = xr.open_dataset(f'/lustre/gmeteo/WORK/reyess/preds/ensemble/predTest_{modelName}.nc')
            if metric == '99Percentile':
                loaded_pred = loaded_pred.resample(time = 'YE').quantile(0.99, dim = 'time')
            elif metric == '1Percentile':
                loaded_pred = loaded_pred.resample(time = 'YE').quantile(0.01, dim = 'time')
            ensemble_data_test.append(loaded_pred.mean(dim=['time', 'lat', 'lon'])['tasmean'])
            # GCM
            loaded_pred = xr.open_dataset(f'/lustre/gmeteo/WORK/reyess/preds/ensemble/predGCM_{modelName}_{GCM_NAME}_ssp585_{future_3[0]}-{future_3[1]}.nc')
            #ensemble_data_gcm[ensemble_number] = loaded_data
            if metric == '99Percentile':
                loaded_pred = loaded_pred.resample(time = 'YE').quantile(0.99, dim = 'time')
            elif metric == '1Percentile':
                loaded_pred = loaded_pred.resample(time = 'YE').quantile(0.01, dim = 'time')
            ensemble_data_gcm.append(loaded_pred.mean(dim=['time', 'lat', 'lon'])['tasmean'])


        for predictand_name in predictands:

            rmse_test = []
            test_pred = []
            gcm_pred = []

            modelName = f'DeepESD_tas_{predictand_name}' 
            loaded_test_obs = utils.getPredictand(DATA_PATH_PREDICTANDS_SAVE, predictand_name, 'tasmean')
            loaded_test_obs = loaded_test_obs.sel(time=slice(*(yearsTest[0], yearsTest[1])))
            loaded_test_obs = utils.maskData(
                        path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                        var='tasmean',
                        to_slice=(yearsTrain[0], yearsTest[1]),
                        objective = loaded_test_obs,
                        secondGrid = loaded_test_obs)
            
            if metric == '99Percentile':
                observed_mean = loaded_test_obs.resample(time = 'YE').quantile(0.99, dim = 'time')
            elif metric == '1Percentile':
                observed_mean = loaded_test_obs.resample(time = 'YE').quantile(0.01, dim = 'time')
            else:
                observed_mean = loaded_test_obs

            observed_mean = observed_mean.mean(dim=['time', 'lat', 'lon'])['tasmean']
            observational_mean[predictands_map[predictand_name]] = observed_mean


            predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
            for predictand_number in predictand_numbered:
                modelName = f'DeepESD_tas_{predictand_number}' 
                loaded_test = xr.open_dataset(f'{PREDS_PATH}predTest_{modelName}.nc')

                if metric == '99Percentile':
                    loaded_test = loaded_test.resample(time = 'YE').quantile(0.99, dim = 'time')
                elif metric == '1Percentile':
                    loaded_test = loaded_test.resample(time = 'YE').quantile(0.01, dim = 'time')
                test_pred.append(loaded_test.mean(dim=['time', 'lat', 'lon'])['tasmean'])

                
                loaded_pred = xr.open_dataset(f'{PREDS_PATH}/GCM/AEMET/predGCM_{modelName}_{GCM_NAME}_{main_scenario}_{future_3[0]}-{future_3[1]}.nc')
                if metric == '99Percentile':
                    loaded_pred = loaded_pred.resample(time = 'YE').quantile(0.99, dim = 'time')
                elif metric == '1Percentile':
                    loaded_pred = loaded_pred.resample(time = 'YE').quantile(0.01, dim = 'time')
                gcm_pred.append(loaded_pred.mean(dim=['time', 'lat', 'lon'])['tasmean'])


            test_pred_total[predictands_map[predictand_name]] = test_pred
            gcm_pred_total[predictands_map[predictand_name]] = gcm_pred



        figName = f'fig{FIG_NUM}_temps_Ensemble{ENSEMBLE_QUANTITY}_{metric}'
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 9))

        # Scatter for dataset 1
        color_num = 0
        print(f"Observational mean: {metric}")
        print(observational_mean)
        
        # Datasets individuales
        for predictand_name in predictands:
            test_values = test_pred_total[predictands_map[predictand_name]]
            gcm_values = gcm_pred_total[predictands_map[predictand_name]]
            ax.scatter(test_values, gcm_values, color=f'{color_list[color_num]}', label=f'{predictands_map[predictand_name]}', alpha=0.7)
            ax.plot([valuesMinMax[metric][0], valuesMinMax[metric][2]], [valuesMinMax[metric][1], valuesMinMax[metric][3]], color='black', linestyle='-', linewidth=1.5)
            ax.axvline(observational_mean[predictands_map[predictand_name]], color=f'{color_list[color_num]}', linestyle='--', linewidth=1)
            #print(f"{predictands_map[predictand_name]}-{metric}: {test_values}")
            test_mean = np.mean([da.data for da in test_values])
            gcm_mean = np.mean([da.data for da in gcm_values])
            ax.scatter(test_mean, gcm_mean, color=f'{color_list[color_num]}', marker='+', s=100)
            color_num += 1

        # Ensemble

        ax.scatter(ensemble_data_test, ensemble_data_gcm, color=f'{color_list[color_num]}', label=f'Ensemble', alpha=0.7)
        ax.plot([valuesMinMax[metric][0], valuesMinMax[metric][2]], [valuesMinMax[metric][1], valuesMinMax[metric][3]], color='black', linestyle='-', linewidth=1.5)
        ensemble_test_mean = np.mean([da.data for da in ensemble_data_test])
        ensemble_gcm_mean = np.mean([da.data for da in ensemble_data_gcm])
        ax.scatter(ensemble_test_mean, ensemble_gcm_mean, color=f'{color_list[color_num]}', marker='+', s=100)
        color_num += 1

        # Labels and legend
        ax.set_xlim(valuesMinMax[metric][0], valuesMinMax[metric][1])
        ax.set_ylim(valuesMinMax[metric][2],valuesMinMax[metric][3])
        ax.set_xlabel('Temperature Test', fontsize=12)
        ax.set_ylabel('Temperature Long', fontsize=12)
        ax.set_title('Long vs Test', fontsize=14)
        ax.legend()


        # Show grid for better readability
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()



    print(f"Figura {FIG_NUM} completada!")
    

if FIG_NUM == '9':
    SHAPE_NAME = ['Iberia', 'Pirineos', 'Tinto', 'Duero']
    def get_dataset(tagus):# Suponiendo que tagus es un objeto de tipo Polygon
        tagus_coords = np.array(tagus.exterior.coords)

        # Extraer las coordenadas de latitud y longitud
        lon = tagus_coords[:, 0]
        lat = tagus_coords[:, 1]

        # Crear el Dataset
        tagus_dataset = xr.Dataset(
            coords={
                'lat': lat,
                'lon': lon
            }
        )
        return tagus_dataset


    def find_nearest(values, ref_array, var):
        nearest_values = []
        
        # Calcular los valores más cercanos manteniendo la forma original
        for value in values:
            nearest_value = ref_array[np.argmin(np.abs(ref_array - value.item()))]
            nearest_values.append(nearest_value)
        
        # Convertir a un numpy array con la misma forma que 'values'
        nearest_values = np.array(nearest_values).reshape(values.shape)
        
        # Eliminar duplicados mientras se mantiene el orden
        _, idx = np.unique(nearest_values, return_index=True)
        nearest_values_unique = nearest_values.flatten()[np.sort(idx)]
        
        filtered_nearest_values = xr.DataArray(
            nearest_values_unique,  # Acceso correcto usando diccionario
            coords={var: nearest_values_unique},        # Asignación de coordenadas
            dims=[var]                                 # Dimensión correcta
        )

        # Retornar el array con valores únicos
        return filtered_nearest_values


    def filter_closest_coords(tagus_dataset, obs2):
    # Obtener las coordenadas de latitud y longitud de obs2
        obs2_lat = obs2['lat'].values
        obs2_lon = obs2['lon'].values
        # Encuentra las coordenadas más cercanas en new_dataset
        lat_indices = []
        lon_indices = []
        for lat in obs2_lat:
            closest_lat_idx = np.argmin(np.abs(tagus_dataset['lat'].values - lat))
            lat_indices.append(closest_lat_idx)
        for lon in obs2_lon:
            closest_lon_idx = np.argmin(np.abs(tagus_dataset['lon'].values - lon))
            lon_indices.append(closest_lon_idx)
        # Obtener los índices únicos de latitud y longitud
        unique_lat_indices = np.unique(lat_indices)
        unique_lon_indices = np.unique(lon_indices)
        # Filtrar los datos en new_dataset
        filtered_lat = tagus_dataset['lat'].isel(lat=unique_lat_indices)
        filtered_lat = filtered_lat.sortby(filtered_lat)
        filtered_lon = tagus_dataset['lon'].isel(lon=unique_lon_indices)
        filtered_lon = filtered_lon.sortby(filtered_lon)
        tagus_new_lat = find_nearest(filtered_lat, obs2_lat, 'lat')
        tagus_new_lon = find_nearest(filtered_lon, obs2_lon, 'lon')
        num_times = 1  # Por ejemplo, el mismo número que 'time' en obs2
        tasmean_data = np.full((num_times, len(tagus_new_lat), len(tagus_new_lon)), np.nan)  # Llenar con NaN
        times = np.arange(np.datetime64('1970-01-01'), np.datetime64('1970-01-01') + np.timedelta64(num_times, 'D'))
        new_filtered_dataset = xr.Dataset(
            {
                'tasmean': (('time', 'lat', 'lon'), tasmean_data)
            },
            {'lat':tagus_new_lat, 'lon':tagus_new_lon, 'time': times})

        return new_filtered_dataset

    def create_coords_dataset(lats, lons, reference):
        new_filtered_dataset = reference.sel(lon=slice(lons[0], lons[1]), lat=slice(lats[0], lats[1]))
        return new_filtered_dataset
    #*********************************************************************+
    references_grid = {shape: [] for shape in SHAPE_NAME}
    shape_name_fig = ''
    reference_grid = xr.open_dataset(f'{PREDS_PATH}/predGCM_DeepESD_tas_AEMET_0.25deg_1_{GCM_NAME}_{MAIN_SCENARIO}_{future_1[0]}-{future_1[1]}.nc')
    reference_grid = reference_grid.sel(time=slice(future_1[0],'2021-01-02'))
    for shape in SHAPE_NAME:
        if shape == 'Tagus':
            shape_file = gpd.read_file(f'{DATA_PATH_SHAPE}{shape_file_path}')
            gdf = gpd.GeoDataFrame(shape_file)
            tagus = gdf[gdf['NAME'] == shape]['geometry'].values[0]
            tagus_dataset = get_dataset(tagus)
            reference_grid_modified = filter_closest_coords(tagus_dataset, reference_grid)
            references_grid[shape] = reference_grid_modified
        elif shape == 'Iberia':
            references_grid[shape] = None
        elif shape == 'Tagus2':
            shape_file = gpd.read_file(f'{DATA_PATH_SHAPE}{shape_file_path}')
            gdf = gpd.GeoDataFrame(shape_file)
            tagus = gdf[gdf['NAME'] == 'Tagus']['geometry'].values[0]
            tagus_dataset = get_dataset(tagus)
            reference_grid_modified = filter_closest_coords(tagus_dataset, reference_grid)
            references_grid[shape] = reference_grid_modified
        elif shape == 'Ebro':
            shape_file = gpd.read_file(f'{DATA_PATH_SHAPE}{shape_file_path_major}')
            gdf = gpd.GeoDataFrame(shape_file)
            tagus = gdf[gdf['NAME'] == shape]['geometry'].values[0]
            tagus_dataset = get_dataset(tagus)
            reference_grid_modified = filter_closest_coords(tagus_dataset, reference_grid)
            references_grid[shape] = reference_grid_modified
        elif shape == 'Pirineos':
            references_grid[shape] = create_coords_dataset(lons=(-0.37, 3.37), lats=(41.42, 42.80), reference=reference_grid)
        elif shape == 'Duero':
            references_grid[shape] = create_coords_dataset(lons=(-6.59, -4.75), lats=(40.85, 42.45), reference=reference_grid)
        elif shape == 'Tinto':
            references_grid[shape] = create_coords_dataset(lons=(-7.23, -5.20), lats=(36.00, 38.20), reference=reference_grid)

        shape_name_fig = f'{shape_name_fig}_{shape}'

    # GRAFICOS BOXPLOT PARA SHORT, MEDIUM y LONG / CCSIGNAL
    periods = [future_2, future_4, future_3]
    xmin = (2, 1.5)
    xmax = (11, 11.5)
    # CC SIGNAL
    colors = ['darkgreen', 'darkblue', 'darkred']
    names = ['Short', 'Medium', 'Long']
    for shape in SHAPE_NAME:
        legend_handles = []
        figName = f'fig{FIG_NUM}_boxPlot_ccsignals_Ensemble{ENSEMBLE_QUANTITY}_{shape}'
        # Crear la figura y los ejes
        fig, ax1 = plt.subplots(figsize=(20, 12))
        ######################################
        ###BOXPLOT GENERALIZADO####
        ##########################################################3
        # Graficar cada set de datos (Short, Medium, Long) en el mismo gráfico
        for i, period in enumerate(periods):
            data_to_plot = []
            data_to_plot_99 = []
            for predictand_name in predictands:
                obs2 = utils.getPredictand(f'{DATA_PATH_PREDICTANDS_SAVE}', predictand_name, 'tasmean')
                obs_temp = obs2.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
                obs2 = utils.maskData(
                            path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                            var='tasmean',
                            to_slice=(hist_baseline[0], hist_baseline[1]),
                            objective = obs2.sel(time=slice(*(hist_baseline[0], hist_baseline[1]))),
                            secondGrid = obs_temp)
                obs2 = obs2.sel(
                        lat=references_grid[shape].lat,
                        lon=references_grid[shape].lon,
                    ) if shape != 'Iberia' else obs2
                obs2_99 = obs2.resample(time = 'YE').quantile(0.99, dim = 'time')

                obs2_mean = obs2.mean(dim=['time', 'lat', 'lon'])
                obs2_99_mean = obs2_99.mean(dim=['time', 'lat', 'lon'])


                predictand_data = []
                ccsignal_predictand = []
                ccsignal_predictand_99 = []
                predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]

                for predictand_number in predictand_numbered:
                    modelName = f'DeepESD_tas_{predictand_number}' 
                    loaded_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{period[0]}-{period[1]}.nc')
                    grided_data = loaded_data.sel(
                        lat=references_grid[shape].lat,
                        lon=references_grid[shape].lon,
                    ) if shape != 'Iberia' else loaded_data

                    grided_data_99 = grided_data.resample(time = 'YE').quantile(0.99, dim = 'time')
                    grided_mean = grided_data.mean(dim=['time', 'lat', 'lon']) 
                    grided_mean_99 = grided_data_99.mean(dim=['time', 'lat', 'lon'])
                    ccsignal_predictand.append(grided_mean- obs2_mean)
                    ccsignal_predictand_99.append(grided_mean_99 - obs2_99_mean)


                ccsignal_array = np.array([ds['tasmean'].values for ds in ccsignal_predictand])
                data_to_plot.append(ccsignal_array)
                ccsignal_array_99 = np.array([ds['tasmean'].values for ds in ccsignal_predictand_99])
                data_to_plot_99.append(ccsignal_array_99)

            ax = ax1.twiny() if i > 0 else ax1  # Crear ejes adicionales solo para Medium y Long
            color = colors[i]
            bplot = ax.boxplot(data_to_plot, positions= 5 + np.arange(len(predictands)), widths=0.35, 
                            patch_artist=True, boxprops=dict(facecolor=color), vert=False, whis=[5, 95],
                            whiskerprops=dict(color=color), flierprops=dict(color=color, markeredgecolor=color),
                            medianprops=dict(color='snow', linewidth=2))
            ax.set_xlim(xmin[1], xmax[1]-2)
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.tick_top()
            ax.set_xticks(np.linspace(xmin[0], xmax[0]-2, 8))
            ax.tick_params(axis='x', labelsize=16)
 

            ax_99 = ax1.twiny() if i > 0 else ax1  # Crear ejes adicionales solo para Medium y Long
            bplot = ax_99.boxplot(data_to_plot_99, positions=np.arange(len(predictands))+0.1 , widths=0.35, 
                            patch_artist=True, boxprops=dict(facecolor=color), vert=False, whis=[5, 95],
                            whiskerprops=dict(color=color), flierprops=dict(color=color, markeredgecolor=color),
                            medianprops=dict(color='snow', linewidth=1.5))
            ax_99.set_xlim(xmin[1], xmax[1])
            ax_99.xaxis.set_ticks_position('bottom')
            ax_99.set_xticks([]) if i>0 else ax_99.set_xticks(np.linspace(xmin[0], xmax[0], 10))
            ax_99.tick_params(axis='x', labelsize=16)

            
            
            # Asignar la etiqueta del eje X solo para el primer eje (ax1)
            if i == 0:
                ax.set_xlabel(f'CC Signal Tasmean {shape}')
            # Crear un handle de la leyenda solo en la primera iteración para cada conjunto de datos
            legend_handles.append(bplot["boxes"][0])


        # Etiquetas del eje Y solo en ax1
        ax1.set_yticks(np.arange(len(predictands)*2) )
        ax1.set_yticklabels(list(predictands_map.values())*2, fontsize=16)

        # Calcular el centro del gráfico
        y_min, y_max = ax1.get_ylim()
        y_center = (y_min + y_max) / 2
        # Dibujar una línea horizontal
        ax1.hlines(y=y_center, xmin=xmin[1], xmax=xmax[1], colors='black', linestyles='dashed', linewidth=1)

        # Agregar la cuadrícula punteada
        ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax_99.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Agregar etiquetas en la parte superior derecha
        ax1.text(xmax[1] - 1, y_max/2 + 0.5, 'Mean', fontsize=14, fontweight='bold', ha='right', va='top', color='black')
        ax_99.text(xmax[1] - 0.5, + 0.5, '99th Percentile', fontsize=14, fontweight='bold', ha='right', va='top', color='black')

        # Agregar una leyenda para cada boxplot
        plt.legend(legend_handles, names, loc='upper right', prop={'size': 14}, frameon=False)


        # Guardar el gráfico
        plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}/{figName}.pdf', bbox_inches='tight')

    print(f"Figura {FIG_NUM} completada!")