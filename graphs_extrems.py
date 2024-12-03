import xarray as xr
import torch
from lib import utils, models, data, settings
import sys, time
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm

DATA_PATH_PREDICTORS = '/lustre/gmeteo/PTICLIMA/DATA/PROJECTIONS/CMIP6_PNACC/CMIP6_models/'
DATA_PATH_PREDICTANDS_READ = '/lustre/gmeteo/PTICLIMA/DATA/AUX/GRID_INTERCOMP/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/extremes-interquartile/'
MODELS_PATH = '/oceano/gmeteo/users/reyess/tfm/official-code/models'
DATA_PREDICTORS_TRANSFORMED = '/lustre/gmeteo/WORK/reyess/data/NorthAtlanticRegion_1.5degree/'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/GCM/AEMET/'
PREDS_PATH_TEST = '/lustre/gmeteo/WORK/reyess/preds/'

METRIC = str(sys.argv[1])
#METRIC = 'Mean'
PREDICTANDS_SIZE = 5
ENSEMBLE_QUANTITY = 50
GCM_NAME = 'EC-Earth3-Veg'
MAIN_SCENARIO = 'ssp585'

if PREDICTANDS_SIZE == 6:
    predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'pti-grid', 'CHELSA']
elif PREDICTANDS_SIZE == 5:
    predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0','CHELSA']# 'pti-grid']
elif PREDICTANDS_SIZE == 4:
    predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0']#, 'pti-grid', 'CHELSA']
elif PREDICTANDS_SIZE == 3:
    predictands = ['ERA5-Land0.25deg', 'AEMET_0.25deg', 'pti-grid']

hist_reference = ('1980-01-01', '2014-12-31')
hist_baseline = ('1995-01-01', '2014-12-31') #95-14
future_1 = ('2021-01-01', '2040-12-31')
future_2 = ('2041-01-01', '2060-12-31')
future_3 = ('2081-01-01', '2100-12-31') 
future_4 = ('2061-01-01', '2080-12-31')
yearsTrain = ('1980-01-01', '2003-12-31')
yearsTest = ('2004-01-01', '2015-12-31')
periods = [future_2, future_4, future_3]


figName = f'extremes_ccsignals_Ensemble{ENSEMBLE_QUANTITY}_{METRIC}'
# Crear la figura y los ejes
fig, axes = plt.subplots(7, PREDICTANDS_SIZE, figsize=(20, 21), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

continuousCMAP = plt.get_cmap('hot_r')
discreteCMAP = ListedColormap(continuousCMAP(np.linspace(0, 1, 10)))

# vmin = 1 if METRIC != '99Percentile' else 3
# vmax = 11 if METRIC != '99Percentile' else 13

for i, predictand_name in enumerate(predictands):

    obs_predictand = utils.getPredictand(f'{DATA_PATH_PREDICTANDS_SAVE}', predictand_name, 'tasmean')
    obs_temp = obs_predictand.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
    obs_predictand = utils.maskData(
                path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                var='tasmean',
                to_slice=(yearsTrain[0], yearsTest[1]),
                objective = obs_predictand.sel(time=slice(*(hist_baseline[0], hist_baseline[1]))),
                secondGrid = obs_temp)
    if METRIC == '99Percentile':
        obs_predictand = obs_predictand.resample(time = 'YE').quantile(0.99, dim = 'time')
    elif METRIC == '1Percentile':
        obs_predictand = obs_predictand.resample(time = 'YE').quantile(0.1, dim = 'time')
    obs_predictand_mean = obs_predictand.mean(dim='time')

    # Future Data
    predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
    predictand_data = {'min': None, 'mean': None, 'max': None, 'min_ccsignal' : None, 'max_ccsignal' : None, 'min_test': None, 'max_test' : None}
    predictand_data_mean = {'min': None, 'mean': None, 'max': None, 'min_test': None, 'max_test' : None}
    number_min_max = {'min': '', 'mean': '', 'max': '', 'min_ccsignal' : '', 'max_ccsignal' : '', 'min_test': '', 'max_test' : ''}
    grided_mean_list = []
    index = 1
    for predictand_number in predictand_numbered:
        
        modelName = f'DeepESD_tas_{predictand_number}' 
        loaded_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{future_3[0]}-{future_3[1]}.nc')
        if METRIC == '99Percentile':
            loaded_data = loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')
        elif METRIC == '1Percentile':
            loaded_data = loaded_data.resample(time = 'YE').quantile(0.1, dim = 'time')
        grided_mean = loaded_data.mean(dim=['time', 'lat', 'lon']) 
        #grided_mean_list.append(grided_mean)
        mean_time = loaded_data.mean(dim='time')

        # TEST
        load_data_test = xr.open_dataset(f'{PREDS_PATH_TEST}/predTest_{modelName}.nc')
        if METRIC == '99Percentile':
            load_data_test = load_data_test.resample(time = 'YE').quantile(0.99, dim = 'time')
        elif METRIC == '1Percentile':
            load_data_test = load_data_test.resample(time = 'YE').quantile(0.1, dim = 'time')
        mean_time_test = load_data_test.mean(dim='time')
        grided_mean_test = load_data_test.mean(dim=['time', 'lat', 'lon']) 


        # CHECK MIN AND MAX AND SAVE MIN, MAX, MEAN FOR CCSIGNAL
        if predictand_data_mean['min'] == None or grided_mean['tasmean'].values < predictand_data_mean['min']:
            predictand_data_mean['min']=grided_mean['tasmean'].values
            predictand_data['min']=mean_time
            number_min_max['min'] = index
            number_min_max['min_ccsignal'] = index
            predictand_data['min_ccsignal'] = mean_time_test

        if predictand_data_mean['max'] == None or grided_mean['tasmean'].values > predictand_data_mean['max']:
            predictand_data_mean['max']=grided_mean['tasmean'].values
            predictand_data['max']=mean_time
            number_min_max['max'] = index
            number_min_max['max_ccsignal'] = index
            predictand_data['max_ccsignal'] = mean_time_test

        if predictand_data['mean'] != None:
            predictand_data['mean'] = predictand_data['mean'] + mean_time
        else:
            predictand_data['mean'] = mean_time

        # CHECK MIN AND MAX AND SAVE MIN, MAX FOR TEST PERIOD
        if predictand_data_mean['min_test'] == None or grided_mean_test['tasmean'].values < predictand_data_mean['min_test']:
            predictand_data_mean['min_test']=grided_mean_test['tasmean'].values
            predictand_data['min_test']=mean_time_test
            number_min_max['min_test'] = index

        if predictand_data_mean['max_test'] == None or grided_mean_test['tasmean'].values > predictand_data_mean['max_test']:
            predictand_data_mean['max_test']=grided_mean_test['tasmean'].values
            predictand_data['max_test']=mean_time_test
            number_min_max['max_test'] = index
        
        index = index +1

    predictand_data['mean'] = (predictand_data['mean']/ENSEMBLE_QUANTITY) - obs_predictand_mean
    predictand_data['min'] = predictand_data['min'] - obs_predictand_mean
    predictand_data['max'] = predictand_data['max'] - obs_predictand_mean


    for j, (metric, metric_data) in enumerate(predictand_data.items()):

        if METRIC != '99Percentile' and metric in ['mean', 'min', 'max']:
            vmin = 1
            vmax = 11
        elif METRIC == '99Percentile' and metric in ['mean', 'min', 'max']:
            vmin = 3
            vmax = 13
        elif METRIC == '99Percentile' and metric not in ['mean', 'min', 'max']:
            vmin = 10
            vmax = 40
        elif METRIC == '1Percentile' and metric not in ['mean', 'min', 'max']:
            vmin = -10
            vmax = 20
        else:
            vmin = 0
            vmax = 30


        ax = axes[j, i]
        if j == 0:
            ax.set_title(f'{predictand_name.capitalize()}', fontsize=16)
        if i == 0:
            ax.text(-0.07, 0.55, f'{metric.capitalize()}-{number_min_max[metric]}', va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor',
                transform=ax.transAxes, fontsize=16)

        ax.coastlines(resolution='10m')
        

        dataToPlot = metric_data['tasmean']
        im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                            dataToPlot,
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAP,
                            vmin=vmin, vmax=vmax)
                            #norm=BoundaryNorm(bounds, cmap.N))

        if i == 0:
            cax = fig.add_axes([0.125, 0.818 - (j * 0.129), 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
            cbar = plt.colorbar(im, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
            cbar.set_ticks(np.linspace(vmin, vmax, 6))
            cbar.ax.tick_params(labelsize=16)

plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
plt.close()