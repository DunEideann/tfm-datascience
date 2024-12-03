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
FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/rmse/'
MODELS_PATH = '/oceano/gmeteo/users/reyess/tfm/official-code/models'
DATA_PREDICTORS_TRANSFORMED = '/lustre/gmeteo/WORK/reyess/data/NorthAtlanticRegion_1.5degree/'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/GCM/AEMET/'
PREDS_PATH_TEST = '/lustre/gmeteo/WORK/reyess/preds/'

METRIC = str(sys.argv[1])
#METRIC = '1Percentile'
ENSEMBLE_QUANTITY = 50
GCM_NAME = 'EC-Earth3-Veg'
MAIN_SCENARIO = 'ssp585'

hist_reference = ('1980-01-01', '2014-12-31')
hist_baseline = ('1995-01-01', '2014-12-31') #95-14
future_1 = ('2021-01-01', '2040-12-31')
future_2 = ('2041-01-01', '2060-12-31')
future_3 = ('2081-01-01', '2100-12-31') 
future_4 = ('2061-01-01', '2080-12-31')
yearsTrain = ('1980-01-01', '2003-12-31')
yearsTest = ('2004-01-01', '2015-12-31')
predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0','CHELSA']
rmse_test = {predictand_name : [] for predictand_name in predictands}
test_pred = {predictand_name : [] for predictand_name in predictands}
gcm_pred = {predictand_name : [] for predictand_name in predictands}


for predictand_name in predictands:
    modelName = f'DeepESD_tas_{predictand_name}' 
    loaded_test_obs = utils.getPredictand(DATA_PATH_PREDICTANDS_SAVE, predictand_name, 'tasmean')
    loaded_test_obs = loaded_test_obs.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
    loaded_test_obs = utils.maskData(
                path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                var='tasmean',
                to_slice=(yearsTrain[0], yearsTest[1]),
                objective = loaded_test_obs,
                secondGrid = loaded_test_obs)
    

    predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
    for predictand_number in predictand_numbered:
        modelName = f'DeepESD_tas_{predictand_number}' 
        loaded_test = xr.open_dataset(f'{PREDS_PATH_TEST}predTest_{modelName}.nc')
        rmse = np.sqrt((((loaded_test - loaded_test_obs)**2).mean(dim=['time', 'lat', 'lon']))['tasmean'])
        rmse_test[predictand_name].append(rmse)

        if METRIC == '99Percentile':
            loaded_test = loaded_test.resample(time = 'YE').quantile(0.99, dim = 'time')
        elif METRIC == '1Percentile':
            loaded_test = loaded_test.resample(time = 'YE').quantile(0.1, dim = 'time')

        test_pred[predictand_name].append(loaded_test.mean(dim=['time', 'lat', 'lon'])['tasmean'])

        loaded_pred = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{future_3[0]}-{future_3[1]}.nc')
        if METRIC == '99Percentile':
            loaded_pred = loaded_pred.resample(time = 'YE').quantile(0.99, dim = 'time')
        elif METRIC == '1Percentile':
            loaded_pred = loaded_pred.resample(time = 'YE').quantile(0.1, dim = 'time')

        gcm_pred[predictand_name].append(loaded_pred.mean(dim=['time', 'lat', 'lon'])['tasmean'])
        



# Datos de ejemplo
rmse_values = [1.2, 1.5, 1.1, 1.4, 1.3]  # Valores de RMSE
temperature_values = [20, 22, 19, 23, 21]  # Valores de temperatura
temperature_values2 = [20, 22, 19, 25, 21]  # Valores de temperatura

labels =  [f"{i}" for i in range(1,ENSEMBLE_QUANTITY+1)] # Etiquetas
limits = {'Mean': {'rmse_lim': (0, 1.2), 'rmse_ticks': (0, 1, 6), 'temp_lim': (11.5, 21.5), 'temp_ticks': (12, 21, 10)}, 
          '99Percentile': {'rmse_lim': (0, 1.2), 'rmse_ticks': (0, 1, 6), 'temp_lim': (23.5, 35.5), 'temp_ticks': (24, 34, 6)}, 
          '1Percentile': {'rmse_lim': (0, 1.2), 'rmse_ticks': (0, 1, 6), 'temp_lim': (3, 12), 'temp_ticks': (4, 11, 8)}}
temperature_ticks = np.linspace(limits[METRIC]['temp_ticks'][0], limits[METRIC]['temp_ticks'][1], limits[METRIC]['temp_ticks'][2])



for predictand_name in predictands:
    # Crear el gráfico
    fig, ax1 = plt.subplots()

    # Configurar el eje izquierdo (barras para RMSE)
    ax1.bar(labels, rmse_test[predictand_name], color='skyblue', label='RMSE')
    ax1.set_ylabel('RMSE', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(limits[METRIC]['rmse_lim'][0], limits[METRIC]['rmse_lim'][1])
    ax1.set_yticks(np.linspace(limits[METRIC]['rmse_ticks'][0], limits[METRIC]['rmse_ticks'][1], limits[METRIC]['rmse_ticks'][2]))
    #ax1.set_xticklabels(labels, rotation=90)

    # Crear el eje derecho (línea para temperatura)
    ax2 = ax1.twinx()
    ax2.plot(labels, test_pred[predictand_name], color='orange', marker='o', label='Temperatura')
    # ax2.set_ylabel('Temperatura (°C)', color='orange')
    # ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(limits[METRIC]['temp_lim'][0], limits[METRIC]['temp_lim'][1])
    ax2.set_yticks([])

    # Girar etiquetas del eje X
    ax1.set_xticks(range(len(labels)))  # Asegúrate de alinear los ticks con las etiquetas
    ax1.set_xticklabels(labels, rotation = 270, fontsize = 10)

    # Crear el eje derecho (línea para temperatura)
    ax3 = ax1.twinx()
    ax3.plot(labels, gcm_pred[predictand_name], color='red', marker='o', label='Temperatura')
    ax3.set_ylabel('Temperatura (°C)', color='red')
    ax3.tick_params(axis='y', labelcolor='red')
    ax3.set_ylim(limits[METRIC]['temp_lim'][0], limits[METRIC]['temp_lim'][1])
    ax3.set_yticks(temperature_ticks)

    # Títulos y leyenda
    ax1.set_title(f'RMSE vs Temperatura {predictand_name}')
    #plt.xticks(rotation=90)
    fig.tight_layout()
    plt.savefig(f'{FIGS_PATH}rmse_{predictand_name}_{METRIC}_Ensemble{ENSEMBLE_QUANTITY}.png', bbox_inches='tight')
    plt.close()


