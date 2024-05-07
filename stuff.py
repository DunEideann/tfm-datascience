import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import dask; dask.config.set({'array.slicing.split_large_chunks': False})
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from lib import utils, models, data
import xarray as xr
import os
import sys
from datetime import datetime


DATA_PATH_PREDICTORS = '/lustre/gmeteo/PTICLIMA/DATA/REANALYSIS/ERA5/data_derived/NorthAtlanticRegion_1.5degree/'
DATA_PATH_PREDICTANDS_READ = '/lustre/gmeteo/PTICLIMA/DATA/AUX/GRID_INTERCOMP/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/'
MODELS_PATH = './models'
DATA_PREDICTORS_TRANSFORMED = '/lustre/gmeteo/WORK/reyess/data/NorthAtlanticRegion_1.5degree/'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/'
VARIABLES_TO_DROP = ['lon_bnds', 'lat_bnds', 'crs']
LAT_SLICE = slice(33.5, 48.6)
LON_SLICE = slice(-10.5, 4.6)
#PREDICTAND_NAME = sys.argv[1]




yPredTest = xr.open_dataset(f'{PREDS_PATH}predTest_DeepESD_tas_E-OBS.nc')

print(yPredTest['tasmean'].values)
print('*************************')
yPr = yPredTest.isel(time = (yPredTest.time.dt.season == 'MAM'))
print(yPr.sel(lat = slice(34, 34.5), lon=slice(-9.876, -9.5))['tasmean'])
print('------------------------')
print(yPr['tasmean'].mean(dim='time'))


# CODIGO UTIL CREACION DE DATASETS PREDICTORES
# En caso de no existir el dataset lo creamos:
PREDICTAND_NAME = 'pti-grid'#'CERRA_0.25reg'
if file_name == None:
    print("No existe predictando para la media, se creara.")

    max_name = utils.getFileName(DATA_PATH_PREDICTANDS_READ, PREDICTAND_NAME, keyword = 'tasmax')
    min_name = utils.getFileName(DATA_PATH_PREDICTANDS_READ, PREDICTAND_NAME, keyword = 'tasmin')
data_max = xr.open_dataset(f'{DATA_PATH_PREDICTANDS_READ}{PREDICTAND_NAME}/{max_name}', chunks=-1)
data_min = xr.open_dataset(f'{DATA_PATH_PREDICTANDS_READ}{PREDICTAND_NAME}/{min_name}', chunks=-1)
data_mean = xr.merge([data_min, data_max])

data_mean = data_mean.assign(tasmean=(data_mean['tasmax'] + data_mean['tasmin']) / 2)
data_mean = data_mean.drop_vars(['tasmin', 'tasmax'])

# Liberamos memoria
data_max.close()
data_min.close()

###############################
##### CASO DE AÑOS EN CARPETAS##########
folders = os.listdir(f'{DATA_PATH_PREDICTANDS_READ}{PREDICTAND_NAME}/tasmax')
folders.sort()
max_todos = []
for year in folders:
    data_predictor_1yr = xr.open_dataset(f'{DATA_PATH_PREDICTANDS_READ}{PREDICTAND_NAME}/tasmax/{year}', chunks=-1)#, combine = 'time') # Mergiamos todos los archivos de 1 año
    #data_predictor_1yr = data_predictor_1yr.drop_vars(VARIABLES_TO_DROP) # Eliminamos variables que no usaremos
    max_todos.append(data_predictor_1yr) # Agregamos a una lista el dataset del año
max_todos_merged = xr.concat(max_todos, dim = 'time') # Mergiamos todos los años en 1 dataset

folders = os.listdir(f'{DATA_PATH_PREDICTANDS_READ}{PREDICTAND_NAME}/tasmin')
folders.sort()
min_todos = []
for year in folders:
    data_predictor_1yr = xr.open_dataset(f'{DATA_PATH_PREDICTANDS_READ}{PREDICTAND_NAME}/tasmin/{year}', chunks=-1)#, combine = 'time') # Mergiamos todos los archivos de 1 año
    #data_predictor_1yr = data_predictor_1yr.drop_vars(VARIABLES_TO_DROP) # Eliminamos variables que no usaremos
    min_todos.append(data_predictor_1yr) # Agregamos a una lista el dataset del año
min_todos_merged = xr.concat(min_todos, dim = 'time') # Mergiamos todos los años en 1 dataset

max_todos_merged = xr.open_dataset(f'{DATA_PATH_PREDICTANDS_READ}{PREDICTAND_NAME}/Iberian_v1.0_DD_025reg_aa3d_tasmax.nc', chunks=-1)
min_todos_merged = xr.open_dataset(f'{DATA_PATH_PREDICTANDS_READ}{PREDICTAND_NAME}/Iberian_v1.0_DD_025reg_aa3d_tasmin.nc', chunks=-1)

data_mean = xr.merge([min_todos_merged, max_todos_merged])

data_mean = data_mean.assign(tasmean=(data_mean['tasmax'] + data_mean['tasmin']) / 2)
data_mean = data_mean.drop_vars(['tasmin', 'tasmax'])

file_name = f'{PREDICTAND_NAME}_tasmean_1961-2022.nc'
if not os.path.exists(f'{DATA_PATH_PREDICTANDS_SAVE}{PREDICTAND_NAME}'):
        os.makedirs(f'{DATA_PATH_PREDICTANDS_SAVE}{PREDICTAND_NAME}')
data_mean.to_netcdf(f'{DATA_PATH_PREDICTANDS_SAVE}{PREDICTAND_NAME}/{file_name}')
data_mean.close()
##################################
