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
from scipy import signal, stats
#/oceano/gmeteo/users/reyess/tfm/official-code

DATA_PATH_PREDICTORS = '/lustre/gmeteo/PTICLIMA/DATA/REANALYSIS/ERA5/data_derived/NorthAtlanticRegion_1.5degree/'
DATA_PATH_PREDICTANDS_READ = '/lustre/gmeteo/PTICLIMA/DATA/AUX/GRID_INTERCOMP/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/'
MODELS_PATH = '/oceano/gmeteo/users/reyess/tfm/official-code/models'
DATA_PREDICTORS_TRANSFORMED = '/lustre/gmeteo/WORK/reyess/data/NorthAtlanticRegion_1.5degree/'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/'
VARIABLES_TO_DROP = ['lon_bnds', 'lat_bnds', 'crs']
LAT_SLICE = slice(33.5, 48.6)
LON_SLICE = slice(-10.5, 4.6)
PREDICTAND_NAME = sys.argv[1]
#PREDICTAND_NAME = 'E-OBS' PREDICTAND_NAME = 'AEMET_0.25deg' PREDICTAND_NAME = 'Iberia01_v1.0' 
# PREDICTAND_NAME = 'pti-grid' PREDICTAND_NAME = 'CHELSA' PREDICTAND_NAME = 'ERA5-Land0.25deg'


predictors_vars = ['t500', 't700', 't850', # Air temperature at 500, 700, 850 hPa
'q500', 'q700', 'q850', # Specific humidity at 500, 700, 850 hPa
'v500', 'v700', 'v850', # Meridional wind component at 500, 700, 850 hPa
'u500', 'u700', 'u850', # Zonal wind component at 500, 700, 850 hPa
'msl'] # Mean sea level pressure (psl)


# Mergiamos los datasets por la coordenada tiempo si es necesario
data_transformed = os.listdir(f'{DATA_PREDICTORS_TRANSFORMED}')
data_predictors = []
for var in predictors_vars:
    # Verificamos si ya existe el dataset mergiado, si ya existe, solo lo cargamos.
    target_file = f'{var}_ERA5.nc'
    if target_file in data_transformed:
        data_predictors.append(xr.open_dataset(f'{DATA_PREDICTORS_TRANSFORMED}/{var}_ERA5.nc'))
    # De no existir, procedemos a mergiarlo.
    else:
        folders = os.listdir(f'{DATA_PATH_PREDICTORS}{var}')
        data_predictor_years = []
        for year in folders:
            data_predictor_1yr = xr.open_mfdataset(f'{DATA_PATH_PREDICTORS}{var}/{year}/{var}_ERA5_day_*.nc', chunks=-1)#, combine = 'time') # Mergiamos todos los archivos de 1 año
            data_predictor_1yr = data_predictor_1yr.drop_vars(VARIABLES_TO_DROP) # Eliminamos variables que no usaremos
            data_predictor_years.append(data_predictor_1yr) # Agregamos a una lista el dataset del año
        data_predictor_merged = xr.concat(data_predictor_years, dim = 'time') # Mergiamos todos los años en 1 dataset
        data_predictor_merged.to_netcdf(f'{DATA_PREDICTORS_TRANSFORMED}{var}_ERA5.nc') # Gaurdamos el dataset
        data_predictors.append(data_predictor_merged) # Agrgamos el dataset de la variable a una lista


predictors = xr.merge(data_predictors) # Mergiamos todos los dataset de distintas variables en 1
print("Predictores terminados!")
# Cargamos el predictando y lo preparamos
# Obtenemos el nombre del dataset
file_name = utils.getFileName(DATA_PATH_PREDICTANDS_SAVE, PREDICTAND_NAME, keyword = 'tasmean')


#/oceano/gmeteo/users/reyess/tfm/official-code
predictand_path = f'{DATA_PATH_PREDICTANDS_SAVE}{PREDICTAND_NAME}/{file_name}'
predictand = xr.open_dataset(predictand_path,
                             chunks=-1) # Near surface air temperature (daily mean)
predictand = utils.checkCorrectData(predictand) # Transform coordinates and dimensions if necessary

predictand = utils.checkIndex(predictand)
predictand = utils.checkUnitsTempt(predictand, 'tasmean')
predictors = predictors.reindex(lat=list(reversed(predictors.lat))) # Reordenamos la latitud del predictor para que tenga el mismo orden del predictando

#Preparamos datos para entrenamiento
# Remove days with nans in the predictor
predictors = utils.removeNANs(grid=predictors)

# Align both datasets in time
predictand=predictand.assign_coords({'time': predictand.indexes['time'].normalize()})
y, x = utils.alignDatasets(grid1=predictand, grid2=predictors, coord='time')

# Split into train and test set
yearsTrain = ('1980-01-01', '2003-12-31')
yearsTest = ('2004-01-01', '2015-12-31')


# Filtrar años en base a predictandos y ver sus NANs y los rangos de años
xTrain = x.sel(time=slice(*yearsTrain)).load()
xTest = x.sel(time=slice(*yearsTest)).load()

yTrain = y.sel(time=slice(*yearsTrain)).load()
yTest = y.sel(time=slice(*yearsTest)).load()

# Standardize the predictor
meanTrain = xTrain.mean('time')
stdTrain = xTrain.std('time')
xTrainStand = (xTrain - meanTrain) / stdTrain

# Extract the raw data from the xarray Dataset
xTrainStand_array = utils.toArray(xTrainStand)

# Remove nans gridpoints and flatten the predictand
baseMask = utils.obtainMask(
    path=f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
    var='tasmean',
    to_slice=(yearsTrain[0], yearsTest[1]))
yTrainFlat = baseMask.flatten(grid=yTrain, var='tasmean')
#plt.figure(); yTrain['tasmean'].mean('time').plot(); plt.savefig('./yTestPre.pdf')

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
    print(f"Valores NAN en yTrain: {np.isnan(yTrainFlat_array).sum()}- Radio de nueva mascara: {secondMask.refArray.shape}/{baseMask.refArray.shape}")


# Comenzamos entrenamiento del modelo
# Load the DeepESD model
modelName = f'DeepESD_tas_{PREDICTAND_NAME}' # Name used to save the model as a .pt file
model = models.DeepESD(spatial_x_dim=xTrainStand_array.shape[2:],
                       out_dim=yTrainFlat_array.shape[1],
                       channelsInputLayer=xTrainStand_array.shape[1],
                       channelsLastLayer=10)
print(model)
# Split training data into training and validation sets
xTrainM, yTrainM, \
xValidM, yValidM = utils.validSet_fromArray(Xarray=xTrainStand_array,
                                            Yarray=yTrainFlat_array,
                                            validPerc=0.1,  
                                            seed=15)

# Create Dataset and DataLoaders
batchSize = 64

trainDataset = data.downscalingDataset(xTrainM, yTrainM)
trainDataloader = DataLoader(trainDataset, batch_size=batchSize,
                             shuffle=True)

validDataset = data.downscalingDataset(xValidM, yValidM)
validDataloader = DataLoader(validDataset, batch_size=batchSize,
                             shuffle=True)

# Optimizer
lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)

# Loss
lossFunction = torch.nn.MSELoss()

# Number of epochs
numEpochs = 10000

# Patience for the early-stopping
patience = 30

# Train the model
trainLoss, valLoss = utils.train_model(model=model, modelName=modelName, modelPath=MODELS_PATH, device='cpu',
                                       lossFunction=lossFunction, optimizer=optimizer,
                                       numEpochs=numEpochs, patience=patience,
                                       trainData=trainDataloader, validData=validDataloader)

print(model)

# Standardize the predictors in the test set (w.r.t train climatology)
xTestStand = (xTest - meanTrain) / stdTrain
xTestStand_array = utils.toArray(xTestStand)

# Compute predictions on the test set
yPredTest = utils.predDataset(X=xTestStand_array,
                              model=model,
                              device='cpu',
                              ref=yTestUnflatten,
                              flattener=maskToUse,
                              var='tasmean')

yPredTest.to_netcdf(f'{PREDS_PATH}predTest_{modelName}.nc')
import time

time.sleep(3)
# Compure predictions on the train set
yPredTrain = utils.predDataset(X=xTrainStand_array,
                              model=model,
                              device='cpu',
                              ref=yTrainUnflatten,
                              flattener=maskToUse,
                              var='tasmean')

yPredTrain.to_netcdf(f'{PREDS_PATH}predTrain_{modelName}.nc')


# Calculate metrics and save graphs
for season_name, months in {'spring': 'MAM', 'summer': 'JJA', 'autumn': 'SON', 'winter': 'DJF'}.items():
    y_test_season = yTestUnflatten.isel(time = (yPredTest.time.dt.season == months))
    y_pred_season = yPredTest.isel(time = (yPredTest.time.dt.season == months))
    y_test_metrics = utils.getMetricsTemp(y_test_season)
    y_pred_metrics = utils.getMetricsTemp(y_pred_season) # y_metrics = {'mean': , '99quantile': , 'std': , 'trend': }
    utils.getGraphsTemp(y_test_metrics, y_pred_metrics, season_name, FIGS_PATH, PREDICTAND_NAME)

print("Terminado con exito!")