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
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/GCM/'
VARIABLES_TO_DROP = ['lon_bnds', 'lat_bnds', 'crs']
LAT_SLICE = slice(33.5, 48.6)
LON_SLICE = slice(-10.5, 4.6)
MODEL_NAME = sys.argv[1]
GCM_NAME = sys.argv[2]
PERIOD = int(sys.argv[3])
SCENARIO = int(sys.argv[4])
#MODEL_NAME = 'ERA5-Land0.25deg'
#GCM_NAME = 'EC-Earth3-Veg'

# Listado de escenarios a predecir
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
main_scenerio = 'ssp370'
hist_reference = ('1980-01-01', '2014-12-31')
hist_baseline = ('1995-01-01', '2014-12-31') #95-14
future_1 = ('2021-01-01', '2040-12-31')
future_2 = ('2041-01-01', '2060-12-31')
future_3 = ('2081-01-01', '2100-12-31')
future_4 = ('2061-01-01', '2080-12-31')
periods = [hist_baseline, future_1, future_2, future_3, future_4]
yearsTrain = ('1980-01-01', '2003-12-31')
yearsTest = ('2004-01-01', '2015-12-31')

# Cargamos los datos del dataset
predictand = utils.getPredictand(DATA_PATH_PREDICTANDS_SAVE, MODEL_NAME, 'tasmean')
predictand = predictand.sel(time=slice(*(yearsTrain[0], yearsTrain[1]))).load()

# Creamos la mascara a usar
baseMask = utils.obtainMask(
    path=f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
    var='tasmean',
    to_slice=(yearsTrain[0], yearsTrain[1]))
maskToUse = baseMask
yFlat = baseMask.flatten(grid=predictand, var='tasmean')
# Extract the raw data from the xarray Dataset
yFlat_array = utils.toArray(yFlat)
yFlat['tasmean'].values = yFlat_array
yUnflatten = baseMask.unFlatten(grid=yFlat, var='tasmean')

if np.isnan(yFlat_array).sum() > 0:
    print("Segunda mascara en proceso")
    secondMask = utils.obtainMask(grid = yUnflatten, var = 'tasmean')
    yFlat = secondMask.flatten(grid=yUnflatten, var='tasmean')
    yFlat_array = utils.toArray(yFlat)
    yFlat['tasmean'].values = yFlat_array
    yUnflatten = secondMask.unFlatten(grid=yFlat, var='tasmean')
    maskToUse = secondMask

era5_data = utils.getPredictors(DATA_PREDICTORS_TRANSFORMED)
era5_predictor = era5_data.sel(time=slice(*(hist_reference[0], hist_reference[1]))).load()
era5_train = era5_data.sel(time=slice(*yearsTrain)).load()

# future = hist_baseline
# scenario = 'ssp126'
SCENARIO = scenarios[SCENARIO]
PERIOD = periods[PERIOD]

#Cargamos DATASET PREDICTORES Y PREPARAMOS DATOS
predictor = utils.loadGcm(GCM_NAME, SCENARIO, (hist_reference[0], future_3[1]), DATA_PATH_PREDICTORS)
hist_predictor = predictor.sel(time=slice(*(hist_reference[0], hist_reference[1]))).load()
future_predictor = predictor.sel(time=slice(*(PERIOD[0], PERIOD[1]))).load()

#start_time = time.time()
hist_metric = utils.getMontlyMetrics(hist_predictor)
future_metric = utils.getMontlyMetrics(future_predictor)
era5_metric = utils.getMontlyMetrics(era5_predictor)

target_predictor = predictor.sel(time=slice(*(PERIOD[0], PERIOD[1]))).load()
target_predictor = utils.standarBiasCorrection(
    target_predictor,
    hist_metric,
    future_metric,
    era5_metric)


# Standardize the predictor
# Tiene que ser Era5 de 80 a 2015
meanTrain = era5_train.mean('time')
stdTrain = era5_train.std('time')
xStand = (target_predictor - meanTrain) / stdTrain
# Extract the raw data from the xarray Dataset

xStand_array = utils.toArray(xStand)#xStand)
print("Forma de xStand y yFlat arrays")
print(xStand_array.shape)
print(yFlat_array.shape)

# Cargamos el modelo
modelName = f'DeepESD_tas_{MODEL_NAME}' 
model = models.DeepESD(spatial_x_dim=xStand_array.shape[2:],
                    out_dim=yFlat_array.shape[1],#output_dim[MODEL_NAME],
                    channelsInputLayer=xStand_array.shape[1],
                    channelsLastLayer=10)

# Load the model state dictionary from a file
checkpoint = torch.load(f'{MODELS_PATH}/DeepESD_tas_{MODEL_NAME}.pt')
model.load_state_dict(checkpoint)

# Realizamos la prediccion
# Compute predictions on the test set
# Transformacion de coordenada time de yUnflatten
yUnflatten = yUnflatten.reindex(time=target_predictor.time, fill_value=np.nan)
#yUnflatten = yUnflatten.resample(time=target_predictor.time)

print("yUnflatten")
print(yUnflatten) # Transformar tiempos a los mismos de GCM
yPred = utils.predDataset(X=xStand_array,
                            model=model,
                            device='cpu',
                            ref=yUnflatten,
                            flattener=maskToUse,
                            var='tasmean')

yPred.to_netcdf(f'{PREDS_PATH}predGCM_{modelName}_{GCM_NAME}_{SCENARIO}_{PERIOD[0]}-{PERIOD[1]}.nc')

print("TERMINADO CON EXITO!")
