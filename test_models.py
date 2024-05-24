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

# Listado de escenarios a predecir
scenarios = ['ssp370']#['ssp126', 'ssp245', 'ssp370', 'ssp585']
main_scenerio = 'ssp370'
historical = ('1980-01-01', '2014-12-31')
future_1 = ('2015-01-01', '2040-12-31')
future_2 = ('2041-01-01', '2070-12-31')
future_3 = ('2071-01-01', '2100-12-31') 

# Cargamos los datos del dataset
predictand = utils.getPredictand(DATA_PATH_PREDICTANDS_SAVE, MODEL_NAME, 'tasmean')
# Split into train and test set
yearsTrain = ('1980-01-01', '2003-12-31')
yearsTest = ('2004-01-01', '2015-12-31')

# Creamos la mascara a usar
baseMask = utils.obtainMask(
    path=f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
    var='tasmean',
    to_slice=(yearsTrain[0], yearsTest[1]))
yFlat = baseMask.flatten(grid=predictand.load(), var='tasmean')
# Extract the raw data from the xarray Dataset
yFlat_array = utils.toArray(yFlat)
yFlat['tasmean'].values = yFlat_array
yUnflatten = baseMask.unFlatten(grid=yFlat, var='tasmean')

era5_predictor = utils.getPredictors(DATA_PREDICTORS_TRANSFORMED)
era5_predictor = era5_predictor.sel(time=slice(*(yearsTrain[0], yearsTest[1])))

for scenario in scenarios:
    #scenario = 'ssp126'
    #Cargamos DATASET PREDICTORES Y PREPARAMOS DATOS
    predictor = utils.loadGcm(GCM_NAME, scenario, (historical[0], future_3[1]), DATA_PATH_PREDICTORS)
    hist_predictor = predictor.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
    future_predictor = predictor.sel(time=slice(*(future_1[0], future_1[1])))

    start_time = time.time()
    hist_metric = utils.getMontlyMetrics(hist_predictor)
    future_metric = utils.getMontlyMetrics(future_predictor)
    era5_metric = utils.getMontlyMetrics(era5_predictor)

    target_predictor = predictor.sel(time=slice(*(future_1[0], future_1[1])))
    target_predictor = utils.standarBiasCorrection(
        target_predictor,
        hist_metric,
        future_metric,
        era5_metric)

    print(target_predictor)
    total_time = time.time() - start_time
    print(f"Primer escalado: {total_time:.2f} segundos.")

    start_time = time.time()
    target_predictor2 = utils.scalingDeltaCorrection(future_predictor, hist_predictor, era5_predictor)
    print(target_predictor2)
    total_time = time.time() - start_time
    print(f"Segundo escalado: {total_time:.2f} segundos.")

    

    # Standardize the predictor
    meanTrain = target_predictor.mean('time')
    stdTrain = target_predictor.std('time')
    xStand = (target_predictor - meanTrain) / stdTrain
    # Extract the raw data from the xarray Dataset
    xStand_array = utils.toArray(xStand)
    print("Forma de xStand y yFlat arrays")
    print(xStand_array.shape)
    print(yFlat_array.shape)

    # Cargamos el modelo
    modelName = f'DeepESD_tas_{MODEL_NAME}' 
    model = models.DeepESD(spatial_x_dim=xStand_array.shape[2:],
                        out_dim=yFlat_array.shape[1],
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
                                flattener=baseMask,
                                var='tasmean')

    yPred.to_netcdf(f'{PREDS_PATH}predGCM_{modelName}_{GCM_NAME}_{scenario}.nc')


# Cargamos Predicciones a escenarios
yPredLoaded = {}
yPredMetrics = {}
for scenario in scenarios:
    yPredLoaded[scenario] = xr.open_dataset(f'{PREDS_PATH}predTest_{modelName}_{GCM_NAME}_{scenario}.nc')
    yPredMetrics[scenario] = utils.getMetricsTemp(yPredLoaded[scenario]) # CARGAMOS METRICAS
    utils.getGraphsTempGCM(yPredMetrics[scenario], scenario, FIGS_PATH, GCM_NAME, MODEL_NAME)# REALIZAMOS GRAFICOS COMPARATIVOS


print("Terminado con exito!")