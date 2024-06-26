import xarray as xr
import torch
from lib import utils, models, data
import sys

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
MODEL_NAME = sys.argv[1]
GCM_NAME = sys.argv[2]
#MODEL_NAME = 'E-OBS'
#GCM_NAME = 'EC-Earth3-Veg'

# Listado de escenarios a predecir
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']


# Cargamos los datos del dataset
file_name = utils.getFileName(DATA_PATH_PREDICTANDS_SAVE, MODEL_NAME, keyword = 'tasmean')
predictand = xr.open_dataset(f'{DATA_PATH_PREDICTANDS_SAVE}{MODEL_NAME}/{file_name}',
                             chunks=-1)
predictand = utils.checkCorrectData(predictand) # Transform coordinates and dimensions if necessary

predictand = utils.checkIndex(predictand)

#predictand = predictand.sel(lon=LON_SLICE, lat=LAT_SLICE) # Tomamos solo los datos de iberia del predictando
print("Predictand 1")
print(predictand)

# Split into train and test set
yearsTrain = ('1980-01-01', '2003-12-31')
yearsTest = ('2004-01-01', '2016-12-31')

# Creamos la mascara a usar
baseMask = utils.obtainMask(
    path=f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
    var='tasmean',
    to_slice=(yearsTrain[0], yearsTest[1]))

for scenario in scenarios:
    #scenario = 'ssp126'
    #Cargamos DATASET PREDICTORES Y PREPARAMOS DATOS
    predictor = utils.loadGcm(GCM_NAME, scenario, DATA_PATH_PREDICTORS)
    print("Predictor 1:")
    print(predictor)
    # Align both datasets in time
    predictand=predictand.assign_coords({'time': predictand.indexes['time'].normalize()})
    print("Predictand 2:")
    print(predictand)


    yFlat = baseMask.flatten(grid=predictand.load(), var='tasmean')

    # Extract the raw data from the xarray Dataset
    yFlat_array = utils.toArray(yFlat)
    yFlat['tasmean'].values = yFlat_array
    yUnflatten = baseMask.unFlatten(grid=yFlat, var='tasmean')

    # Standardize the predictor
    meanTrain = predictor.mean('time')
    stdTrain = predictor.std('time')
    xStand = (predictor - meanTrain) / stdTrain
    # Extract the raw data from the xarray Dataset
    xStand_array = utils.toArray(xStand)
    print("Forma de xStand y yFlat arrays")
    print(xStand_array.shape)
    print(yFlat_array.shape[1])

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
    yUnflatten = yUnflatten.resample(time=predictor.time)

    print("yUnflatten")
    print(yUnflatten) # Transformar tiempos a los mismos de GCM
    yPred = utils.predDataset(X=xStand_array,
                                model=model,
                                device='cpu',
                                ref=yUnflatten,
                                flattener=baseMask,
                                var='tasmean')

    yPred.to_netcdf(f'{PREDS_PATH}predTest_{modelName}_{GCM_NAME}_{scenario}.nc')


# Cargamos Predicciones a escenarios
yPredLoaded = {}
yPredMetrics = {}
for scenario in scenarios:
    yPredLoaded[scenario] = xr.open_dataset(f'{PREDS_PATH}predTest_{modelName}_{GCM_NAME}_{scenario}.nc')
    yPredMetrics[scenario] = utils.getMetricsTemp(yPredLoaded[scenario]) # CARGAMOS METRICAS
    utils.getGraphsTempGCM(yPredMetrics[scenario], scenario, FIGS_PATH, GCM_NAME, MODEL_NAME)# REALIZAMOS GRAFICOS COMPARATIVOS


print("Terminado con exito!")