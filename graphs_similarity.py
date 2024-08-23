import xarray as xr
from lib import utils, models, data
import sys
import numpy as np

FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/similarity'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/GCM/AEMET/'
PREDS_PATH_TRAIN = '/lustre/gmeteo/WORK/reyess/preds/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
# MODEL_NAME = sys.argv[1]
# PERIOD = int(sys.argv[2])
# SCENARIO = int(sys.argv[3])
# PREDICTOR = sys.argv[4]
#MODEL_NAME = sys.argv[1]
PERIOD = 3
SCENARIO = 3
PREDICTOR = 'EC-Earth3-Veg'

# Listado de escenarios a predecir
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
main_scenerio = 'ssp585'
hist_reference = ('1980-01-01', '2014-12-31')
hist_baseline = ('1995-01-01', '2014-12-31') #95-14
future_1 = ('2021-01-01', '2040-12-31')
future_2 = ('2041-01-01', '2060-12-31')
future_3 = ('2081-01-01', '2100-12-31') 
future_4 = ('2061-01-01', '2080-12-31')
yearsTrain = ('1980-01-01', '2003-12-31')
yearsTest = ('2004-01-01', '2015-12-31')
periods = [hist_baseline, future_1, future_2, future_3, future_4, yearsTest]

scenario = scenarios[SCENARIO]
period = periods[PERIOD]
predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'pti-grid', 'CHELSA']
# ['AEMET_0.25deg', 'Iberia01_v1.0', 'CHELSA'] PARA VER 3 VARIANZAS
# CARGA DE PREDICCIONES
yPredLoaded = {scenario: {}}
yPredMetrics = {scenario: {}}
yPredTenMetrics = {scenario: {}}
yPredVariances = {scenario: {}}


# PARA PERIODOS FUTUROS
for predictand in predictands:
    predictand_numbered = [f"{predictand}_{i}" for i in range(1, 11)]
    yPredLoaded[scenario][predictand] = {}
    yPredMetrics[scenario][predictand] = {}
    yPredTenMetrics[scenario][predictand] = {}

    for predictand_number in predictand_numbered:
        modelName = f'DeepESD_tas_{predictand_number}'
        print(modelName)
        data_temp = xr.open_dataset(f'{PREDS_PATH}predGCM_{modelName}_{PREDICTOR}_{scenario}_{period[0]}-{period[1]}.nc')

        yPredLoaded[scenario][predictand][predictand_number] = xr.open_dataset(f'{PREDS_PATH}predGCM_{modelName}_{PREDICTOR}_{scenario}_{period[0]}-{period[1]}.nc')
        yPredTenMetrics[scenario][predictand][predictand_number] = utils.getMetricsTemp(yPredLoaded[scenario][predictand][predictand_number], var = 'tasmean', short = True)

    yPredMetrics[scenario][predictand] = utils.getMetricsSimilarity(yPredLoaded[scenario][predictand])
yPredVariances[scenario] = utils.getVariance(yPredLoaded[scenario])
utils.graphVariances(yPredVariances, scenario, FIGS_PATH, vmin=0, vmax=2.5, extra='2081-2100')


# IMPRESION DE 10 MODELOS POR PREDICTANDO
#figs_path, vmin = 10, vmax = 30, var = 'mean', scenario = 'ssp585', extension = 'pdf', color='hot_r', color_change = None):
utils.graphMultipleTrains(yPredTenMetrics, figs_path=FIGS_PATH, vmin=12, vmax=27, var='mean', scenario=scenario)

#IMPRESION DE GRAFICOS DE DESVIACION ESTANDAR Y MEDIA
utils.graphSimilarityMetric(yPredMetrics['ssp585'], FIGS_PATH, 0, 1.5, 'std')
utils.graphSimilarityMetric(yPredMetrics['ssp585'], FIGS_PATH, 15, 25, 'mean')


# IMPRESION D EGRAFICOS DE PORCENTAJE
utils.graphSimilarityPercentage(yPredMetrics, FIGS_PATH, scenario, sigma_number=1)
utils.graphSimilarityPercentage(yPredMetrics, FIGS_PATH, scenario, sigma_number=2)


# IMPRESION DE GRAFICOS DE SIMILITUD DE PUNTOS DE LA GRID
utils.graphSimilarityGrid(yPredMetrics, FIGS_PATH, scenario, sigma_number=1)
utils.graphSimilarityGrid(yPredMetrics, FIGS_PATH, scenario, sigma_number=2)

print("Graphs Similarities Done!")




######################################################
# CARGA DE PREDICCIONES
scenario_train = 'train'
scenario_test = 'test'
yPredLoadedTrain = {}
yPredMetricsTrain = {}
yPredLoadedTrain[scenario_train] = {}
yPredMetricsTrain[scenario_train] = {}
yPredLoadedTest = {}
yPredMetricsTest = {}
yPredLoadedTest[scenario_test] = {}
yPredMetricsTest[scenario_test] = {}
yPredMetricsTestTen = {}
yPredMetricsTestTen[scenario_test] = {}
# PARA TRAIN/TEST BASICOS
for predictand in predictands:
    predictand_numbered = [f"{predictand}_{i}" for i in range(1, 11)]
    yPredLoadedTrain[scenario_train][predictand] = {}
    yPredMetricsTrain[scenario_train][predictand] = {}
    yPredLoadedTest[scenario_test][predictand] = {}
    yPredMetricsTest[scenario_test][predictand] = {}

    for predictand_number in predictand_numbered:
        modelName = f'DeepESD_tas_{predictand_number}'
        data_temp_train = xr.open_dataset(f'{PREDS_PATH_TRAIN}predTrain_{modelName}.nc')
        data_temp_test = xr.open_dataset(f'{PREDS_PATH_TRAIN}predTest_{modelName}.nc')

        yPredLoadedTrain[scenario_train][predictand][predictand_number] = data_temp_train
        yPredLoadedTest[scenario_test][predictand][predictand_number] = data_temp_test

    yPredMetricsTrain[scenario_train][predictand] = utils.getMetricsSimilarity(yPredLoadedTrain[scenario_train][predictand])
    yPredMetricsTest[scenario_test][predictand] = utils.getMetricsSimilarity(yPredLoadedTest[scenario_test][predictand])


# IMPRESION DE GRAFICOS de 10 VALORES
utils.graphMultipleTrains(yPredTenMetrics, figs_path=FIGS_PATH, vmin=12, vmax=27, var='mean', scenario=scenario)


# IMPRESION DE GRAFICOS DE DESVIACION ESTANDAR Y MEDIA
utils.graphSimilarityMetric(yPredMetricsTrain[scenario_train], FIGS_PATH, 0, 1.5, 'std', scenario_train)
utils.graphSimilarityMetric(yPredMetricsTrain[scenario_train], FIGS_PATH, 5, 20, 'mean', scenario_train)
utils.graphSimilarityMetric(yPredMetricsTest[scenario_test], FIGS_PATH, 0, 1.5, 'std', scenario_test)
utils.graphSimilarityMetric(yPredMetricsTest[scenario_test], FIGS_PATH, 5, 20, 'mean', scenario_test)

# IMPRESION D EGRAFICOS DE PORCENTAJE
utils.graphSimilarityPercentage(yPredMetricsTrain, FIGS_PATH, scenario_train, sigma_number=1)
utils.graphSimilarityPercentage(yPredMetricsTest, FIGS_PATH, scenario_test, sigma_number=1)
utils.graphSimilarityPercentage(yPredMetricsTrain, FIGS_PATH, scenario_train, sigma_number=2)
utils.graphSimilarityPercentage(yPredMetricsTest, FIGS_PATH, scenario_test, sigma_number=2)


# IMPRESION DE GRAFICOS DE SIMILITUD DE PUNTOS DE LA GRID
utils.graphSimilarityGrid(yPredMetricsTrain, FIGS_PATH, scenario_train, sigma_number=1)
utils.graphSimilarityGrid(yPredMetricsTest, FIGS_PATH, scenario_test, sigma_number=1)
utils.graphSimilarityGrid(yPredMetricsTrain, FIGS_PATH, scenario_train, sigma_number=2)
utils.graphSimilarityGrid(yPredMetricsTest, FIGS_PATH, scenario_test, sigma_number=2)



print("Graphs Similarities Done!")


####################################################################



#Different points
chelsaMean = yPredMetrics['ssp585']['CHELSA']['mean']
current_dif = 0
max_dif = 0
max_key = ''

for key, metric in yPredMetrics['ssp585'].items():
    dif = chelsaMean['tasmean'] - metric['mean']['tasmean']
    current_dif = np.abs(dif).max()
    if current_dif > max_dif:
        max_dif = current_dif
        max_key = key
        last_dif = dif

max_location = last_dif.where(np.abs(last_dif) == max_dif, drop=True)

lat_max = max_location['lat'].values.item()
lon_max = max_location['lon'].values.item()

# Encuentra los índices en lat y lon más cercanos a lat_max y lon_max
lat_idx = chelsaMean['lat'].sel(lat=lat_max, method='nearest').lat
lon_idx = chelsaMean['lon'].sel(lon=lon_max, method='nearest').lon

print(f"Predictand with bigger difference: {max_key}")

for key, metric in yPredMetrics['ssp585'].items():
    mean = metric['mean']['tasmean'].sel(lat=lat_idx, lon=lon_idx).item()
    sigma = metric['std']['tasmean'].sel(lat=lat_idx, lon=lon_idx).item()
    interval = (mean-sigma, mean+sigma)
    print(f"Predictand: {key} - Mean: {mean} - Range: {interval}")


utils.graphSimilarityGrid(yPredMetrics, FIGS_PATH, scenario, sigma_number=1, grid_selection=(lat_idx, lon_idx))


###########################################################
# GRAFICOS de 10 PREDICCIONES
