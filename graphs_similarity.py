import xarray as xr
from lib import utils, models, data
import sys
import numpy as np

FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/similarity'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/GCM/AEMET/'
PREDS_PATH_TRAIN = '/lustre/gmeteo/WORK/reyess/preds/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
# PERIOD = int(sys.argv[1])
# PREDICTANDS_SIZE = int(sys.argv[2])
# ENSEMBLE_QUANTITY = int(sys.argc[3])
PREDICTANDS_SIZE = 6
PERIOD = 3
ENSEMBLE_QUANTITY = 10
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
scenario_train = 'train'
scenario_test = 'test'
scenario_whole = 'whole'
scenario_ccsignal = 'ccsignal'
period = periods[PERIOD]
if PREDICTANDS_SIZE == 6:
    predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'pti-grid', 'CHELSA']
elif PREDICTANDS_SIZE == 5:
    predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0','CHELSA']# 'pti-grid']
elif PREDICTANDS_SIZE == 4:
    predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0']#, 'pti-grid', 'CHELSA']
elif PREDICTANDS_SIZE == 3:
    predictands = ['ERA5-Land0.25deg', 'AEMET_0.25deg', 'pti-grid']
    
len_predictands = len(predictands)
extra_period = 'MEDIUM' if PERIOD == 4 else 'LONG'
#['AEMET_0.25deg', 'Iberia01_v1.0', 'CHELSA'] PARA VER 3 VARIANZAS
#CARGA DE PREDICCIONES
yPredLoaded = {scenario: {}, scenario_train: {}, scenario_test: {}, scenario_whole: {}}
yPredMetrics = {scenario: {}, scenario_train: {}, scenario_test: {}, scenario_ccsignal: {}}

yPredTenMetrics = {scenario: {}}
yMeanVariances = {scenario: {}, scenario_train: {}, scenario_test: {}, scenario_ccsignal: {}}
y99quanVariances = {scenario: {}, scenario_train: {}, scenario_test: {}, scenario_ccsignal: {}}
y1quanVariances = {scenario: {}, scenario_train: {}, scenario_test: {}, scenario_ccsignal: {}}


# PARA PERIODOS FUTUROS
for predictand in predictands:
    predictand_numbered = [f"{predictand}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
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

print("1")
# VARIANCES metric = mean, 99quantile, 1quantile, sd 
yMeanVariances[scenario] = utils.getVariance(yPredLoaded[scenario], metric='mean', percentage=True) 
utils.graphVariances(yMeanVariances, scenario, FIGS_PATH, vmin=0, vmax=100, extra=f'{period[0]}-{period[1]}_Mean_Percentage_{len_predictands}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'Mean {extra_period}')
utils.graphVariancesMeanSd(yMeanVariances, scenario, FIGS_PATH, extra=f'{period[0]}-{period[1]}_Mean_Percentage_{len_predictands}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'Mean {extra_period}')
print("2")
# 99 Quantile
y99quanVariances[scenario] = utils.getVariance(yPredLoaded[scenario], metric='99quantile', percentage=True) 
utils.graphVariances(y99quanVariances, scenario, FIGS_PATH, vmin=0, vmax=100, extra=f'{period[0]}-{period[1]}_99Quantile_Percentage_{len_predictands}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'99Percentile {extra_period}')
utils.graphVariancesMeanSd(y99quanVariances, scenario, FIGS_PATH, extra=f'{period[0]}-{period[1]}_99Quantile_Percentage_{len_predictands}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'99Percentile {extra_period}')
# 1 Quantile
y1quanVariances[scenario] = utils.getVariance(yPredLoaded[scenario], metric='1quantile', percentage=True) 
utils.graphVariances(y1quanVariances, scenario, FIGS_PATH, vmin=0, vmax=100, extra=f'{period[0]}-{period[1]}_1Quantile_Percentage_{len_predictands}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'1Percentil {extra_period}')
utils.graphVariancesMeanSd(y1quanVariances, scenario, FIGS_PATH, extra=f'{period[0]}-{period[1]}_1Quantile_Percentage_{len_predictands}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'1Percentil {extra_period}')




# # # IMPRESION DE 10 MODELOS POR PREDICTANDO
# # #figs_path, vmin = 10, vmax = 30, var = 'mean', scenario = 'ssp585', extension = 'pdf', color='hot_r', color_change = None):
# # utils.graphMultipleTrains(yPredTenMetrics, figs_path=FIGS_PATH, vmin=12, vmax=27, var='mean', scenario=scenario)

# # #IMPRESION DE GRAFICOS DE DESVIACION ESTANDAR Y MEDIA
# # utils.graphSimilarityMetric(yPredMetrics['ssp585'], FIGS_PATH, 0, 1.5, 'std')
# # utils.graphSimilarityMetric(yPredMetrics['ssp585'], FIGS_PATH, 15, 25, 'mean')


# # # IMPRESION D EGRAFICOS DE PORCENTAJE
# # utils.graphSimilarityPercentage(yPredMetrics, FIGS_PATH, scenario, sigma_number=1)
# # utils.graphSimilarityPercentage(yPredMetrics, FIGS_PATH, scenario, sigma_number=2)


# # # IMPRESION DE GRAFICOS DE SIMILITUD DE PUNTOS DE LA GRID
# # utils.graphSimilarityGrid(yPredMetrics, FIGS_PATH, scenario, sigma_number=1)
# # utils.graphSimilarityGrid(yPredMetrics, FIGS_PATH, scenario, sigma_number=2)

# print("Graphs Similarities Done!")




# ######################################################
# # CARGA DE PREDICCIONES
# #TODO Agregar scenarios a un solo diccionario y no muchos distintos

# yPredMetricsTestTen = {scenario_test: {}}

# PARA TRAIN/TEST BASICOS
for predictand in predictands:
    predictand_numbered = [f"{predictand}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
    yPredLoaded[scenario_train][predictand] = {}
    yPredMetrics[scenario_train][predictand] = {}
    yPredLoaded[scenario_test][predictand] = {}
    yPredMetrics[scenario_test][predictand] = {}
    yPredLoaded[scenario_whole][predictand] = {}

    for predictand_number in predictand_numbered:
        modelName = f'DeepESD_tas_{predictand_number}'
        data_temp_train = xr.open_dataset(f'{PREDS_PATH_TRAIN}predTrain_{modelName}.nc')
        data_temp_test = xr.open_dataset(f'{PREDS_PATH_TRAIN}predTest_{modelName}.nc')

        yPredLoaded[scenario_whole][predictand][predictand_number] = xr.merge([data_temp_train, data_temp_test]).sel(time=slice(*(hist_baseline[0], hist_baseline[1])))
        yPredLoaded[scenario_train][predictand][predictand_number] = data_temp_train
        yPredLoaded[scenario_test][predictand][predictand_number] = data_temp_test

    yPredMetrics[scenario_train][predictand] = utils.getMetricsSimilarity(yPredLoaded[scenario_train][predictand])
    yPredMetrics[scenario_test][predictand] = utils.getMetricsSimilarity(yPredLoaded[scenario_test][predictand])


# VARIANCES metric = mean, 99quantile, 1quantile FOR TRAIN/TEST
scenarios = [scenario_train, scenario_test]
for scenario_name in scenarios:
    yMeanVariances[scenario_name] = utils.getVariance(yPredLoaded[scenario_name], metric='mean', percentage=True) 
    utils.graphVariances(yMeanVariances, scenario_name, FIGS_PATH, vmin=0, vmax=100, extra=f'Mean_Percentage_{len_predictands}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'Mean {scenario_name}')
    utils.graphVariancesMeanSd(yMeanVariances, scenario_name, FIGS_PATH, extra=f'Mean_Percentage_{len_predictands}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'Mean {scenario_name}')
    # 99 Quantile
    y99quanVariances[scenario_name] = utils.getVariance(yPredLoaded[scenario_name], metric='99quantile', percentage=True) 
    utils.graphVariances(y99quanVariances, scenario_name, FIGS_PATH, vmin=0, vmax=100, extra=f'99Quantile_Percentage_{len_predictands}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'99Percentil {scenario_name}')
    utils.graphVariancesMeanSd(y99quanVariances, scenario_name, FIGS_PATH, extra=f'99Quantile_Percentage_{len_predictands}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'99Percentil {scenario_name}')
    # 1 Quantile
    y1quanVariances[scenario_name] = utils.getVariance(yPredLoaded[scenario_name], metric='1quantile', percentage=True) 
    utils.graphVariances(y1quanVariances, scenario_name, FIGS_PATH, vmin=0, vmax=100, extra=f'1Quantile_Percentage_{len_predictands}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'1Percentile {scenario_name}')
    utils.graphVariancesMeanSd(y1quanVariances, scenario_name, FIGS_PATH, extra=f'1Quantile_Percentage_{len_predictands}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'1Percentile {scenario_name}')

# VARIANCES FOR CCSIGNAL

# Mean
yMeanVariances[scenario_ccsignal] = utils.getVariance(yPredLoaded[scenario], yPredLoaded[scenario_whole], metric='mean', percentage=True)
utils.graphVariances(yMeanVariances, scenario_ccsignal, FIGS_PATH, vmin=0, vmax=100, extra=f'Mean_Percentage_{len_predictands}_{extra_period}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'Mean {extra_period}')
utils.graphVariancesMeanSd(yMeanVariances, scenario_ccsignal, FIGS_PATH, extra=f'Mean_Percentage_{len_predictands}_{extra_period}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'Mean {extra_period}')
# 99 Quantile
y99quanVariances[scenario_ccsignal] = utils.getVariance(yPredLoaded[scenario], yPredLoaded[scenario_whole], metric='99quantile', percentage=True)
utils.graphVariances(y99quanVariances, scenario_ccsignal, FIGS_PATH, vmin=0, vmax=100, extra=f'99Quantile_Percentage_{len_predictands}_{extra_period}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'99Percentil {extra_period}')
utils.graphVariancesMeanSd(y99quanVariances, scenario_ccsignal, FIGS_PATH, extra=f'99Quantile_Percentage_{len_predictands}_{extra_period}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'99Percentil {extra_period}')
# 1 Quantile
y1quanVariances[scenario_ccsignal] = utils.getVariance(yPredLoaded[scenario], yPredLoaded[scenario_whole], metric='1quantile', percentage=True)
utils.graphVariances(y1quanVariances, scenario_ccsignal, FIGS_PATH, vmin=0, vmax=100, extra=f'1Quantile_Percentage_{len_predictands}_{extra_period}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'1Percentile {extra_period}')
utils.graphVariancesMeanSd(y1quanVariances, scenario_ccsignal, FIGS_PATH, extra=f'1Quantile_Percentage_{len_predictands}_{extra_period}_Quantity_{ENSEMBLE_QUANTITY}', extension='png', extra_title=f'1Percentile {extra_period}')


###########################SACAR T)OD=O CTRL K U DESDE ACA HACIA ARRIBA
# # IMPRESION DE GRAFICOS de 10 VALORES
# utils.graphMultipleTrains(yPredTenMetrics, figs_path=FIGS_PATH, vmin=12, vmax=27, var='mean', scenario=scenario)


# # IMPRESION DE GRAFICOS DE DESVIACION ESTANDAR Y MEDIA
# utils.graphSimilarityMetric(yPredMetrics[scenario_train], FIGS_PATH, 0, 1.5, 'std', scenario_train)
# utils.graphSimilarityMetric(yPredMetrics[scenario_train], FIGS_PATH, 5, 20, 'mean', scenario_train)
# utils.graphSimilarityMetric(yPredMetrics[scenario_test], FIGS_PATH, 0, 1.5, 'std', scenario_test)
# utils.graphSimilarityMetric(yPredMetrics[scenario_test], FIGS_PATH, 5, 20, 'mean', scenario_test)

# # IMPRESION D EGRAFICOS DE PORCENTAJE
# utils.graphSimilarityPercentage(yPredMetrics, FIGS_PATH, scenario_train, sigma_number=1)
# utils.graphSimilarityPercentage(yPredMetrics, FIGS_PATH, scenario_test, sigma_number=1)
# utils.graphSimilarityPercentage(yPredMetrics, FIGS_PATH, scenario_train, sigma_number=2)
# utils.graphSimilarityPercentage(yPredMetrics, FIGS_PATH, scenario_test, sigma_number=2)


# # IMPRESION DE GRAFICOS DE SIMILITUD DE PUNTOS DE LA GRID
# utils.graphSimilarityGrid(yPredMetrics, FIGS_PATH, scenario_train, sigma_number=1)
# utils.graphSimilarityGrid(yPredMetrics, FIGS_PATH, scenario_test, sigma_number=1)
# utils.graphSimilarityGrid(yPredMetrics, FIGS_PATH, scenario_train, sigma_number=2)
# utils.graphSimilarityGrid(yPredMetrics, FIGS_PATH, scenario_test, sigma_number=2)



print("Graphs Similarities Done!")

## GRAFICOS DE MEDIA Y SD
