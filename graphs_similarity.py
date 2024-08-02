import xarray as xr
from lib import utils, models, data
import sys

FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/GCM/AEMET/'
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
main_scenerio = 'ssp370'
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


yPredLoaded = {}
yPredMetrics = {}
yPredLoaded[scenario] = {}
yPredMetrics[scenario] = {}

for predictand in predictands:
    predictand_numbered = [f"{predictand}_{i}" for i in range(1, 11)]
    yPredLoaded[scenario][predictand] = {}
    yPredMetrics[scenario][predictand] = {}

    for predictand_number in predictand_numbered:
        modelName = f'DeepESD_tas_{predictand_number}'
        yPredLoaded[scenario][predictand][predictand_number] = xr.open_dataset(f'{PREDS_PATH}predGCM_{modelName}_{PREDICTOR}_{scenario}_{period[0]}-{period[1]}.nc')

    yPredMetrics[scenario][predictand] = utils.getMetricsSimilarity(yPredLoaded[scenario][predictand])


utils.graphSimilarityPercentage(yPredMetrics, FIGS_PATH, scenario, sigma_number=1)
utils.graphSimilarityPercentage(yPredMetrics, FIGS_PATH, scenario, sigma_number=2)

utils.graphSimilarityGrid(yPredMetrics, FIGS_PATH, scenario, sigma_number=1)
utils.graphSimilarityGrid(yPredMetrics, FIGS_PATH, scenario, sigma_number=1)


print("Graphs Similarities Done!")


