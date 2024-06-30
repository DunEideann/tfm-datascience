import xarray as xr
import torch
from lib import utils, models, data, settings
import sys, time
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt

DATA_PATH_PREDICTORS = '/lustre/gmeteo/PTICLIMA/DATA/PROJECTIONS/CMIP6_PNACC/CMIP6_models/'
DATA_PATH_PREDICTANDS_READ = '/lustre/gmeteo/PTICLIMA/DATA/AUX/GRID_INTERCOMP/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/GCM/TFM2/'
MODELS_PATH = '/oceano/gmeteo/users/reyess/tfm/official-code/models'
DATA_PREDICTORS_TRANSFORMED = '/lustre/gmeteo/WORK/reyess/data/NorthAtlanticRegion_1.5degree/'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/'
VARIABLES_TO_DROP = ['lon_bnds', 'lat_bnds', 'crs']
GCM_NAME = 'EC-Earth3-Veg'
#TODO SACAR DE ESTE CODIGO TODO LO INECESARIO
#TODO Ciclos for no mas de 1??
scenarios = ['ssp585'] #'ssp126', 'ssp245', 'ssp370', 
hist_reference = ('1980-01-01', '2014-12-31')
hist_baseline = ('1995-01-01', '2014-12-31') #95-14
future_1 = ('2021-01-01', '2040-12-31')
future_2 = ('2041-01-01', '2060-12-31')
future_3 = ('2081-01-01', '2100-12-31') 
future = future_3
scenario = 'ssp370'
values_extended = True if scenario == 'ssp585' else False

yearsTrain = ('1980-01-01', '2003-12-31') #24 años
yearsTest = ('2004-01-01', '2015-12-31') #12 años

predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'pti-grid', 'CHELSA']
metrics = ['mean', '99quantile', '1quantile', 'std', 'over30']
seasons = {'spring': 'MAM', 'summer': 'JJA', 'autumn': 'SON', 'winter': 'DJF'}


# DATOS OBSERVACION
obs = {}
train_obs = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
train_obs_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
test_obs = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
test_obs_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
whole_obs = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
whole_obs_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}

for predictand_name in predictands:

    modelName = f'DeepESD_tas_{predictand_name}' 
    obs[predictand_name] = utils.getPredictand(DATA_PATH_PREDICTANDS_SAVE, predictand_name, 'tasmean')
    obs[predictand_name] = obs[predictand_name].sel(time=slice(*(yearsTrain[0], yearsTest[1])))
    obs[predictand_name] = utils.maskData(
                path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                var='tasmean',
                to_slice=(yearsTrain[0], yearsTest[1]),
                objective = obs[predictand_name],
                secondGrid = obs[predictand_name])
    train_obs['annual'][predictand_name] = obs[predictand_name].sel(time=slice(*(yearsTrain[0], yearsTrain[1])))
    test_obs['annual'][predictand_name] = obs[predictand_name].sel(time=slice(*(yearsTest[0], yearsTest[1])))
    whole_obs['annual'][predictand_name] = obs[predictand_name]

    train_obs_metrics['annual'][predictand_name] = utils.getMetricsTemp(train_obs['annual'][predictand_name], short = True)
    test_obs_metrics['annual'][predictand_name] = utils.getMetricsTemp(test_obs['annual'][predictand_name], short = True)
    whole_obs_metrics['annual'][predictand_name] = utils.getMetricsTemp(whole_obs['annual'][predictand_name], short = True)

    for season_name, months in seasons.items():
        train_obs[season_name][predictand_name] = train_obs['annual'][predictand_name].isel(time = (train_obs['annual'][predictand_name].time.dt.season == months))
        test_obs[season_name][predictand_name] = test_obs['annual'][predictand_name].isel(time = (test_obs['annual'][predictand_name].time.dt.season == months))
        whole_obs[season_name][predictand_name] = whole_obs['annual'][predictand_name].isel(time = (whole_obs['annual'][predictand_name].time.dt.season == months))

        train_obs_metrics[season_name][predictand_name] = utils.getMetricsTemp(train_obs[season_name][predictand_name], short = True)
        test_obs_metrics[season_name][predictand_name] = utils.getMetricsTemp(test_obs[season_name][predictand_name], short = True)
        whole_obs_metrics[season_name][predictand_name] = utils.getMetricsTemp(whole_obs[season_name][predictand_name], short = True)




# FIGURA 3
fig_num = 3
for period, data_metrics in whole_obs_metrics.items():
    utils.metricsGraph(data_metrics, figs_path=FIGS_PATH, vmin=[0, 0, -5, 0, 1], vmax=[35, 40, 15, 15, 501], pred_type='observation_whole', fig_num = fig_num, period = period, extension='png')
    utils.metricsGraph(test_obs_metrics[period], figs_path=FIGS_PATH, vmin=[0, 0, -5, 0, 1], vmax=[35, 40, 15, 15, 501], pred_type='observation_test', fig_num = fig_num, period = period, extension='png')
    utils.metricsGraph(train_obs_metrics[period], figs_path=FIGS_PATH, vmin=[0, 0, -5, 0, 1], vmax=[35, 40, 15, 15, 501], pred_type='observation_train', fig_num = fig_num, period = period, extension='png')
    fig_num += Decimal('0.1')

# DATOS PREDICCION
train_preds = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
train_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
test_preds = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
test_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
whole_preds = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
whole_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}

for predictand_name in predictands:

    modelName = f'DeepESD_tas_{predictand_name}' 
    train_preds['annual'][predictand_name] = xr.open_dataset(f'{PREDS_PATH}predTrain_{modelName}.nc')
    test_preds['annual'][predictand_name] = xr.open_dataset(f'{PREDS_PATH}predTest_{modelName}.nc')
    whole_preds['annual'][predictand_name] = xr.merge([train_preds['annual'][predictand_name], test_preds['annual'][predictand_name]])

    train_metrics['annual'][predictand_name] = utils.getMetricsTemp(train_preds['annual'][predictand_name], short = True)
    test_metrics['annual'][predictand_name] = utils.getMetricsTemp(test_preds['annual'][predictand_name], short = True)
    whole_metrics['annual'][predictand_name] = utils.getMetricsTemp(whole_preds['annual'][predictand_name], short = True)

    for season_name, months in seasons.items():
        train_preds[season_name][predictand_name] = train_preds['annual'][predictand_name].isel(time = (train_preds['annual'][predictand_name].time.dt.season == months))
        test_preds[season_name][predictand_name] = test_preds['annual'][predictand_name].isel(time = (test_preds['annual'][predictand_name].time.dt.season == months))
        whole_preds[season_name][predictand_name] = whole_preds['annual'][predictand_name].isel(time = (whole_preds['annual'][predictand_name].time.dt.season == months))

        train_metrics[season_name][predictand_name] = utils.getMetricsTemp(train_preds[season_name][predictand_name], short = True)
        test_metrics[season_name][predictand_name] = utils.getMetricsTemp(test_preds[season_name][predictand_name], short = True)
        whole_metrics[season_name][predictand_name] = utils.getMetricsTemp(whole_preds[season_name][predictand_name], short = True)

        

# FIGURA 5
fig_num = 5
for period, data_metrics in whole_metrics.items():
    utils.metricsGraph(data_metrics, figs_path=FIGS_PATH, vmin=[0, 0, -5, 0, 1], vmax=[35, 40, 15, 15, 501], pred_type='prediction_whole', fig_num = fig_num, period = period, extension='png')
    utils.metricsGraph(test_metrics[period], figs_path=FIGS_PATH, vmin=[0, 0, -5, 0, 1], vmax=[35, 40, 15, 15, 501], pred_type='prediction_test', fig_num = fig_num, period = period, extension='png')
    utils.metricsGraph(train_metrics[period], figs_path=FIGS_PATH, vmin=[0, 0, -5, 0, 1], vmax=[35, 40, 15, 15, 501], pred_type='prediction_train', fig_num = fig_num, period = period, extension='png')
    fig_num += Decimal('0.1')

# DATOS DIFERENCIA TEST PRED - OBS
diff_test_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}

for predictand_name in predictands:
    diff_test_metrics['annual'][predictand_name] = {
            key: test_metrics['annual'][predictand_name][key]-train_obs_metrics['annual'][predictand_name][key] 
                if key != 'std' 
                else test_metrics['annual'][predictand_name][key]/train_obs_metrics['annual'][predictand_name][key] 
                for key in metrics}
    for season_name, months in seasons.items():
        diff_test_metrics[season_name][predictand_name] = {
            key: test_metrics[season_name][predictand_name][key]-train_obs_metrics[season_name][predictand_name][key] 
                if key != 'std' 
                else test_metrics[season_name][predictand_name][key]/train_obs_metrics[season_name][predictand_name][key] 
                for key in metrics}

# # FIGURA 5
fig_num = 5
for period, data_metrics in diff_test_metrics.items():
    utils.metricsGraph(data_metrics, figs_path=FIGS_PATH, vmin=[-1, -1, -1, 0.3, 1], vmax=[1.5, 1.5, 1.5, 1.3, 11], pred_type='diff_test', fig_num = fig_num, period = period, extension='png', noWhite=True)
    fig_num += Decimal('0.1')


# PREPARACION EFEMERIDES
obs_efemeride = {}
preds_efemeride = {}

efemeride_days_train = {'cold': '2001-12-25' ,'hot': '2003-08-12'}
efemeride_days_test = {'cold': '2012-02-12' ,'hot': '2012-08-10'}

efe_train_obs_cold = {}
efe_train_obs_hot = {} 
efe_test_obs_cold = {}
efe_test_obs_hot = {}
efe_test_pred_cold = {} 
efe_test_pred_hot = {} 
efe_diff_cold = {}
efe_diff_hot = {}

for predictand_name in predictands:

    obs_efemeride[predictand_name] = obs[predictand_name].sel(time=slice(*(yearsTrain[0], yearsTest[1])))
    # obs_efemeride[predictand_name] = utils.maskData(
    #             path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
    #             var='tasmean',
    #             to_slice=(yearsTrain[0], yearsTest[1]),
    #             objective = obs_efemeride[predictand_name],
    #             secondGrid = obs_efemeride[predictand_name])
    
    preds_efemeride[predictand_name] = whole_preds['annual'][predictand_name].sel(time=slice(*(yearsTrain[0], yearsTest[1])))
    # preds_efemeride[predictand_name] = utils.maskData(
    #             path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
    #             var='tasmean',
    #             to_slice=(yearsTrain[0], yearsTest[1]),
    #             objective = preds_efemeride[predictand_name],
    #             secondGrid = preds_efemeride[predictand_name])
    
    efe_train_obs_cold[predictand_name] = obs_efemeride[predictand_name].sel(time=efemeride_days_train['cold'])
    efe_train_obs_hot[predictand_name] = obs_efemeride[predictand_name].sel(time=efemeride_days_train['hot'])
    #efe_train_pred_cold = preds_efemeride[predictand_name].sel(time=efemeride_days_train['cold'])
    #efe_train_pred_hot = preds_efemeride[predictand_name].sel(time=efemeride_days_train['hot'])
    efe_test_obs_cold[predictand_name] = obs_efemeride[predictand_name].sel(time=efemeride_days_train['cold'])
    efe_test_obs_hot[predictand_name] = obs_efemeride[predictand_name].sel(time=efemeride_days_train['hot'])
    efe_test_pred_cold[predictand_name] = preds_efemeride[predictand_name].sel(time=efemeride_days_train['cold'])
    efe_test_pred_hot[predictand_name] = preds_efemeride[predictand_name].sel(time=efemeride_days_train['hot'])

    efe_diff_cold[predictand_name] = efe_test_pred_cold[predictand_name] - efe_test_obs_cold[predictand_name]
    efe_diff_hot[predictand_name] = efe_test_pred_hot[predictand_name] - efe_test_obs_hot[predictand_name]


vmin = (-10, 15)
vmax = (15, 40)

# FIGURA 4
utils.efemerideGraph(efe_test_obs_cold, figs_path=FIGS_PATH, vmin=vmin[0], vmax=vmax[0], pred_type=f'efemeride_test_obs_cold', title='Coldwave day observations', fig_num = '4', extension='png')
utils.efemerideGraph(efe_test_obs_hot, figs_path=FIGS_PATH, vmin=vmin[1], vmax=vmax[1], pred_type=f'efemeride_test_obs_hot', title='Heatwave day observations', fig_num = '4', extension='png')
utils.efemerideGraph(efe_train_obs_cold, figs_path=FIGS_PATH, vmin=vmin[0], vmax=vmax[0], pred_type=f'efemeride_train_obs_cold', title='Coldwave day observations', fig_num = '4.1', extension='png')
utils.efemerideGraph(efe_train_obs_hot, figs_path=FIGS_PATH, vmin=vmin[1], vmax=vmax[1], pred_type=f'efemeride_train_obs_hot', title='Heatwave day observations', fig_num = '4.1', extension='png')


# FIGURA 6
utils.efemerideGraph(efe_diff_cold, figs_path=FIGS_PATH, vmin=-2.5, vmax=2.5, pred_type=f'efemeride_diff_cold', title='Coldwave day predictions', fig_num = '6', extension='png')
utils.efemerideGraph(efe_diff_hot, figs_path=FIGS_PATH, vmin=-2.5, vmax=2.5, pred_type=f'efemeride_diff_hot', title='Heatwave day predictions', fig_num = '6', extension='png')
utils.efemerideGraph(efe_test_pred_cold, figs_path=FIGS_PATH, vmin=vmin[0], vmax=vmax[0], pred_type=f'efemeride_test_pred_cold', title='Coldwave day predictions', fig_num = '6.1', extension='png')
utils.efemerideGraph(efe_test_pred_hot, figs_path=FIGS_PATH, vmin=vmin[1], vmax=vmax[1], pred_type=f'efemeride_test_pred_hot', title='Heatwave day predictions', fig_num = '6.1', extension='png')




# FIGURA 7:
long_preds = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
long_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
for predictand_name in predictands:
    modelName = f'DeepESD_tas_{predictand_name}_EC-Earth3-Veg_ssp585_2081-01-01-2100-12-31' 
    long_preds['annual'][predictand_name] = xr.open_dataset(f'{PREDS_PATH}GCM/predGCM_{modelName}.nc')
    long_metrics['annual'][predictand_name] = utils.getMetricsTemp(long_preds['annual'][predictand_name], short = True)
    for season_name, months in seasons.items():
        long_preds[season_name][predictand_name] = long_preds['annual'][predictand_name].isel(time = (long_preds['annual'][predictand_name].time.dt.season == months))
        long_metrics[season_name][predictand_name] = utils.getMetricsTemp(long_preds[season_name][predictand_name], short = True)

fig_num = 7
for period, data_metrics in long_metrics.items(): 
    utils.metricsGraph(data_metrics, figs_path=FIGS_PATH, vmin=[0, 0, -10, 0, 1], vmax=[40, 50, 20, 15, 3001], pred_type='ssp585', fig_num = fig_num, period = period, extension='png')
    fig_num += Decimal('0.1')

# FIGURA 8:
obs_hist = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
obs_hist_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
diff_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
for predictand_name in predictands:
    obs_hist['annual'][predictand_name] = obs[predictand_name].sel(time=slice(*(hist_baseline[0], hist_baseline[1])))
    obs_hist_metrics['annual'][predictand_name] = utils.getMetricsTemp(obs_hist['annual'][predictand_name], short = True)
    diff_metrics['annual'][predictand_name] = {
            key: long_metrics['annual'][predictand_name][key]-obs_hist_metrics['annual'][predictand_name][key] 
                if key != 'std' 
                else long_metrics['annual'][predictand_name][key]/obs_hist_metrics['annual'][predictand_name][key] 
                for key in metrics}
    for season_name, months in seasons.items():
        obs_hist[season_name][predictand_name] = obs_hist['annual'][predictand_name].isel(time = (obs_hist['annual'][predictand_name].time.dt.season == months))
        obs_hist_metrics[season_name][predictand_name] = utils.getMetricsTemp(obs_hist[season_name][predictand_name], short = True)
        diff_metrics[season_name][predictand_name] = {
            key: long_metrics[season_name][predictand_name][key]-obs_hist_metrics[season_name][predictand_name][key] 
                if key != 'std' 
                else long_metrics[season_name][predictand_name][key]/obs_hist_metrics[season_name][predictand_name][key] 
                for key in metrics}

fig_num = 8
for period, data_metrics in diff_metrics.items(): 
    utils.metricsGraph(data_metrics, figs_path=FIGS_PATH, vmin=[1, 2, 0, 0.6, 1], vmax=[11, 17, 10, 1.6, 3001], pred_type='climate_signal', fig_num = fig_num, period = period, extension='png')
    fig_num += Decimal('0.1')



# FIGURA EXTRA:
past_timeline = ('1970-01-01', '2020-01-01')
hist_baseline = ('1995-01-01', '2014-12-31') #95-14
future_1 = ('2021-01-01', '2040-12-31')
future_2 = ('2041-01-01', '2060-12-31')
future_3 = ('2081-01-01', '2100-12-31') 
future_4 = ('2061-01-01', '2080-12-31')
futures = [future_1, future_2, future_3, future_4]
main_scenerio = 'ssp585'

obs2 = {}
obs_temp = {}
gcm_preds = {}
hist_gcm = {}
hist_gcm_mean = {}
hist_gcm_mean_flat = {}

for predictand_name in predictands:

    gcms_futures = []
    modelName = f'DeepESD_tas_{predictand_name}' 

    obs2[predictand_name] = utils.getPredictand(f'{DATA_PATH_PREDICTANDS_SAVE}', predictand_name, 'tasmean')
    obs_temp[predictand_name] = obs2[predictand_name].sel(time=slice(*(yearsTrain[0], yearsTest[1])))
    obs2[predictand_name] = utils.maskData(
                path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                var='tasmean',
                to_slice=(yearsTrain[0], yearsTest[1]),
                objective = obs_temp[predictand_name],
                secondGrid = obs_temp[predictand_name])
    obs2[predictand_name] = obs2[predictand_name].sel(time=slice(*(past_timeline[0], past_timeline[1])))
    
    for future in futures:
        gcms_futures.append(xr.open_dataset(f'{PREDS_PATH}GCM/predGCM_{modelName}_{GCM_NAME}_{main_scenerio}_{future[0]}-{future[1]}.nc'))
    gcm_preds[predictand_name] = xr.merge(gcms_futures)
    hist_gcm[predictand_name] = xr.merge([obs2[predictand_name], gcm_preds[predictand_name]])
    hist_gcm_mean[predictand_name] = hist_gcm[predictand_name].resample(time = 'YE').mean()
    hist_gcm_mean[predictand_name] = hist_gcm_mean[predictand_name].mean(dim=['lat', 'lon'])#resample(time='1Y')
    hist_gcm_mean_flat[predictand_name] = hist_gcm_mean[predictand_name].tasmean.values.ravel()
    hist_gcm_mean_flat[predictand_name] = hist_gcm_mean_flat[predictand_name][~np.isnan(hist_gcm_mean_flat[predictand_name])]

# Crear una figura y un conjunto de ejes
figName = f'histogramFull'
plt.figure(figsize=(12, 8))

# Iterar sobre cada dataset en el diccionario
for key, dataset in hist_gcm_mean.items():
    # Extraer los valores de 'tasmean'
    tasmean_values = dataset['tasmean'].values
    # Extraer los años de la coordenada 'time'
    years = dataset['time'].dt.year

    # Graficar los valores de 'tasmean' contra los años
    plt.plot(years, tasmean_values, label=key)

# Configurar etiquetas y título
plt.xlabel('Year')
plt.ylabel('Tasmean')
#plt.title('Tasmean Values Over Time for Different Datasets')
plt.legend(loc='best')  # Añadir la leyenda

# Mostrar el gráfico
plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
plt.close()



# ANTIGUAS
# # FIGURA 3: Predicciones de TRAIN y metricas
# #Load predictions and metrics
# train_preds = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
# train_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
# test_preds = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
# test_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}

# for predictand_name in predictands:

#     modelName = f'DeepESD_tas_{predictand_name}' 
#     train_preds['annual'][predictand_name] = xr.open_dataset(f'{PREDS_PATH}predTrain_{modelName}.nc')
#     train_metrics['annual'][predictand_name] = utils.getMetricsTemp(train_preds['annual'][predictand_name], short = True)
#     for season_name, months in seasons.items():
#         train_preds[season_name][predictand_name] = train_preds['annual'][predictand_name].isel(time = (train_preds['annual'][predictand_name].time.dt.season == months))
#         train_metrics[season_name][predictand_name] = utils.getMetricsTemp(train_preds[season_name][predictand_name], short = True)
        
# fig_num = 3
# for period, data_metrics in train_metrics.items():
#     utils.metricsGraph(data_metrics, figs_path=FIGS_PATH, vmin=[0, 0, -10, 0, 1], vmax=[35, 40, 10, 10, 501], pred_type='train', fig_num = fig_num, period = period)
#     fig_num += Decimal('0.1')




# # FIGURA 5: Predicciones de TEST y metricas
# #Load predictions and metrics
# test_preds = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
# test_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
# for predictand_name in predictands:
#     modelName = f'DeepESD_tas_{predictand_name}' 
#     test_preds['annual'][predictand_name] = xr.open_dataset(f'{PREDS_PATH}predTest_{modelName}.nc')
#     test_metrics['annual'][predictand_name] = utils.getMetricsTemp(test_preds['annual'][predictand_name], short = True)
#     for season_name, months in seasons.items():
#         test_preds[season_name][predictand_name] = test_preds['annual'][predictand_name].isel(time = (test_preds['annual'][predictand_name].time.dt.season == months))
#         test_metrics[season_name][predictand_name] = utils.getMetricsTemp(test_preds[season_name][predictand_name], short = True)

# fig_num = 5
# for period, data_metrics in test_metrics.items():  
#     utils.metricsGraph(data_metrics, figs_path=FIGS_PATH, vmin=[0, 0, -10, 0, 1], vmax=[35, 40, 10, 10, 501], pred_type='test', fig_num = fig_num, period = period)
#     fig_num += Decimal('0.1')


# # FIGURA 4: 
# #diff_preds = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
# diff_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
# for predictand_name in predictands:
#     test_metrics['annual'][predictand_name]['over30'] = test_metrics['annual'][predictand_name]['over30']/(12*12)
#     train_metrics['annual'][predictand_name]['over30'] = train_metrics['annual'][predictand_name]['over30']/(24*12)
#     diff_metrics['annual'][predictand_name] = {
#             key: test_metrics['annual'][predictand_name][key]-train_metrics['annual'][predictand_name][key] 
#                 if key != 'std' 
#                 else test_metrics['annual'][predictand_name][key]/train_metrics['annual'][predictand_name][key] 
#                 for key in metrics}
#     for season_name, months in seasons.items():
#         test_metrics[season_name][predictand_name]['over30'] = test_metrics[season_name][predictand_name]['over30']/(12*3)
#         train_metrics[season_name][predictand_name]['over30'] = train_metrics[season_name][predictand_name]['over30']/(24*3)
#         diff_metrics[season_name][predictand_name] = {
#             key: test_metrics[season_name][predictand_name][key]-train_metrics[season_name][predictand_name][key] 
#                 if key != 'std' 
#                 else test_metrics[season_name][predictand_name][key]/train_metrics[season_name][predictand_name][key] 
#                 for key in metrics}

# fig_num = 4
# for period, data_metrics in diff_metrics.items(): 
#     utils.metricsGraph(data_metrics, figs_path=FIGS_PATH, vmin=[-0.2, 0.9, -1.0, -1.3, 0], vmax=[1.3, 1.15, 1.5, 1.2, 1], pred_type='diff', fig_num = fig_num, period = period, extension='pdf')
#     fig_num += Decimal('0.1')

# # FIGURA 6 y 7 a 10:
# long_preds = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
# long_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
# for predictand_name in predictands:
#     modelName = f'DeepESD_tas_{predictand_name}_EC-Earth3-Veg_ssp585_2081-01-01-2100-12-31' 
#     long_preds['annual'][predictand_name] = xr.open_dataset(f'{PREDS_PATH}GCM/predGCM_{modelName}.nc')
#     long_metrics['annual'][predictand_name] = utils.getMetricsTemp(long_preds['annual'][predictand_name], short = True)
#     for season_name, months in seasons.items():
#         long_preds[season_name][predictand_name] = long_preds['annual'][predictand_name].isel(time = (long_preds['annual'][predictand_name].time.dt.season == months))
#         long_metrics[season_name][predictand_name] = utils.getMetricsTemp(long_preds[season_name][predictand_name], short = True)

# fig_num = 6
# for period, data_metrics in long_metrics.items(): 
#     utils.metricsGraph(data_metrics, figs_path=FIGS_PATH, vmin=[0, 0, -10, 0, 1], vmax=[35, 50, 20, 15, 3001], pred_type='ssp585', fig_num = fig_num, period = period)
#     fig_num += 1


# # FIGURA 13: EFEMERIDE TEST OBS
# yearsTrain = ('1980-01-01', '2003-12-31') #24 años
# yearsTest = ('2004-01-01', '2015-12-31') #12 años

# obs_efemeride = {}
# obs_test_efemeride = {}
# efemeride_days = {'cold': '2012-02-12' ,'hot': '2012-08-10'}
# fig_num = 13

# for key, efemeride_day in efemeride_days.items():
#     for predictand_name in predictands:
#         obs_efemeride[predictand_name] = utils.getPredictand(DATA_PATH_PREDICTANDS_SAVE, predictand_name, 'tasmean')
#         obs_efemeride[predictand_name] = obs_efemeride[predictand_name].sel(time=slice(*(yearsTrain[0], yearsTest[1])))
#         obs_efemeride[predictand_name] = utils.maskData(
#                 path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
#                 var='tasmean',
#                 to_slice=(yearsTrain[0], yearsTest[1]),
#                 objective = obs_efemeride[predictand_name],
#                 secondGrid = obs_efemeride[predictand_name])
#         obs_test_efemeride[predictand_name] = obs_efemeride[predictand_name].sel(time=efemeride_day)
        
#     vmin = 15 if key == 'hot' else -10
#     vmax = 40 if key == 'hot' else 15
#     fig_num += 0.1
#     utils.efemerideGraph(obs_test_efemeride, figs_path=FIGS_PATH, vmin=vmin, vmax=vmax, pred_type=f'efemeride_test_obs_{key}', fig_num = fig_num)


# # FIGURA 14: EFEMERIDE TEST PREDICTION
# test_efemeride = {}
# fig_num = 14

# for key, efemeride_day in efemeride_days.items():
#     for predictand_name in predictands:
#         modelName = f'DeepESD_tas_{predictand_name}' 
#         test_efemeride[predictand_name] = xr.open_dataset(f'{PREDS_PATH}predTest_{modelName}.nc')
#         test_efemeride[predictand_name] = test_efemeride[predictand_name].sel(time=efemeride_day)
            
#     vmin = 15 if key == 'hot' else -10
#     vmax = 40 if key == 'hot' else 15
#     fig_num += 0.1
#     utils.efemerideGraph(test_efemeride, figs_path=FIGS_PATH, vmin=vmin, vmax=vmax, pred_type=f'efemeride_test_pred_{key}', fig_num = fig_num)

# # FIGURA 11: EFEMERIDE TRAIN OBS
# obs_train_efemeride = {}
# efemeride_days = {'cold': '2001-12-25' ,'hot': '2003-08-12'}

# fig_num = 11
# for key, efemeride_day in efemeride_days.items():
#     for predictand_name in predictands:
#         obs_train_efemeride[predictand_name] = obs_efemeride[predictand_name].sel(time=efemeride_day)
            
#     vmin = 15 if key == 'hot' else -10
#     vmax = 40 if key == 'hot' else 15
#     fig_num += 0.1
#     utils.efemerideGraph(obs_train_efemeride, figs_path=FIGS_PATH, vmin=vmin, vmax=vmax, pred_type=f'efemeride_train_obs_{key}', fig_num = fig_num)


# # FIGURA 12: EFEMERIDE TRAIN PREDICTION
# train_efemeride = {}
# fig_num = 12
# for key, efemeride_day in efemeride_days.items():
#     for predictand_name in predictands:
#         modelName = f'DeepESD_tas_{predictand_name}' 
#         train_efemeride[predictand_name] = xr.open_dataset(f'{PREDS_PATH}predTrain_{modelName}.nc')
#         train_efemeride[predictand_name] = train_efemeride[predictand_name].sel(time=efemeride_day)
            
#     vmin = 15 if key == 'hot' else -10
#     vmax = 40 if key == 'hot' else 15
#     fig_num += 0.1
#     utils.efemerideGraph(train_efemeride, figs_path=FIGS_PATH, vmin=vmin, vmax=vmax, pred_type=f'efemeride_train_pred_{key}', fig_num = fig_num)
    
