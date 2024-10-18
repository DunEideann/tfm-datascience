import xarray as xr
from lib import utils, models, data
import sys
import numpy as np
import matplotlib.pyplot as plt

FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/similarity'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/GCM/AEMET/'
PREDS_PATH_TRAIN = '/lustre/gmeteo/WORK/reyess/preds/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
# PERIOD = int(sys.argv[1])
# PREDICTANDS_SIZE = int(sys.argv[2])
# ENSEMBLE_QUANTITY = int(sys.argv[3])
PREDICTANDS_SIZE = 6
PERIOD = 3
ENSEMBLE_QUANTITY = 50
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





past_timeline = ('1970-01-01', '2020-12-31')
hist_baseline = ('1995-01-01', '2014-12-31') #95-14
futures = [future_1, future_2, future_3, future_4]
main_scenerio = 'ssp585'
gcm_name = 'EC-Earth3-Veg'



# EFFICIENT CODE:
# Crear una figura y un conjunto de ejes
figName = f'histogramFull_Ensemble{ENSEMBLE_QUANTITY}'
plt.figure(figsize=(12, 7))

xticks = np.linspace(1970, 2100, 14)

for predictand_name in predictands:

    gcms_futures = []
    modelName = f'DeepESD_tas_{predictand_name}' 

    obs2 = utils.getPredictand(f'{DATA_PATH_PREDICTANDS_SAVE}', predictand_name, 'tasmean')
    obs_temp = obs2.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
    obs2 = utils.maskData(
                path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                var='tasmean',
                to_slice=(yearsTrain[0], yearsTest[1]),
                objective = obs2.sel(time=slice(*(past_timeline[0], past_timeline[1]))),
                secondGrid = obs_temp)
    
    #PREDICTED DATA:
    hist_gcm_mean = {}
    temporal_mean = []
    predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
    for predictand_number in predictand_numbered:
        modelName = f'DeepESD_tas_{predictand_number}' 
        print(modelName)
        gcms_futures = []
        for future in futures:
            gcms_futures.append(xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{gcm_name}_{main_scenerio}_{future[0]}-{future[1]}.nc'))
        gcm_preds = xr.merge(gcms_futures)
        temp_gcm = xr.merge([obs2, gcm_preds])
        temp_gcm_mean = temp_gcm.resample(time = 'YE').mean()
        temp_gcm_mean = temp_gcm_mean.mean(dim=['lat', 'lon'])#resample(time='1Y')
        
        temporal_mean.append(temp_gcm_mean)

    temporal_mean_concat = xr.concat(temporal_mean, dim='member')
    hist_gcm_mean['mean'] = temporal_mean_concat.mean('member')
    hist_gcm_mean['max'] = temporal_mean_concat.max('member')
    hist_gcm_mean['min'] = temporal_mean_concat.min('member')

    del gcms_futures, temporal_mean, gcm_preds, obs2, obs_temp, temporal_mean_concat



    # Iterar sobre cada dataset en el diccionario
    # Extraer los valores de 'tasmean'
    tasmean_values = hist_gcm_mean['mean']['tasmean'].values
    # Extraer los años de la coordenada 'time'
    years = hist_gcm_mean['mean']['time'].dt.year

    # Graficar los valores de 'tasmean' contra los años
    plt.plot(years, tasmean_values, label=predictand_name)
    plt.fill_between(years, hist_gcm_mean['min']['tasmean'].values, hist_gcm_mean['max']['tasmean'].values, alpha=0.3)

# Añadir una línea punteada vertical en el año 2021
plt.axvline(x=2021, color='r', linestyle='--', linewidth=1)

# Configurar etiquetas y título
plt.xlabel('Year')
plt.ylabel('Tasmean')
plt.legend(loc='best')  # Añadir la leyenda

#EJE Y
yticks = [10, 12, 14, 16, 18, 20, 22, 24]
plt.yticks(yticks)
plt.xticks(xticks)

# Ajustar el espaciado entre los subplots
plt.tight_layout()

# Mostrar el gráfico
plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
plt.close()
del hist_gcm_mean, tasmean_values, years
print("Future predictions!")

# GRAFICO DE MEAN 6 GRAFICOS
# Definir los periodos actuales
#current_periods = [future_3, future_4, (future_1[0], future_3[1])]
fig_size = len(predictands)//5

#custom_xticks = np.linspace(yearsTrain[0], future_3[1])
# Iterar sobre los periodos
period_variance = (future_1[0], future_3[1])

# Crear una figura con 6 subplots (3 filas, 2 columnas) por cada periodo
fig, axes = plt.subplots(len(predictands), 1, figsize=(12, 2 + len(predictands)*2))

# Aplanar la matriz de ejes (axes) para iterar más fácilmente
axes = axes.flatten()

# Iterador para asignar cada gráfico a un subplot
plot_idx = 0
for predictand_name in predictands:

    gcms_futures = []
    modelName = f'DeepESD_tas_{predictand_name}' 

    #PREDICTED DATA:
    hist_gcm_mean = {}
    temporal_mean = []
    predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
    for predictand_number in predictand_numbered:
        modelName = f'DeepESD_tas_{predictand_number}' 
        print(modelName)
        gcms_futures = []
        for future in futures:
            gcms_futures.append(xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{gcm_name}_{main_scenerio}_{future[0]}-{future[1]}.nc'))
        gcm_preds = xr.merge(gcms_futures)
        # hist_gcm = xr.merge([obs2, gcm_preds])
        hist_gcm = gcm_preds
        hist_gcm_mean[predictand_number] = hist_gcm.resample(time = 'YE').mean()
        hist_gcm_mean[predictand_number] = hist_gcm_mean[predictand_number].mean(dim=['lat', 'lon'])#resample(time='1Y')
        
        temporal_mean.append(hist_gcm_mean[predictand_number])

    temporal_mean_concat = xr.concat(temporal_mean, dim='member')
    hist_gcm_mean['median'] = temporal_mean_concat.median('member')
    hist_gcm_mean['max'] = temporal_mean_concat.max('member')
    hist_gcm_mean['min'] = temporal_mean_concat.min('member')


    # CLEANING MEMORY
    del gcms_futures, temporal_mean, temporal_mean_concat, gcm_preds, hist_gcm

    ax = axes[plot_idx]  # Selecciona el subplot actual

    # Extraer los valores de 'tasmean'
    dataset_filtered = hist_gcm_mean['median'].sel(time=slice(yearsTrain[0], period_variance[1]))
    dataset_min_filtered = hist_gcm_mean['min'].sel(time=slice(period_variance[0], period_variance[1]))['tasmean'].values
    dataset_max_filtered = hist_gcm_mean['max'].sel(time=slice(period_variance[0], period_variance[1]))['tasmean'].values
    tasmean_values = dataset_filtered['tasmean'].values
    years = dataset_filtered['time'].dt.year

    # Graficar los valores de 'tasmean' contra los años
    ax.plot(years, tasmean_values - tasmean_values, label=predictand_name)
    ax.fill_between(years, dataset_min_filtered, dataset_max_filtered, alpha=0.3)

    for key_number, numbered_set in hist_gcm_mean.items():
        if key_number not in ['median', 'min', 'max']:
            numbered_set_filtered = numbered_set.sel(time=slice(period_variance[0], period_variance[1]))
            years_number = numbered_set_filtered['time'].dt.year
            tasmean_values_realization = numbered_set_filtered['tasmean'].values - tasmean_values
            ax.plot(years_number, tasmean_values_realization, linestyle='dashed', color='red', linewidth=0.5)

    # Establecer límites para el eje Y
    ax.set_ylim(-1, 1)
    
    # Configurar etiquetas
    ax.set_xlabel('Year')
    ax.set_ylabel('Tasmean')
    ax.legend(loc='best')
    ax.set_xticks(xticks)

    # Incrementar el índice para el siguiente subplot
    plot_idx += 1

# Ajustar el espaciado entre los subplots
plt.tight_layout()

# Guardar la figura para este periodo
figName = f'histogramVariance_Mean_{period_variance[0]}-{period_variance[1]}_Datasets={len(predictands)}_Ensemble{ENSEMBLE_QUANTITY}'
plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
plt.close()
del hist_gcm_mean
print("CC Signal Mean")

# GRAFICO DE 99Percentil 6 GRAFICOS
# Iterar sobre los periodos
plt.figure(figsize=(12, 7))
#current_periods = [future_3, future_4, (future_1[0], future_3[1])]
fig_size = len(predictands)//5
period_variance = (future_1[0], future_3[1])
# Crear una figura con 6 subplots (3 filas, 2 columnas) por cada periodo
fig, axes = plt.subplots(len(predictands), 1, figsize=(12, 2 + len(predictands)*2))

# Aplanar la matriz de ejes (axes) para iterar más fácilmente
axes = axes.flatten()

# Iterador para asignar cada gráfico a un subplot
plot_idx = 0

for predictand_name in predictands:

    gcms_futures = []
    
    #PREDICTED DATA:
    hist_gcm_99 = {}
    temporal_99 = []
    predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
    for predictand_number in predictand_numbered:
        modelName = f'DeepESD_tas_{predictand_number}' 
        print(modelName)
        gcms_futures = []
        for future in futures:
            gcms_futures.append(xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{gcm_name}_{main_scenerio}_{future[0]}-{future[1]}.nc'))
        gcm_preds = xr.merge(gcms_futures)
        #hist_gcm = xr.merge([obs2, gcm_preds])
        hist_gcm = gcm_preds
        hist_gcm_99[predictand_number] = hist_gcm.resample(time = 'YE').quantile(0.99, dim = 'time')
        hist_gcm_99[predictand_number] = hist_gcm_99[predictand_number].mean(dim=['lat', 'lon'])
        

        temporal_99.append(hist_gcm_99[predictand_number])

    temporal_99_concat = xr.concat(temporal_99, dim='member')
    hist_gcm_99['median'] = temporal_99_concat.median('member')
    hist_gcm_99['max'] = temporal_99_concat.max('member')
    hist_gcm_99['min'] = temporal_99_concat.min('member')

    # CLEANING MEMORY
    del gcms_futures, temporal_99, temporal_99_concat, gcm_preds, hist_gcm


    ax = axes[plot_idx]  # Selecciona el subplot actual

    # Extraer los valores de 'tasmean'
    dataset_filtered = hist_gcm_99['median'].sel(time=slice(yearsTrain[0], period_variance[1]))
    dataset_min_filtered = hist_gcm_99['min'].sel(time=slice(period_variance[0], period_variance[1]))['tasmean'].values
    dataset_max_filtered = hist_gcm_99['max'].sel(time=slice(period_variance[0], period_variance[1]))['tasmean'].values
    tasmean_values = dataset_filtered['tasmean'].values
    years = dataset_filtered['time'].dt.year

    # Graficar los valores de 'tasmean' contra los años
    ax.plot(years, tasmean_values - tasmean_values, label=predictand_name)
    ax.fill_between(years, dataset_min_filtered, dataset_max_filtered, alpha=0.3)

    for key_number, numbered_set in hist_gcm_99.items():
        if key_number not in ['median', 'min', 'max']:
            numbered_set_filtered = numbered_set.sel(time=slice(period_variance[0], period_variance[1]))
            years_number = numbered_set_filtered['time'].dt.year
            tasmean_values_realization = numbered_set_filtered['tasmean'].values - tasmean_values
            ax.plot(years_number, tasmean_values_realization, linestyle='dashed', color='red', linewidth=0.5)

    # Establecer límites para el eje Y
    ax.set_ylim(-2, 2)

    # Configurar etiquetas
    ax.set_xlabel('Year')
    ax.set_ylabel('Tasmean')
    ax.legend(loc='best')
    ax.set_xticks(xticks)

    # Incrementar el índice para el siguiente subplot
    plot_idx += 1

# Ajustar el espaciado entre los subplots
plt.tight_layout()

# Guardar la figura para este periodo
figName = f'histogramVariance_99Quantile_{period_variance[0]}-{period_variance[1]}_Datasets={len(predictands)}_Ensemble{ENSEMBLE_QUANTITY}'
plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
plt.close()

print("CC Signal 99")
del hist_gcm_99

########################################################################
############ TRAIN TEST#############################################
###################################################################
xticks = np.linspace(yearsTrain[0], yearsTest[1], 8)
fig_size = len(predictands)//5
# Iterar sobre los periodos
period_variance = (yearsTrain[0], yearsTest[1])

# Crear una figura con 6 subplots (3 filas, 2 columnas) por cada periodo
fig, axes = plt.subplots(len(predictands), 1, figsize=(12, 2 + len(predictands)*2))

# Aplanar la matriz de ejes (axes) para iterar más fácilmente
axes = axes.flatten()

# Iterador para asignar cada gráfico a un subplot
plot_idx = 0
for predictand_name in predictands:

   
    #PREDICTED DATA:
    whole_mean = {}
    temporal_mean = []
    predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
    for predictand_number in predictand_numbered:
        modelName = f'DeepESD_tas_{predictand_number}' 
        print(modelName)
        preds_test = xr.open_dataset(f'{PREDS_PATH_TRAIN}/predTest_{modelName}.nc')
        preds_train = xr.open_dataset(f'{PREDS_PATH_TRAIN}/predTrain_{modelName}.nc')
        whole_preds = xr.merge([preds_train, preds_test])
        whole_mean[predictand_number] = whole_preds.resample(time = 'YE').mean()
        whole_mean[predictand_number] = whole_mean[predictand_number].mean(dim=['lat', 'lon'])#resample(time='1Y')
        
        temporal_mean.append(whole_mean[predictand_number])

    temporal_mean_concat = xr.concat(temporal_mean, dim='member')
    whole_mean['median'] = temporal_mean_concat.median('member')
    whole_mean['max'] = temporal_mean_concat.max('member')
    whole_mean['min'] = temporal_mean_concat.min('member')

    del temporal_mean, temporal_mean_concat, whole_preds, preds_test, preds_train

    ax = axes[plot_idx]  # Selecciona el subplot actual

    # Extraer los valores de 'tasmean'
    dataset_filtered = whole_mean['median'].sel(time=slice(period_variance[0], period_variance[1]))
    dataset_min_filtered = whole_mean['min'].sel(time=slice(period_variance[0], period_variance[1]))['tasmean'].values
    dataset_max_filtered = whole_mean['max'].sel(time=slice(period_variance[0], period_variance[1]))['tasmean'].values
    tasmean_values = dataset_filtered['tasmean'].values
    years = dataset_filtered['time'].dt.year

    # Graficar los valores de 'tasmean' contra los años
    ax.plot(years, tasmean_values - tasmean_values, label=predictand_name)
    ax.fill_between(years, dataset_min_filtered, dataset_max_filtered, alpha=0.3)

    for key_number, numbered_set in whole_mean.items():
        if key_number not in ['median', 'min', 'max']:
            numbered_set_filtered = numbered_set.sel(time=slice(period_variance[0], period_variance[1]))
            years_number = numbered_set_filtered['time'].dt.year
            tasmean_values_realization = numbered_set_filtered['tasmean'].values - tasmean_values
            ax.plot(years_number, tasmean_values_realization, linestyle='dashed', color='red', linewidth=0.5)

    # Establecer límites para el eje Y
    ax.set_ylim(-1, 1)
    
    # Configurar etiquetas
    ax.set_xlabel('Year')
    ax.set_ylabel('Tasmean')
    ax.legend(loc='best')
    ax.set_xticks(xticks)

    # Incrementar el índice para el siguiente subplot
    plot_idx += 1

# Ajustar el espaciado entre los subplots
plt.tight_layout()

# Guardar la figura para este periodo
figName = f'histogramTestTrain_Mean_{period_variance[0]}-{period_variance[1]}_Datasets={len(predictands)}_Ensemble{ENSEMBLE_QUANTITY}'
plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
plt.close()

del whole_mean
print("TestTrain Mean")

##### TEST TAIN QUANTILE 99#############

# Crear una figura con 6 subplots (3 filas, 2 columnas) por cada periodo
fig, axes = plt.subplots(len(predictands), 1, figsize=(12, 2 + len(predictands)*2))

# Aplanar la matriz de ejes (axes) para iterar más fácilmente
axes = axes.flatten()

# Iterador para asignar cada gráfico a un subplot
plot_idx = 0
for predictand_name in predictands:

   
    #PREDICTED DATA:
    whole_99 = {}
    temporal_99 = []
    predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
    for predictand_number in predictand_numbered:
        modelName = f'DeepESD_tas_{predictand_number}' 
        print(modelName)
        preds_test = xr.open_dataset(f'{PREDS_PATH_TRAIN}/predTest_{modelName}.nc')
        preds_train = xr.open_dataset(f'{PREDS_PATH_TRAIN}/predTrain_{modelName}.nc')
        whole_preds = xr.merge([preds_train, preds_test])
        whole_99[predictand_number] = whole_preds.resample(time = 'YE').quantile(0.99, dim = 'time')
        whole_99[predictand_number] = whole_99[predictand_number].mean(dim=['lat', 'lon'])
        
        temporal_99.append(whole_99[predictand_number])

    temporal_99_concat = xr.concat(temporal_99, dim='member')
    whole_99['median'] = temporal_99_concat.median('member')
    whole_99['max'] = temporal_99_concat.max('member')
    whole_99['min'] = temporal_99_concat.min('member')

    del temporal_99, temporal_99_concat, whole_preds, preds_test, preds_train

    ax = axes[plot_idx]  # Selecciona el subplot actual

    # Extraer los valores de 'tasmean'
    dataset_filtered = whole_99['median'].sel(time=slice(period_variance[0], period_variance[1]))
    dataset_min_filtered = whole_99['min'].sel(time=slice(period_variance[0], period_variance[1]))['tasmean'].values
    dataset_max_filtered = whole_99['max'].sel(time=slice(period_variance[0], period_variance[1]))['tasmean'].values
    tasmean_values = dataset_filtered['tasmean'].values
    years = dataset_filtered['time'].dt.year

    # Graficar los valores de 'tasmean' contra los años
    ax.plot(years, tasmean_values - tasmean_values, label=predictand_name)
    ax.fill_between(years, dataset_min_filtered, dataset_max_filtered, alpha=0.3)

    for key_number, numbered_set in whole_99.items():
        if key_number not in ['median', 'min', 'max']:
            numbered_set_filtered = numbered_set.sel(time=slice(period_variance[0], period_variance[1]))
            years_number = numbered_set_filtered['time'].dt.year
            tasmean_values_realization = numbered_set_filtered['tasmean'].values - tasmean_values
            ax.plot(years_number, tasmean_values_realization, linestyle='dashed', color='red', linewidth=0.5)

    # Establecer límites para el eje Y
    ax.set_ylim(-1, 1)
    
    # Configurar etiquetas
    ax.set_xlabel('Year')
    ax.set_ylabel('Tasmean')
    ax.legend(loc='best')
    ax.set_xticks(xticks)

    # Incrementar el índice para el siguiente subplot
    plot_idx += 1

# Ajustar el espaciado entre los subplots
plt.tight_layout()

# Guardar la figura para este periodo
figName = f'histogramTestTrain_99percentil_{period_variance[0]}-{period_variance[1]}_Datasets={len(predictands)}_Ensemble{ENSEMBLE_QUANTITY}'
plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
plt.close()
print("TestTrain 99")