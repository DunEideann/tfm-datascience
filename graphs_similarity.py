import xarray as xr
from lib import utils, models, data
import sys
import numpy as np

FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/similarity'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/GCM/AEMET/'
PREDS_PATH_TRAIN = '/lustre/gmeteo/WORK/reyess/preds/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
PERIOD = int(sys.argv[1])
PREDICTANDS_SIZE = int(sys.argv[2])
# PREDICTANDS_SIZE = 6
# PERIOD = 3
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
# ['AEMET_0.25deg', 'Iberia01_v1.0', 'CHELSA'] PARA VER 3 VARIANZAS
# CARGA DE PREDICCIONES
# yPredLoaded = {scenario: {}, scenario_train: {}, scenario_test: {}, scenario_whole: {}}
# yPredMetrics = {scenario: {}, scenario_train: {}, scenario_test: {}, scenario_ccsignal: {}}

# yPredTenMetrics = {scenario: {}}
# yMeanVariances = {scenario: {}, scenario_train: {}, scenario_test: {}, scenario_ccsignal: {}}
# y99quanVariances = {scenario: {}, scenario_train: {}, scenario_test: {}, scenario_ccsignal: {}}
# y1quanVariances = {scenario: {}, scenario_train: {}, scenario_test: {}, scenario_ccsignal: {}}


# # PARA PERIODOS FUTUROS
# for predictand in predictands:
#     predictand_numbered = [f"{predictand}_{i}" for i in range(1, 11)]
#     yPredLoaded[scenario][predictand] = {}
#     yPredMetrics[scenario][predictand] = {}
#     yPredTenMetrics[scenario][predictand] = {}

#     for predictand_number in predictand_numbered:
#         modelName = f'DeepESD_tas_{predictand_number}'
#         print(modelName)
#         data_temp = xr.open_dataset(f'{PREDS_PATH}predGCM_{modelName}_{PREDICTOR}_{scenario}_{period[0]}-{period[1]}.nc')

#         yPredLoaded[scenario][predictand][predictand_number] = xr.open_dataset(f'{PREDS_PATH}predGCM_{modelName}_{PREDICTOR}_{scenario}_{period[0]}-{period[1]}.nc')
#         yPredTenMetrics[scenario][predictand][predictand_number] = utils.getMetricsTemp(yPredLoaded[scenario][predictand][predictand_number], var = 'tasmean', short = True)

#     yPredMetrics[scenario][predictand] = utils.getMetricsSimilarity(yPredLoaded[scenario][predictand])

# # VARIANCES metric = mean, 99quantile, 1quantile, sd 
# yMeanVariances[scenario] = utils.getVariance(yPredLoaded[scenario], metric='mean', percentage=True) 
# utils.graphVariances(yMeanVariances, scenario, FIGS_PATH, vmin=0, vmax=100, extra=f'{period[0]}-{period[1]}_Mean_Percentage_{len_predictands}', extension='png')
# # 99 Quantile
# y99quanVariances[scenario] = utils.getVariance(yPredLoaded[scenario], metric='99quantile', percentage=True) 
# utils.graphVariances(y99quanVariances, scenario, FIGS_PATH, vmin=0, vmax=100, extra=f'{period[0]}-{period[1]}_99Quantile_Percentage_{len_predictands}', extension='png')
# # 1 Quantile
# y1quanVariances[scenario] = utils.getVariance(yPredLoaded[scenario], metric='1quantile', percentage=True) 
# utils.graphVariances(y1quanVariances, scenario, FIGS_PATH, vmin=0, vmax=100, extra=f'{period[0]}-{period[1]}_1Quantile_Percentage_{len_predictands}', extension='png')



# # # # IMPRESION DE 10 MODELOS POR PREDICTANDO
# # # #figs_path, vmin = 10, vmax = 30, var = 'mean', scenario = 'ssp585', extension = 'pdf', color='hot_r', color_change = None):
# # # utils.graphMultipleTrains(yPredTenMetrics, figs_path=FIGS_PATH, vmin=12, vmax=27, var='mean', scenario=scenario)

# # # #IMPRESION DE GRAFICOS DE DESVIACION ESTANDAR Y MEDIA
# # # utils.graphSimilarityMetric(yPredMetrics['ssp585'], FIGS_PATH, 0, 1.5, 'std')
# # # utils.graphSimilarityMetric(yPredMetrics['ssp585'], FIGS_PATH, 15, 25, 'mean')


# # # # IMPRESION D EGRAFICOS DE PORCENTAJE
# # # utils.graphSimilarityPercentage(yPredMetrics, FIGS_PATH, scenario, sigma_number=1)
# # # utils.graphSimilarityPercentage(yPredMetrics, FIGS_PATH, scenario, sigma_number=2)


# # # # IMPRESION DE GRAFICOS DE SIMILITUD DE PUNTOS DE LA GRID
# # # utils.graphSimilarityGrid(yPredMetrics, FIGS_PATH, scenario, sigma_number=1)
# # # utils.graphSimilarityGrid(yPredMetrics, FIGS_PATH, scenario, sigma_number=2)

# # print("Graphs Similarities Done!")




# # ######################################################
# # # CARGA DE PREDICCIONES
# # #TODO Agregar scenarios a un solo diccionario y no muchos distintos

# # yPredMetricsTestTen = {scenario_test: {}}

# # PARA TRAIN/TEST BASICOS
# for predictand in predictands:
#     predictand_numbered = [f"{predictand}_{i}" for i in range(1, 11)]
#     yPredLoaded[scenario_train][predictand] = {}
#     yPredMetrics[scenario_train][predictand] = {}
#     yPredLoaded[scenario_test][predictand] = {}
#     yPredMetrics[scenario_test][predictand] = {}
#     yPredLoaded[scenario_whole][predictand] = {}

#     for predictand_number in predictand_numbered:
#         modelName = f'DeepESD_tas_{predictand_number}'
#         data_temp_train = xr.open_dataset(f'{PREDS_PATH_TRAIN}predTrain_{modelName}.nc')
#         data_temp_test = xr.open_dataset(f'{PREDS_PATH_TRAIN}predTest_{modelName}.nc')

#         yPredLoaded[scenario_whole][predictand][predictand_number] = xr.merge([data_temp_train, data_temp_test]).sel(time=slice(*(hist_baseline[0], hist_baseline[1])))
#         yPredLoaded[scenario_train][predictand][predictand_number] = data_temp_train
#         yPredLoaded[scenario_test][predictand][predictand_number] = data_temp_test

#     yPredMetrics[scenario_train][predictand] = utils.getMetricsSimilarity(yPredLoaded[scenario_train][predictand])
#     yPredMetrics[scenario_test][predictand] = utils.getMetricsSimilarity(yPredLoaded[scenario_test][predictand])


# # VARIANCES metric = mean, 99quantile, 1quantile FOR TRAIN/TEST
# scenarios = [scenario_train, scenario_test]
# for scenario_name in scenarios:
#     yMeanVariances[scenario_name] = utils.getVariance(yPredLoaded[scenario_name], metric='mean', percentage=True) 
#     utils.graphVariances(yMeanVariances, scenario_name, FIGS_PATH, vmin=0, vmax=100, extra=f'Mean_Percentage_{len_predictands}', extension='png')
#     # 99 Quantile
#     y99quanVariances[scenario_name] = utils.getVariance(yPredLoaded[scenario_name], metric='99quantile', percentage=True) 
#     utils.graphVariances(y99quanVariances, scenario_name, FIGS_PATH, vmin=0, vmax=100, extra=f'99Quantile_Percentage_{len_predictands}', extension='png')
#     # 1 Quantile
#     y1quanVariances[scenario_name] = utils.getVariance(yPredLoaded[scenario_name], metric='1quantile', percentage=True) 
#     utils.graphVariances(y1quanVariances, scenario_name, FIGS_PATH, vmin=0, vmax=100, extra=f'1Quantile_Percentage_{len_predictands}', extension='png')

# # VARIANCES FOR CCSIGNAL
# extra_period = 'MEDIUM' if PERIOD == 4 else 'LONG'
# # Mean
# yMeanVariances[scenario_ccsignal] = utils.getVariance(yPredLoaded[scenario], yPredLoaded[scenario_whole], metric='mean', percentage=True)
# utils.graphVariances(yMeanVariances, scenario_ccsignal, FIGS_PATH, vmin=0, vmax=100, extra=f'Mean_Percentage_{len_predictands}_{extra_period}', extension='png')
# # 99 Quantile
# yMeanVariances[scenario_ccsignal] = utils.getVariance(yPredLoaded[scenario], yPredLoaded[scenario_whole], metric='99quantile', percentage=True)
# utils.graphVariances(yMeanVariances, scenario_ccsignal, FIGS_PATH, vmin=0, vmax=100, extra=f'99Quantile_Percentage_{len_predictands}_{extra_period}', extension='png')
# # 1 Quantile
# yMeanVariances[scenario_ccsignal] = utils.getVariance(yPredLoaded[scenario], yPredLoaded[scenario_whole], metric='1quantile', percentage=True)
# utils.graphVariances(yMeanVariances, scenario_ccsignal, FIGS_PATH, vmin=0, vmax=100, extra=f'1Quantile_Percentage_{len_predictands}_{extra_period}', extension='png')


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


# GRAPH HISTORICAL TASMEAN
# NECESITA AL MENOS 64GB (128 Ideal), y no es eficiente para nada (REPENSAR LA FORMA)
if len_predictands == 6 and PERIOD == 3:
    import matplotlib.pyplot as plt

    past_timeline = ('1970-01-01', '2020-12-31')
    hist_baseline = ('1995-01-01', '2014-12-31') #95-14
    futures = [future_1, future_2, future_3, future_4]
    main_scenerio = 'ssp585'
    gcm_name = 'EC-Earth3-Veg'

    obs2 = {}
    obs_temp = {}
    gcm_preds = {}
    hist_gcm = {}
    hist_gcm_mean = {}
    hist_gcm_99 = {}

    for predictand_name in predictands:

        gcms_futures = []
        modelName = f'DeepESD_tas_{predictand_name}' 

        obs2[predictand_name] = utils.getPredictand(f'{DATA_PATH_PREDICTANDS_SAVE}', predictand_name, 'tasmean')
        obs_temp[predictand_name] = obs2[predictand_name].sel(time=slice(*(yearsTrain[0], yearsTest[1])))
        obs2[predictand_name] = utils.maskData(
                    path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                    var='tasmean',
                    to_slice=(yearsTrain[0], yearsTest[1]),
                    objective = obs2[predictand_name].sel(time=slice(*(past_timeline[0], past_timeline[1]))),
                    secondGrid = obs_temp[predictand_name])
        
        #PREDICTED DATA:
        gcm_preds[predictand_name] = {}
        hist_gcm[predictand_name] = {}
        hist_gcm_mean[predictand_name] = {}
        hist_gcm_99[predictand_name] = {}
        temporal_mean = []
        temporal_99 = []
        predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, 11)]
        for predictand_number in predictand_numbered:
            modelName = f'DeepESD_tas_{predictand_number}' 
            print(modelName)
            gcms_futures = []
            for future in futures:
                gcms_futures.append(xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{gcm_name}_{main_scenerio}_{future[0]}-{future[1]}.nc'))
            gcm_preds[predictand_name][predictand_number] = xr.merge(gcms_futures)
            hist_gcm[predictand_name][predictand_number] = xr.merge([obs2[predictand_name], gcm_preds[predictand_name][predictand_number]])
            hist_gcm_mean[predictand_name][predictand_number] = hist_gcm[predictand_name][predictand_number].resample(time = 'YE').mean()
            hist_gcm_mean[predictand_name][predictand_number] = hist_gcm_mean[predictand_name][predictand_number].mean(dim=['lat', 'lon'])#resample(time='1Y')
            hist_gcm_99[predictand_name][predictand_number] = hist_gcm[predictand_name][predictand_number].resample(time = 'YE').quantile(0.99, dim = 'time')
            hist_gcm_99[predictand_name][predictand_number] = hist_gcm_99[predictand_name][predictand_number].mean(dim=['lat', 'lon'])
            
            temporal_mean.append(hist_gcm_mean[predictand_name][predictand_number])
            temporal_99.append(hist_gcm_99[predictand_name][predictand_number])

        temporal_mean_concat = xr.concat(temporal_mean, dim='member')
        temporal_99_concat = xr.concat(temporal_99, dim='member')
        hist_gcm_mean[predictand_name]['mean'] = temporal_mean_concat.mean('member')
        hist_gcm_mean[predictand_name]['max'] = temporal_mean_concat.max('member')
        hist_gcm_mean[predictand_name]['min'] = temporal_mean_concat.min('member')
        hist_gcm_99[predictand_name]['mean'] = temporal_99_concat.mean('member')
        hist_gcm_99[predictand_name]['max'] = temporal_99_concat.max('member')
        hist_gcm_99[predictand_name]['min'] = temporal_99_concat.min('member')

        gcms_futures = []
        temporal_mean = []



    # GRAFICO DE VARIACION DE REALIZACION (PANELES JUNTOS)
    # Definir los periodos actuales
    current_periods = [future_3, future_4, (future_1[0], future_3[1])]
    fig_size = len(predictands)//5
    # Iterar sobre los periodos
    for period_variance in current_periods:
        # Crear una figura con 6 subplots (3 filas, 2 columnas) por cada periodo
        fig, axes = plt.subplots(2+fig_size, 2, figsize=(14, 12+(6*fig_size)))  # 3 filas, 2 columnas

        # Aplanar la matriz de ejes (axes) para iterar más fácilmente
        axes = axes.flatten()

        # Iterador para asignar cada gráfico a un subplot
        plot_idx = 0

        for key, dataset in hist_gcm_mean.items():
            ax = axes[plot_idx]  # Selecciona el subplot actual

            # Extraer los valores de 'tasmean'
            dataset_filtered = dataset['mean'].sel(time=slice(period_variance[0], period_variance[1]))
            dataset_min_filtered = dataset['min'].sel(time=slice(period_variance[0], period_variance[1]))['tasmean'].values
            dataset_max_filtered = dataset['max'].sel(time=slice(period_variance[0], period_variance[1]))['tasmean'].values
            tasmean_values = dataset_filtered['tasmean'].values
            years = dataset_filtered['time'].dt.year

            # Graficar los valores de 'tasmean' contra los años
            ax.plot(years, tasmean_values - tasmean_values, label=key)
            ax.fill_between(years, dataset_min_filtered, dataset_max_filtered, alpha=0.3)

            for key_number, numbered_set in dataset.items():
                if key_number not in ['mean', 'min', 'max']:
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

            # Incrementar el índice para el siguiente subplot
            plot_idx += 1

        # Ajustar el espaciado entre los subplots
        plt.tight_layout()

        # Guardar la figura para este periodo
        figName = f'histogramVariance_Mean_{period_variance[0]}-{period_variance[1]}_Datasets={len(predictands)}'
        plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
        plt.close()


    # Iterar sobre los periodos
    plt.figure(figsize=(12, 8))
    current_periods = [future_3, future_4, (future_1[0], future_3[1])]
    fig_size = len(predictands)//5
    for period_variance in current_periods:
        # Crear una figura con 6 subplots (3 filas, 2 columnas) por cada periodo
        fig, axes = plt.subplots(2+fig_size, 2, figsize=(14, 12+(6*fig_size)))  # 3 filas, 2 columnas

        # Aplanar la matriz de ejes (axes) para iterar más fácilmente
        axes = axes.flatten()

        # Iterador para asignar cada gráfico a un subplot
        plot_idx = 0

        for key, dataset in hist_gcm_99.items():
            ax = axes[plot_idx]  # Selecciona el subplot actual

            # Extraer los valores de 'tasmean'
            dataset_filtered = dataset['mean'].sel(time=slice(period_variance[0], period_variance[1]))
            dataset_min_filtered = dataset['min'].sel(time=slice(period_variance[0], period_variance[1]))['tasmean'].values
            dataset_max_filtered = dataset['max'].sel(time=slice(period_variance[0], period_variance[1]))['tasmean'].values
            tasmean_values = dataset_filtered['tasmean'].values
            years = dataset_filtered['time'].dt.year

            # Graficar los valores de 'tasmean' contra los años
            ax.plot(years, tasmean_values - tasmean_values, label=key)
            ax.fill_between(years, dataset_min_filtered, dataset_max_filtered, alpha=0.3)

            for key_number, numbered_set in dataset.items():
                if key_number not in ['mean', 'min', 'max']:
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

            # Incrementar el índice para el siguiente subplot
            plot_idx += 1

        # Ajustar el espaciado entre los subplots
        plt.tight_layout()

        # Guardar la figura para este periodo
        figName = f'histogramVariance_99Quantile_{period_variance[0]}-{period_variance[1]}_Datasets={len(predictands)}'
        plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
        plt.close()

    ####################################################################

    # Crear una figura y un conjunto de ejes
    figName = f'histogramFull_TenTrains'
    plt.figure(figsize=(12, 8))

    # Iterar sobre cada dataset en el diccionario
    for key, dataset in hist_gcm_mean.items():
        # Extraer los valores de 'tasmean'
        tasmean_values = dataset['mean']['tasmean'].values
        # Extraer los años de la coordenada 'time'
        years = dataset['mean']['time'].dt.year

        # Graficar los valores de 'tasmean' contra los años
        plt.plot(years, tasmean_values, label=key)
        plt.fill_between(years, dataset['min']['tasmean'].values, dataset['max']['tasmean'].values, alpha=0.3)

    # Añadir una línea punteada vertical en el año 2021
    plt.axvline(x=2021, color='r', linestyle='--', linewidth=1)

    # Configurar etiquetas y título
    plt.xlabel('Year')
    plt.ylabel('Tasmean')
    plt.legend(loc='best')  # Añadir la leyenda

    # Mostrar el gráfico
    plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
    plt.close()
# ########################################################
# # Histogramas individuales
# plt.figure(figsize=(12, 8))
# # Iterar sobre cada dataset en el diccionario
# for key, dataset in hist_gcm_mean.items():
#     figName = f'histogramFull_TenTrains_{key}'
#     # Extraer los valores de 'tasmean'
#     dataset_filtered = dataset['mean'].sel(time=slice('2084', '2100'))
#     dataset_min_filtered = dataset['min'].sel(time=slice('2084', '2100'))['tasmean'].values
#     dataset_max_filtered = dataset['max'].sel(time=slice('2084', '2100'))['tasmean'].values
#     tasmean_values = dataset_filtered['tasmean'].values
#     # Extraer los años de la coordenada 'time'
#     years = dataset_filtered['time'].dt.year

#     # Graficar los valores de 'tasmean' contra los años
#     plt.plot(years, tasmean_values, label=key)
#     plt.fill_between(years, dataset_min_filtered, dataset_max_filtered, alpha=0.3)

#     for key_number, numbered_set in dataset.items():
#         if key_number not in ['mean', 'min', 'max']:
#             numbered_set_filtered = numbered_set.sel(time=slice('2084', '2100'))
#             years_number = numbered_set_filtered['time'].dt.year
#             tasmean_values = numbered_set_filtered['tasmean'].values
#             plt.plot(years_number, tasmean_values, linestyle='dashed', color='red', linewidth=0.5)

#     # Configurar etiquetas y título
#     plt.xlabel('Year')
#     plt.ylabel('Tasmean')
#     plt.legend(loc='best')  # Añadir la leyenda

#     # Mostrar el gráfico
#     plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
#     plt.close()



# ####################################################################
# # GRAFICO DE VARIACION DE REALIZACION (PANELES INDIVIDUALES)
# current_periods = [future_3, future_4, (future_1[0], future_3[1])]
# for period_variance in current_periods:
#     for key, dataset in hist_gcm_mean.items():
#         plt.figure(figsize=(12, 8))
#         figName = f'histogramVariance_{key}_{period_variance[0]}-{period_variance[1]}'
#         # Extraer los valores de 'tasmean'
#         dataset_filtered = dataset['mean'].sel(time=slice(period_variance[0], period_variance[1]))
#         dataset_min_filtered = dataset['min'].sel(time=slice(period_variance[0], period_variance[1]))['tasmean'].values
#         dataset_max_filtered = dataset['max'].sel(time=slice(period_variance[0], period_variance[1]))['tasmean'].values
#         tasmean_values = dataset_filtered['tasmean'].values
#         # Extraer los años de la coordenada 'time'
#         years = dataset_filtered['time'].dt.year

#         # Graficar los valores de 'tasmean' contra los años
#         plt.plot(years, tasmean_values-tasmean_values, label=key)
#         plt.fill_between(years, dataset_min_filtered, dataset_max_filtered, alpha=0.3)

#         for key_number, numbered_set in dataset.items():
#             if key_number not in ['mean', 'min', 'max']:
#                 numbered_set_filtered = numbered_set.sel(time=slice(period_variance[0], period_variance[1]))
#                 years_number = numbered_set_filtered['time'].dt.year
#                 tasmean_values_realization = numbered_set_filtered['tasmean'].values - tasmean_values
#                 plt.plot(years_number, tasmean_values_realization, linestyle='dashed', color='red', linewidth=0.5)

#         plt.ylim(-1, 1)
#         # Configurar etiquetas y título
#         plt.xlabel('Year')
#         plt.ylabel('Tasmean')
#         plt.legend(loc='best')  # Añadir la leyenda

#         # Mostrar el gráfico
#         plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
#         plt.close()

####################################################################



# #Different points
# chelsaMean = yPredMetrics['ssp585']['CHELSA']['mean']
# current_dif = 0
# max_dif = 0
# max_key = ''

# for key, metric in yPredMetrics['ssp585'].items():
#     dif = chelsaMean['tasmean'] - metric['mean']['tasmean']
#     current_dif = np.abs(dif).max()
#     if current_dif > max_dif:
#         max_dif = current_dif
#         max_key = key
#         last_dif = dif

# max_location = last_dif.where(np.abs(last_dif) == max_dif, drop=True)

# lat_max = max_location['lat'].values.item()
# lon_max = max_location['lon'].values.item()

# # Encuentra los índices en lat y lon más cercanos a lat_max y lon_max
# lat_idx = chelsaMean['lat'].sel(lat=lat_max, method='nearest').lat
# lon_idx = chelsaMean['lon'].sel(lon=lon_max, method='nearest').lon

# print(f"Predictand with bigger difference: {max_key}")

# for key, metric in yPredMetrics['ssp585'].items():
#     mean = metric['mean']['tasmean'].sel(lat=lat_idx, lon=lon_idx).item()
#     sigma = metric['std']['tasmean'].sel(lat=lat_idx, lon=lon_idx).item()
#     interval = (mean-sigma, mean+sigma)
#     print(f"Predictand: {key} - Mean: {mean} - Range: {interval}")


# utils.graphSimilarityGrid(yPredMetrics, FIGS_PATH, scenario, sigma_number=1, grid_selection=(lat_idx, lon_idx))


###########################################################

