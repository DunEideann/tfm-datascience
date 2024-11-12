import xarray as xr
from lib import utils, models, data, settings
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/similarity'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/GCM/AEMET/'
PREDS_PATH_TRAIN = '/lustre/gmeteo/WORK/reyess/preds/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
DATA_PATH_SHAPE = '/lustre/gmeteo/WORK/reyess/shapes/'
# SHAPE_NAME = int(sys.argv[1])
# PREDICTANDS_SIZE = int(sys.argv[2])
# ENSEMBLE_QUANTITY = int(sys.argv[3])
# METRIC = int(sys.argv[4])
PREDICTANDS_SIZE = 5
ENSEMBLE_QUANTITY = 10
SCENARIO = 3
#SHAPE_NAME = ['Iberia', 'Tagus', 'Ebro']#'Tagus2']
SHAPE_NAME = ['Iberia', 'Pirineos', 'Tinto', 'Duero']
#SHAPE_NAME = ['Tinto']
METRIC = '1Percentile' #'1Percentile'
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
periods = [future_2, future_4, future_3]

scenario = scenarios[SCENARIO]
scenario_train = 'train'
scenario_test = 'test'
scenario_whole = 'whole'
scenario_ccsignal = 'ccsignal'

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
shape_file_path = 'river-basins_shapefile/river_basins.shp'
shape_file_path_major = 'major_basins_of_the_world_0_0_0/Major_Basins_of_the_World.shp'

#***************FUNCIONES**********************
# Filtrar las coordenadas de new_dataset
def get_dataset(tagus):# Suponiendo que tagus es un objeto de tipo Polygon
    tagus_coords = np.array(tagus.exterior.coords)

    # Extraer las coordenadas de latitud y longitud
    lon = tagus_coords[:, 0]
    lat = tagus_coords[:, 1]

    # Crear el Dataset
    tagus_dataset = xr.Dataset(
        coords={
            'lat': lat,
            'lon': lon
        }
    )
    return tagus_dataset


def find_nearest(values, ref_array, var):
    nearest_values = []
    
    # Calcular los valores más cercanos manteniendo la forma original
    for value in values:
        nearest_value = ref_array[np.argmin(np.abs(ref_array - value.item()))]
        nearest_values.append(nearest_value)
    
    # Convertir a un numpy array con la misma forma que 'values'
    nearest_values = np.array(nearest_values).reshape(values.shape)
    
    # Eliminar duplicados mientras se mantiene el orden
    _, idx = np.unique(nearest_values, return_index=True)
    nearest_values_unique = nearest_values.flatten()[np.sort(idx)]
    
    filtered_nearest_values = xr.DataArray(
        nearest_values_unique,  # Acceso correcto usando diccionario
        coords={var: nearest_values_unique},        # Asignación de coordenadas
        dims=[var]                                 # Dimensión correcta
    )

    # Retornar el array con valores únicos
    return filtered_nearest_values


def filter_closest_coords(tagus_dataset, obs2):
# Obtener las coordenadas de latitud y longitud de obs2
    obs2_lat = obs2['lat'].values
    obs2_lon = obs2['lon'].values
    # Encuentra las coordenadas más cercanas en new_dataset
    lat_indices = []
    lon_indices = []
    for lat in obs2_lat:
        closest_lat_idx = np.argmin(np.abs(tagus_dataset['lat'].values - lat))
        lat_indices.append(closest_lat_idx)
    for lon in obs2_lon:
        closest_lon_idx = np.argmin(np.abs(tagus_dataset['lon'].values - lon))
        lon_indices.append(closest_lon_idx)
    # Obtener los índices únicos de latitud y longitud
    unique_lat_indices = np.unique(lat_indices)
    unique_lon_indices = np.unique(lon_indices)
    # Filtrar los datos en new_dataset
    filtered_lat = tagus_dataset['lat'].isel(lat=unique_lat_indices)
    filtered_lat = filtered_lat.sortby(filtered_lat)
    filtered_lon = tagus_dataset['lon'].isel(lon=unique_lon_indices)
    filtered_lon = filtered_lon.sortby(filtered_lon)
    tagus_new_lat = find_nearest(filtered_lat, obs2_lat, 'lat')
    tagus_new_lon = find_nearest(filtered_lon, obs2_lon, 'lon')
    num_times = 1  # Por ejemplo, el mismo número que 'time' en obs2
    tasmean_data = np.full((num_times, len(tagus_new_lat), len(tagus_new_lon)), np.nan)  # Llenar con NaN
    times = np.arange(np.datetime64('1970-01-01'), np.datetime64('1970-01-01') + np.timedelta64(num_times, 'D'))
    new_filtered_dataset = xr.Dataset(
        {
            'tasmean': (('time', 'lat', 'lon'), tasmean_data)
        },
        {'lat':tagus_new_lat, 'lon':tagus_new_lon, 'time': times})

    return new_filtered_dataset

def create_coords_dataset(lats, lons, reference):
    new_filtered_dataset = reference.sel(lon=slice(lons[0], lons[1]), lat=slice(lats[0], lats[1]))
    return new_filtered_dataset
#*********************************************************************+
# shape_file_major = gpd.read_file(f'{DATA_PATH_SHAPE}{shape_file_path_major}')
# with pd.option_context('display.max_rows', None):
#     print(shape_file_major)
# EMPIEZA CODIGO
references_grid = {shape: [] for shape in SHAPE_NAME}
shape_name_fig = ''
reference_grid = xr.open_dataset(f'{PREDS_PATH}/predGCM_DeepESD_tas_AEMET_0.25deg_1_{gcm_name}_{main_scenerio}_{future_1[0]}-{future_1[1]}.nc')
reference_grid = reference_grid.sel(time=slice(future_1[0],'2021-01-02'))
for shape in SHAPE_NAME:
    if shape == 'Tagus':
        shape_file = gpd.read_file(f'{DATA_PATH_SHAPE}{shape_file_path}')
        gdf = gpd.GeoDataFrame(shape_file)
        tagus = gdf[gdf['NAME'] == shape]['geometry'].values[0]
        tagus_dataset = get_dataset(tagus)
        reference_grid_modified = filter_closest_coords(tagus_dataset, reference_grid)
        references_grid[shape] = reference_grid_modified
    elif shape == 'Iberia':
        references_grid[shape] = None
    elif shape == 'Tagus2':
        shape_file = gpd.read_file(f'{DATA_PATH_SHAPE}{shape_file_path}')
        gdf = gpd.GeoDataFrame(shape_file)
        tagus = gdf[gdf['NAME'] == 'Tagus']['geometry'].values[0]
        tagus_dataset = get_dataset(tagus)
        reference_grid_modified = filter_closest_coords(tagus_dataset, reference_grid)
        references_grid[shape] = reference_grid_modified
    elif shape == 'Ebro':
        shape_file = gpd.read_file(f'{DATA_PATH_SHAPE}{shape_file_path_major}')
        gdf = gpd.GeoDataFrame(shape_file)
        tagus = gdf[gdf['NAME'] == shape]['geometry'].values[0]
        tagus_dataset = get_dataset(tagus)
        reference_grid_modified = filter_closest_coords(tagus_dataset, reference_grid)
        references_grid[shape] = reference_grid_modified
    elif shape == 'Pirineos':
        references_grid[shape] = create_coords_dataset(lons=(-0.37, 3.37), lats=(41.42, 42.80), reference=reference_grid)
    elif shape == 'Duero':
        references_grid[shape] = create_coords_dataset(lons=(-6.59, -4.75), lats=(40.85, 42.45), reference=reference_grid)
    elif shape == 'Tinto':
        references_grid[shape] = create_coords_dataset(lons=(-7.23, -5.20), lats=(36.00, 38.20), reference=reference_grid)

    shape_name_fig = f'{shape_name_fig}_{shape}'

# GRAFICOS BOXPLOT PARA SHORT, MEDIUM y LONG / minimos y maximos

figName = f'boxPlot_ccsignals_Ensemble{ENSEMBLE_QUANTITY}{shape_name_fig}_{METRIC}'
xmin = 0 if METRIC != '1Percentile' else 1
xmax = 8 if METRIC != '99Percentile' else 12

# # DE VARIAS
# # Etiquetas
# colors = ['lightgreen', 'lightblue', 'lightcoral']
# names = ['Short', 'Medium', 'Long']
# legend_handles = []

# # Crear la figura y los ejes
# fig, ax1 = plt.subplots(figsize=(10, 8))

# # Posiciones iniciales para cada conjunto de datos (Short, Medium, Long)
# offsets = [-0.3, 0, 0.3]  # Desplazamientos para cada grupo en el eje Y


# # Graficar cada set de datos (Short, Medium, Long) en el mismo gráfico
# for i, period in enumerate(periods):
#     print(period)
#     data_to_plot = {shape: [] for shape in SHAPE_NAME}
#     for predictand_name in predictands:
#         obs2_mean = {shape:None for shape in SHAPE_NAME}
#         obs2 = utils.getPredictand(f'{DATA_PATH_PREDICTANDS_SAVE}', predictand_name, 'tasmean')
#         obs_temp = obs2.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
#         obs2 = utils.maskData(
#                     path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
#                     var='tasmean',
#                     to_slice=(yearsTrain[0], yearsTest[1]),
#                     objective = obs2.sel(time=slice(*(past_timeline[0], past_timeline[1]))),
#                     secondGrid = obs_temp)
#         if METRIC == '99Percentile':
#             obs2 = obs2.resample(time = 'YE').quantile(0.99, dim = 'time')
#         elif METRIC == '1Percentile':
#             obs2 = obs2.resample(time = 'YE').quantile(0.1, dim = 'time')
#         for shape in SHAPE_NAME:
#             obs2_gridded = obs2.sel(
#                 lat=references_grid[shape].lat,
#                 lon=references_grid[shape].lon,
#             ) if shape != 'Iberia' else obs2
#             obs2_mean[shape] = obs2.mean(dim=['time', 'lat', 'lon'])


#         predictand_data = []
#         ccsignal_predictand = {shape: [] for shape in SHAPE_NAME}
#         predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]

#         for predictand_number in predictand_numbered:
#             modelName = f'DeepESD_tas_{predictand_number}' 
#             loaded_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{gcm_name}_{main_scenerio}_{period[0]}-{period[1]}.nc')
#             # grided_data = loaded_data.sel(
#             #     lat=reference_grid.lat,
#             #     lon=reference_grid.lon,
#             # ) if SHAPE_NAME != 'Iberia' else loaded_data
#             #predictand_data.append(grided_data.resample(time = 'YE').mean())
#             if METRIC == '99Percentile':
#                 loaded_data = loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')
#             elif METRIC == '1Percentile':
#                 loaded_data = loaded_data.resample(time = 'YE').quantile(0.1, dim = 'time')
#             for shape in SHAPE_NAME:
#                 grided_data = loaded_data.sel(
#                     lat=references_grid[shape].lat,
#                     lon=references_grid[shape].lon,
#                 ) if shape != 'Iberia' else loaded_data
                
#                 grided_mean = grided_data.mean(dim=['time', 'lat', 'lon']) 
#                 ccsignal_predictand[shape].append(grided_mean- obs2_mean[shape])
#         for shape in SHAPE_NAME:
#             ccsignal_array = np.array([ds['tasmean'].values for ds in ccsignal_predictand[shape]])
#             data_to_plot[shape].append(ccsignal_array)

#     print(data_to_plot)
#     color = colors[i]
#     for j, shape in enumerate(SHAPE_NAME):
#         ax = ax1.twiny() if (i>0) or (j>0) else ax1  # Crear ejes adicionales solo para Medium y Long
#         bplot = ax.boxplot(data_to_plot[shape], positions=np.arange(len(predictands)) * 2.0 + offsets[j], widths=0.25, 
#                         patch_artist=True, boxprops=dict(facecolor=color), vert=False, whis=[0, 100])
#     #ax.set_xticks([]) if i > 0 else None  # Eliminar ticks en el eje X superior para ax2 y ax3
#         ax.set_xlim(xmin, xmax)
#         ax.set_xticks([]) if (i>0) or (j>0) else None
    
#     # Asignar la etiqueta del eje X solo para el primer eje (ax1)
#     if i == 0:
#         ax.set_xlabel('CC Signal Tasmean')
#     # Crear un handle de la leyenda solo en la primera iteración para cada conjunto de datos
#     legend_handles.append(bplot["boxes"][0])

# # Etiquetas del eje Y solo en ax1
# ax1.set_yticks(np.arange(len(predictands)) * 2.0)
# ax1.set_yticklabels(predictands)

# # Agregar una leyenda para cada boxplot
# plt.legend(legend_handles, names, loc='lower right', prop={'size': 10}, frameon=False)
# # for i, name in enumerate(names):
# #     ax1.legend([bplot["boxes"][0]], [name], loc='upper right' if i == 2 else 'lower left', prop={'size': 10}, frameon=False)

# # Guardar el gráfico
# plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
# plt.show()



# DE 1 Limite
for shape in SHAPE_NAME:
    # Etiquetas
    colors = ['lightgreen', 'lightblue', 'lightcoral']
    names = ['Short', 'Medium', 'Long']
    legend_handles = []

    figName = f'boxPlot_ccsignals_Ensemble{ENSEMBLE_QUANTITY}_{shape}_{METRIC}'
    # Crear la figura y los ejes
    fig, ax1 = plt.subplots(figsize=(10, 8))

    # Posiciones iniciales para cada conjunto de datos (Short, Medium, Long)
    offsets = [-0.3, 0, 0.3]  # Desplazamientos para cada grupo en el eje Y


    # Graficar cada set de datos (Short, Medium, Long) en el mismo gráfico
    for i, period in enumerate(periods):
        print(period)
        data_to_plot = []
        for predictand_name in predictands:
            obs2 = utils.getPredictand(f'{DATA_PATH_PREDICTANDS_SAVE}', predictand_name, 'tasmean')
            obs_temp = obs2.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
            obs2 = utils.maskData(
                        path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                        var='tasmean',
                        to_slice=(yearsTrain[0], yearsTest[1]),
                        objective = obs2.sel(time=slice(*(past_timeline[0], past_timeline[1]))),
                        secondGrid = obs_temp)
            obs2 = obs2.sel(
                    lat=references_grid[shape].lat,
                    lon=references_grid[shape].lon,
                ) if shape != 'Iberia' else obs2
            if METRIC == '99Percentile':
                obs2 = obs2.resample(time = 'YE').quantile(0.99, dim = 'time')
            elif METRIC == '1Percentile':
                obs2 = obs2.resample(time = 'YE').quantile(0.1, dim = 'time')
            obs2_mean = obs2.mean(dim=['time', 'lat', 'lon'])


            predictand_data = []
            ccsignal_predictand = []
            predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]

            for predictand_number in predictand_numbered:
                modelName = f'DeepESD_tas_{predictand_number}' 
                loaded_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{gcm_name}_{main_scenerio}_{period[0]}-{period[1]}.nc')
                grided_data = loaded_data.sel(
                    lat=references_grid[shape].lat,
                    lon=references_grid[shape].lon,
                ) if shape != 'Iberia' else loaded_data
                #predictand_data.append(grided_data.resample(time = 'YE').mean())
                if METRIC == '99Percentile':
                    grided_data = grided_data.resample(time = 'YE').quantile(0.99, dim = 'time')
                elif METRIC == '1Percentile':
                    grided_data = grided_data.resample(time = 'YE').quantile(0.1, dim = 'time')
                grided_mean = grided_data.mean(dim=['time', 'lat', 'lon']) 
                ccsignal_predictand.append(grided_mean- obs2_mean)
            #predictand_mean = xr.concat(predictand_data, dim='member').mean('member')
            #ccsignal = predictand_mean.mean(dim='time') - obs2_mean.mean(dim='time')
            ccsignal_array = np.array([ds['tasmean'].values for ds in ccsignal_predictand])
            #flatten = ccsignal['tasmean'].values.flatten()
            #flatten = flatten[~np.isnan(flatten)]
            data_to_plot.append(ccsignal_array)

        print(data_to_plot)
        ax = ax1.twiny() if i > 0 else ax1  # Crear ejes adicionales solo para Medium y Long
        color = colors[i]
        bplot = ax.boxplot(data_to_plot, positions=np.arange(len(predictands)) * 2.0 + offsets[i], widths=0.25, 
                        patch_artist=True, boxprops=dict(facecolor=color), vert=False, whis=[5, 95])
        #ax.set_xticks([]) if i > 0 else None  # Eliminar ticks en el eje X superior para ax2 y ax3
        ax.set_xlim(xmin, xmax)
        ax.set_xticks([]) if i>0 else None
        
        # Asignar la etiqueta del eje X solo para el primer eje (ax1)
        if i == 0:
            ax.set_xlabel('CC Signal Tasmean')
        # Crear un handle de la leyenda solo en la primera iteración para cada conjunto de datos
        legend_handles.append(bplot["boxes"][0])

    # Etiquetas del eje Y solo en ax1
    ax1.set_yticks(np.arange(len(predictands)) * 2.0)
    ax1.set_yticklabels(predictands)

    # Agregar una leyenda para cada boxplot
    plt.legend(legend_handles, names, loc='lower right', prop={'size': 10}, frameon=False)
    # for i, name in enumerate(names):
    #     ax1.legend([bplot["boxes"][0]], [name], loc='upper right' if i == 2 else 'lower left', prop={'size': 10}, frameon=False)

    # Guardar el gráfico
    plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
    plt.show()