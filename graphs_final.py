import xarray as xr
import torch
from lib import utils, models, data, settings
import sys, time
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm

DATA_PATH_PREDICTORS = '/lustre/gmeteo/PTICLIMA/DATA/PROJECTIONS/CMIP6_PNACC/CMIP6_models/'
DATA_PATH_PREDICTANDS_READ = '/lustre/gmeteo/PTICLIMA/DATA/AUX/GRID_INTERCOMP/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
DATA_PATH_SHAPE = '/lustre/gmeteo/WORK/reyess/shapes/'
FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/paper-december/'
MODELS_PATH = '/oceano/gmeteo/users/reyess/tfm/official-code/models'
DATA_PREDICTORS_TRANSFORMED = '/lustre/gmeteo/WORK/reyess/data/NorthAtlanticRegion_1.5degree/'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/GCM/AEMET/'
PREDS_PATH_TEST = '/lustre/gmeteo/WORK/reyess/preds/'

# INPUT DATA
FIGS = str(sys.argv[1])
ENSEMBLE_QUANTITY = 50
GCM_NAME = 'EC-Earth3-Veg'
MAIN_SCENARIO = 'ssp585'
SHAPE_NAME = ['Iberia', 'Pirineos', 'Tinto', 'Duero']

# GENERAL VARIABLES
predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'CHELSA']#, 'pti-grid',]
metrics_1 = ['mean', '99quantile', '1quantile', 'std']
metrics_2 = ['Mean', '99Percentile', '1Percentile']
seasons = {'spring': 'MAM', 'summer': 'JJA', 'autumn': 'SON', 'winter': 'DJF'}

shape_file_path = 'river-basins_shapefile/river_basins.shp'
shape_file_path_major = 'major_basins_of_the_world_0_0_0/Major_Basins_of_the_World.shp'
hist_baseline = ('1995-01-01', '2014-12-31') #95-14
yearsTrain = ('1980-01-01', '2003-12-31')
yearsTest = ('2004-01-01', '2015-12-31')
future_1 = ('2021-01-01', '2040-12-31')
future_2 = ('2041-01-01', '2060-12-31')
future_3 = ('2081-01-01', '2100-12-31') 
future_4 = ('2061-01-01', '2080-12-31')


### # FIG1 # ####
if '1' in FIGS:
# DATOS OBSERVACION
    obs = {}
    whole_obs = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}
    whole_obs_metrics = {'annual': {}, 'spring': {}, 'summer': {}, 'autumn': {}, 'winter': {}}

    for predictand_name in predictands:

        modelName = f'DeepESD_tas_{predictand_name}' 
        print(predictand_name)
        obs[predictand_name] = utils.getPredictand(DATA_PATH_PREDICTANDS_SAVE, predictand_name, 'tasmean')
        obs[predictand_name] = obs[predictand_name].sel(time=slice(*(yearsTrain[0], yearsTest[1])))
        obs[predictand_name] = utils.maskData(
                    path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                    var='tasmean',
                    to_slice=(yearsTrain[0], yearsTest[1]),
                    objective = obs[predictand_name],
                    secondGrid = obs[predictand_name])
        whole_obs['annual'][predictand_name] = obs[predictand_name]
        whole_obs_metrics['annual'][predictand_name] = utils.getMetricsTemp(whole_obs['annual'][predictand_name], short = True)

        for season_name, months in seasons.items():
            whole_obs[season_name][predictand_name] = whole_obs['annual'][predictand_name].isel(time = (whole_obs['annual'][predictand_name].time.dt.season == months))
            whole_obs_metrics[season_name][predictand_name] = utils.getMetricsTemp(whole_obs[season_name][predictand_name], short = True)


    fig_num = 1
    for period, data_metrics in whole_obs_metrics.items():
        utils.metricsGraph(data_metrics, figs_path=FIGS_PATH, vmin=[0, 0, -5, 0, 1], vmax=[35, 40, 15, 15, 31], pred_type='observation_whole', fig_num = fig_num, period = period)#, extension='png')
        fig_num += Decimal('0.1')

            
    del obs, whole_obs, whole_obs_metrics

### # FIG2 # ####
if '2' in FIGS:
    # CLIMATOLOGY - ccsignal
    for metric in metrics_2:
        vminMetric = {'Mean': 10, '1Percentile': 0, '99Percentile': 20}
        vmaxMetric = {'Mean': 30, '1Percentile': 20, '99Percentile': 40}


        figName = f'fig2_Climatology_{ENSEMBLE_QUANTITY}_{metric}_part1'
        # Crear la figura y los ejes
        fig, axes = plt.subplots(3, 5, figsize=(20, 9), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

        continuousCMAP = plt.get_cmap('hot_r')
        discreteCMAP = ListedColormap(continuousCMAP(np.linspace(0, 1, 10)))
        discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, 11)[1:]))


        predictands_total_mean = []
        predictands_total_inter = []
        for i, predictand_name in enumerate(predictands):

            # Future Data
            predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
            predictand_data = {'mean': None, 'interq': None}
            mean_list = []

            grided_mean_list = []
            for predictand_number in predictand_numbered:
                modelName = f'DeepESD_tas_{predictand_number}' 
                loaded_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{future_3[0]}-{future_3[1]}.nc')
                if metric == '99Percentile':
                    loaded_data = loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')
                elif metric == '1Percentile':
                    loaded_data = loaded_data.resample(time = 'YE').quantile(0.1, dim = 'time')
                mean_time = loaded_data.mean(dim='time')
                mean_list.append(mean_time)

            predictand_data_ensemble = xr.concat(mean_list, dim='member')
            predictand_data_75 = predictand_data_ensemble.reduce(np.percentile, q=75, dim='member')
            predictand_data_25 = predictand_data_ensemble.reduce(np.percentile, q=25, dim='member')
            #predictand_data_75 = predictand_data_ensemble.quantile(0.75, dim='member')
            #predictand_data_25 = predictand_data_ensemble.quantile(0.25, dim='member')
            predictand_data['mean'] = predictand_data_ensemble.mean('member')
            #predictand_data['sd'] = predictand_data_ensemble.std('member')
            predictand_data['interq'] = predictand_data_75 - predictand_data_25
            predictands_total_mean.append(predictand_data['mean'])
            predictands_total_inter.append(predictand_data['interq'])
            #predictands_total_median.append(predictand_data_ensemble.median('member'))

            # INTERQUARTIL A MANO
            #print(np.sort(predictand_data_ensemble.sel(lat=40, lon=-2, method='nearest')['tasmean'].values)[6] - np.sort(predictand_data_ensemble.sel(lat=40, lon=-2, method='nearest')['tasmean'].values)[2])
            #print(np.percentile(np.sort(predictand_data_ensemble.sel(lat=40, lon=-2, method='nearest')['tasmean'].values), 75)-np.percentile(np.sort(predictand_data_ensemble.sel(lat=40, lon=-2, method='nearest')['tasmean'].values), 25))
            for j, (metric, metric_data) in enumerate(predictand_data.items()):
                if metric == 'mean':
                    vmin = vminMetric[metric]
                    vmax = vmaxMetric[metric]
                elif metric == 'sd':
                    vmin = 0
                    vmax = 0.8
                elif metric == 'interq':
                    vmin = 0
                    vmax = 1
                ax = axes[j, i]
                if j == 0:
                    ax.set_title(f'{predictand_name.capitalize()}', fontsize=16)
                if i == 0:
                    ax.text(-0.07, 0.55, metric.capitalize(), va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize=16)

                ax.coastlines(resolution='10m')
                

                dataToPlot = metric_data['tasmean']
                im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                                    dataToPlot,
                                    transform=ccrs.PlateCarree(),
                                    cmap=discreteCMAPnoWhite,
                                    vmin=vmin, vmax=vmax)
                                    #norm=BoundaryNorm(bounds, cmap.N))

                if i == 0:
                    cax = fig.add_axes([0.125, 0.53, 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
                    cbar = plt.colorbar(im, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
                    ticks = np.linspace(vmin, vmax, 6)
                    cbar.set_ticks(ticks)
                    cbar.ax.tick_params(labelsize=16)
                    tick_labels = [tick.get_text() for tick in cbar.ax.get_xticklabels()]
                    tick_labels[-1] += '+'
                    cbar.ax.set_xticklabels(tick_labels) 

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.close()


        figName = f'fig2_Climatology_{ENSEMBLE_QUANTITY}_{metric}_part2'
        # Crear la figura y los ejes
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

        data_to_plot = {'mean': None, 'sd mean': None, 'mean inter': None, 'sd inter': None}
        mean_combined = xr.concat(predictands_total_mean, dim='member')
        inter_combined = xr.concat(predictands_total_inter, dim='member')


        data_to_plot['mean'] = mean_combined.mean(dim='member')
        data_to_plot['sd mean'] = mean_combined.std(dim='member')
        data_to_plot['mean inter'] = inter_combined.mean(dim='member')
        data_to_plot['sd inter'] = inter_combined.std(dim='member')


        vmin_sd = 0
        vmax_sd = 0.8

        ax1 = axes[0, 0]
        ax2 = axes[1, 0]
        ax3 = axes[0, 1]
        ax4 = axes[1, 1]


        ax1.set_title(f'Mean', fontsize=16)
        ax3.set_title(f'StandarDeviation', fontsize=16)

        ax1.coastlines(resolution='10m')
        ax2.coastlines(resolution='10m')
        ax3.coastlines(resolution='10m')
        ax4.coastlines(resolution='10m')

        im1 = ax1.pcolormesh(data_to_plot['mean']['tasmean'].coords['lon'].values, data_to_plot['mean']['tasmean'].coords['lat'].values,
                            data_to_plot['mean']['tasmean'],
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAPnoWhite,
                            vmin=vminMetric[metric], vmax=vmaxMetric[metric])

        cax = fig.add_axes([0.125, 0.53, 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
        cbar = plt.colorbar(im1, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
        cbar.set_ticks(np.linspace(vminMetric[metric], vmaxMetric[metric], 6))
        cbar.ax.tick_params(labelsize=16)

        im2 = ax2.pcolormesh(data_to_plot['mean inter']['tasmean'].coords['lon'].values, data_to_plot['mean inter']['tasmean'].coords['lat'].values,
                            data_to_plot['mean inter']['tasmean'],
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAPnoWhite,
                            vmin=vminMetric[metric], vmax=vmaxMetric[metric])

        # Desviaciones estandar
        im3 = ax3.pcolormesh(data_to_plot['sd mean']['tasmean'].coords['lon'].values, data_to_plot['sd mean']['tasmean'].coords['lat'].values,
                            data_to_plot['sd mean']['tasmean'],
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAPnoWhite,
                            vmin=vmin_sd, vmax=vmax_sd)

        cax = fig.add_axes([0.125, 0.115, 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
        cbar = plt.colorbar(im3, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
        cbar.set_ticks(np.linspace(vmin_sd, vmax_sd, 6))
        cbar.ax.tick_params(labelsize=16)

        im4 = ax4.pcolormesh(data_to_plot['sd inter']['tasmean'].coords['lon'].values, data_to_plot['sd inter']['tasmean'].coords['lat'].values,
                            data_to_plot['sd inter']['tasmean'],
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAPnoWhite,
                            vmin=vmin_sd, vmax=vmax_sd)


        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.close()

    del predictand_data, predictand_data_ensemble, mean_combined, inter_combined, predictands_total_inter, predictands_total_mean, mean_list


    # climatology - CCSIGNAL
    for metric in metrics_2:
        vminMetric = {'Mean': 10, '1Percentile': 0, '99Percentile': 20}
        vmaxMetric = {'Mean': 30, '1Percentile': 20, '99Percentile': 40}


        figName = f'fig2_CCSignal_{ENSEMBLE_QUANTITY}_{metric}_part1'
        # Crear la figura y los ejes
        fig, axes = plt.subplots(3, 5, figsize=(20, 9), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

        continuousCMAP = plt.get_cmap('hot_r')
        discreteCMAP = ListedColormap(continuousCMAP(np.linspace(0, 1, 10)))
        discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, 11)[1:]))

        


        predictands_total_mean = []
        predictands_total_inter = []
        for i, predictand_name in enumerate(predictands):

            # Historical Data
            obs_predictand = utils.getPredictand(f'{DATA_PATH_PREDICTANDS_SAVE}', predictand_name, 'tasmean')
            obs_temp = obs_predictand.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
            obs_predictand = utils.maskData(
                        path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                        var='tasmean',
                        to_slice=(yearsTrain[0], yearsTest[1]),
                        objective = obs_predictand.sel(time=slice(*(hist_baseline[0], hist_baseline[1]))),
                        secondGrid = obs_temp)
            if metric == '99Percentile':
                obs_predictand = obs_predictand.resample(time = 'YE').quantile(0.99, dim = 'time')
            elif metric == '1Percentile':
                obs_predictand = obs_predictand.resample(time = 'YE').quantile(0.1, dim = 'time')
            obs_predictand_mean = obs_predictand.mean(dim='time')

            # Future Data
            predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
            predictand_data = {'mean': None, 'interq': None}
            mean_list = []

            grided_mean_list = []
            for predictand_number in predictand_numbered:
                modelName = f'DeepESD_tas_{predictand_number}' 
                loaded_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{future_3[0]}-{future_3[1]}.nc')
                if metric == '99Percentile':
                    loaded_data = loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')
                elif metric == '1Percentile':
                    loaded_data = loaded_data.resample(time = 'YE').quantile(0.1, dim = 'time')
                mean_time = loaded_data.mean(dim='time')
                mean_list.append(mean_time)

            predictand_data_ensemble = xr.concat(mean_list, dim='member') - obs_predictand_mean
            predictand_data_75 = predictand_data_ensemble.reduce(np.percentile, q=75, dim='member')
            predictand_data_25 = predictand_data_ensemble.reduce(np.percentile, q=25, dim='member')
            predictand_data['mean'] = predictand_data_ensemble.mean('member')
            predictand_data['interq'] = predictand_data_75 - predictand_data_25
            predictands_total_mean.append(predictand_data['mean'])
            predictands_total_inter.append(predictand_data['interq'])

            # INTERQUARTIL A MANO
            for j, (metric, metric_data) in enumerate(predictand_data.items()):
                if metric == 'mean':
                    vmin = vminMetric[metric]
                    vmax = vmaxMetric[metric]
                elif metric == 'sd':
                    vmin = 0
                    vmax = 0.8
                elif metric == 'interq':
                    vmin = 0
                    vmax = 1
                ax = axes[j, i]
                if j == 0:
                    ax.set_title(f'{predictand_name.capitalize()}', fontsize=16)
                if i == 0:
                    ax.text(-0.07, 0.55, metric.capitalize(), va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize=16)

                ax.coastlines(resolution='10m')
                

                dataToPlot = metric_data['tasmean']
                im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                                    dataToPlot,
                                    transform=ccrs.PlateCarree(),
                                    cmap=discreteCMAPnoWhite,
                                    vmin=vmin, vmax=vmax)
                                    #norm=BoundaryNorm(bounds, cmap.N))

                if i == 0:
                    cax = fig.add_axes([0.125, 0.53, 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
                    cbar = plt.colorbar(im, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
                    ticks = np.linspace(vmin, vmax, 6)
                    cbar.set_ticks(ticks)
                    cbar.ax.tick_params(labelsize=16)
                    tick_labels = [tick.get_text() for tick in cbar.ax.get_xticklabels()]
                    tick_labels[-1] += '+'
                    cbar.ax.set_xticklabels(tick_labels) 

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.close()


        figName = f'fig2_CCSignal_{ENSEMBLE_QUANTITY}_{metric}_part2'
        # Crear la figura y los ejes
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

        data_to_plot = {'mean': None, 'sd mean': None, 'mean inter': None, 'sd inter': None}
        mean_combined = xr.concat(predictands_total_mean, dim='member')
        inter_combined = xr.concat(predictands_total_inter, dim='member')


        data_to_plot['mean'] = mean_combined.mean(dim='member')
        data_to_plot['sd mean'] = mean_combined.std(dim='member')
        data_to_plot['mean inter'] = inter_combined.mean(dim='member')
        data_to_plot['sd inter'] = inter_combined.std(dim='member')


        vmin_sd = 0
        vmax_sd = 0.8

        ax1 = axes[0, 0]
        ax2 = axes[1, 0]
        ax3 = axes[0, 1]
        ax4 = axes[1, 1]


        ax1.set_title(f'Mean', fontsize=16)
        ax3.set_title(f'StandarDeviation', fontsize=16)

        ax1.coastlines(resolution='10m')
        ax2.coastlines(resolution='10m')
        ax3.coastlines(resolution='10m')
        ax4.coastlines(resolution='10m')

        im1 = ax1.pcolormesh(data_to_plot['mean']['tasmean'].coords['lon'].values, data_to_plot['mean']['tasmean'].coords['lat'].values,
                            data_to_plot['mean']['tasmean'],
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAPnoWhite,
                            vmin=vminMetric[metric], vmax=vmaxMetric[metric])

        cax = fig.add_axes([0.125, 0.53, 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
        cbar = plt.colorbar(im1, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
        cbar.set_ticks(np.linspace(vminMetric[metric], vmaxMetric[metric], 6))
        cbar.ax.tick_params(labelsize=16)

        im2 = ax2.pcolormesh(data_to_plot['mean inter']['tasmean'].coords['lon'].values, data_to_plot['mean inter']['tasmean'].coords['lat'].values,
                            data_to_plot['mean inter']['tasmean'],
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAPnoWhite,
                            vmin=vminMetric[metric], vmax=vmaxMetric[metric])

        # Desviaciones estandar
        im3 = ax3.pcolormesh(data_to_plot['sd mean']['tasmean'].coords['lon'].values, data_to_plot['sd mean']['tasmean'].coords['lat'].values,
                            data_to_plot['sd mean']['tasmean'],
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAPnoWhite,
                            vmin=vmin_sd, vmax=vmax_sd)

        cax = fig.add_axes([0.125, 0.115, 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
        cbar = plt.colorbar(im3, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
        cbar.set_ticks(np.linspace(vmin_sd, vmax_sd, 6))
        cbar.ax.tick_params(labelsize=16)

        im4 = ax4.pcolormesh(data_to_plot['sd inter']['tasmean'].coords['lon'].values, data_to_plot['sd inter']['tasmean'].coords['lat'].values,
                            data_to_plot['sd inter']['tasmean'],
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAPnoWhite,
                            vmin=vmin_sd, vmax=vmax_sd)


        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.close()

    del predictand_data, predictand_data_ensemble, mean_combined, inter_combined, predictands_total_inter, predictands_total_mean




### # FIG3 # ####
if '3' in FIGS:
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
    reference_grid = xr.open_dataset(f'{PREDS_PATH}/predGCM_DeepESD_tas_AEMET_0.25deg_1_{GCM_NAME}_{MAIN_SCENARIO}_{future_1[0]}-{future_1[1]}.nc')
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
    periods = [future_2, future_4, future_3]
    xmin = (2, 1)
    xmax = (11, 12)
    # DE 1 Limite
    for shape in SHAPE_NAME:
        # Etiquetas
        colors = ['lightgreen', 'lightblue', 'lightcoral']
        names = ['Short', 'Medium', 'Long']
        legend_handles = []

        figName = f'boxPlot_ccsignals_Ensemble{ENSEMBLE_QUANTITY}_{shape}'
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
                            to_slice=(hist_baseline[0], hist_baseline[1]),
                            objective = obs2.sel(time=slice(*(hist_baseline[0], hist_baseline[1]))),
                            secondGrid = obs_temp)
                obs2 = obs2.sel(
                        lat=references_grid[shape].lat,
                        lon=references_grid[shape].lon,
                    ) if shape != 'Iberia' else obs2
                obs2_99 = obs2.resample(time = 'YE').quantile(0.99, dim = 'time')

                obs2_mean = obs2.mean(dim=['time', 'lat', 'lon'])
                obs2_99_mean = obs2_99.mean(dim=['time', 'lat', 'lon'])


                predictand_data = []
                ccsignal_predictand = []
                ccsignal_predictand_99 = []
                predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]

                for predictand_number in predictand_numbered:
                    modelName = f'DeepESD_tas_{predictand_number}' 
                    loaded_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{period[0]}-{period[1]}.nc')
                    grided_data = loaded_data.sel(
                        lat=references_grid[shape].lat,
                        lon=references_grid[shape].lon,
                    ) if shape != 'Iberia' else loaded_data

                    grided_data_99 = grided_data.resample(time = 'YE').quantile(0.99, dim = 'time')
                    grided_mean = grided_data.mean(dim=['time', 'lat', 'lon']) 
                    grided_mean_99 = grided_data_99.mean(dim=['time', 'lat', 'lon'])
                    ccsignal_predictand.append(grided_mean- obs2_mean)
                    ccsignal_predictand_99.append(grided_mean_99 - obs2_99_mean)


                ccsignal_array = np.array([ds['tasmean'].values for ds in ccsignal_predictand])
                data_to_plot.append(ccsignal_array)

            ax = ax1.twiny() if i > 0 else ax1  # Crear ejes adicionales solo para Medium y Long
            color = colors[i]
            bplot = ax.boxplot(data_to_plot, positions=np.arange(len(predictands)) * 2.0 + offsets[i], widths=0.25, 
                            patch_artist=True, boxprops=dict(facecolor=color), vert=False, whis=[5, 95])
            #ax.set_xticks([]) if i > 0 else None  # Eliminar ticks en el eje X superior para ax2 y ax3
            ax.set_xlim(xmin[0], xmax[0])
            #AÑADIR DE ALGUNA FORMA LIMITES FUERA DE TICKS Y CONSIDERAR ARRIBA MEAN Y 99, 1 POR LINEA O SEA DOBLAR LO DE ARRIBA
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

        # Guardar el gráfico
        plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
        plt.show()
### # FIG4 # ####
if '4' in FIGS:

### # FIG5 # ####
if '5' in FIGS:


### # FIG OPTIONAL # ####
if '0' in FIGS: