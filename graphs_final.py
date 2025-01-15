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
from matplotlib.patches import Patch

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
#FIGS = '5'
ENSEMBLE_QUANTITY = 50
GCM_NAME = 'EC-Earth3-Veg'
MAIN_SCENARIO = 'ssp585'
SHAPE_NAME = ['Iberia', 'Pirineos', 'Tinto', 'Duero']

# GENERAL VARIABLES
predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'CHELSA']#, 'pti-grid',]
metrics_1 = ['mean', '99quantile', '1quantile', 'std']
metrics_2 = ['Mean', '99Percentile', '1Percentile']
metrics_2 = ['Mean', '99Percentile']

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
    whole_obs = {'annual': {}}
    whole_obs_metrics = {'annual': {}}

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
        whole_obs['annual'][predictand_name] = obs[predictand_name]
        whole_obs_metrics['annual'][predictand_name] = utils.getMetricsTemp(whole_obs['annual'][predictand_name], short = True)

        # for season_name, months in seasons.items():
        #     whole_obs[season_name][predictand_name] = whole_obs['annual'][predictand_name].isel(time = (whole_obs['annual'][predictand_name].time.dt.season == months))
        #     whole_obs_metrics[season_name][predictand_name] = utils.getMetricsTemp(whole_obs[season_name][predictand_name], short = True)


    fig_num = 1
    print(f"WHOLE METRICS")
    print(whole_obs_metrics)
    for period, data_metrics in whole_obs_metrics.items():
        print(data_metrics)
        utils.metricsGraph(datasets_metrics=data_metrics, figs_path=FIGS_PATH, vmin=[5, 15, -8, 4.5], vmax=[25, 35, 12, 9.5], pred_type='observation_whole', fig_num = fig_num, period = period)#, extension='png')
        utils.metricsGraph(datasets_metrics=data_metrics, figs_path=FIGS_PATH, vmin=[5, 15, -8, 4.5], vmax=[25, 35, 12, 9.5], pred_type='observation_whole', fig_num = fig_num, period = period, extension='png')
        fig_num += Decimal('0.1')

            
    del obs, whole_obs, whole_obs_metrics
    print("Figura 1 completada!")

### # FIG2 # ####
if '2' in FIGS:
    # CLIMATOLOGY - ccsignal
    for metric in metrics_2:
        vminMetric = {'Mean': (10, 0, 0), '1Percentile': (0, 0, 0), '99Percentile': (20, 0, 0)}
        vmaxMetric = {'Mean': (30, 1, 0.8), '1Percentile': (20, 1, 0.8), '99Percentile': (40, 2, 1)}


        figName = f'fig2_Statistics_Climatology_{ENSEMBLE_QUANTITY}_{metric}_part1'
        # Crear la figura y los ejes
        fig, axes = plt.subplots(3, 5, figsize=(20, 9), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

        continuousCMAP = plt.get_cmap('hot_r')
        discreteCMAP = ListedColormap(continuousCMAP(np.linspace(0, 1, 10)))
        discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, 11)[1:]))


        predictands_total_mean = []
        #predictands_total_inter = []
        for i, predictand_name in enumerate(predictands):

            # Future Data
            predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
            predictand_data = {'mean': None, 'interq': None, 'sd': None}
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
            predictand_data['sd'] = predictand_data_ensemble.std('member')
            predictand_data['interq'] = predictand_data_75 - predictand_data_25
            predictands_total_mean.append(predictand_data['mean'])
            #predictands_total_inter.append(predictand_data['interq'])
            #predictands_total_median.append(predictand_data_ensemble.median('member'))

            for j, (metric_fig, metric_data) in enumerate(predictand_data.items()):
                if metric_fig == 'mean':
                    vmin = vminMetric[metric][0]
                    vmax = vmaxMetric[metric][0]
                elif metric_fig == 'sd':
                    vmin = vminMetric[metric][2]
                    vmax = vmaxMetric[metric][2]
                elif metric_fig == 'interq':
                    vmin = vminMetric[metric][1]
                    vmax = vmaxMetric[metric][1]
                ax = axes[j, i]
                if j == 0:
                    ax.set_title(f'{predictand_name.capitalize()}', fontsize=16)
                if i == 0:
                    ax.text(-0.07, 0.55, metric_fig.capitalize(), va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize=16)

                ax.coastlines(resolution='10m')
                

                dataToPlot = metric_data['tasmean']
                if metric_fig != 'mean':
                    dataToGraph = np.log(dataToPlot + 1)
                else:
                    dataToGraph = dataToPlot
                im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                                    dataToGraph,
                                    transform=ccrs.PlateCarree(),
                                    cmap=discreteCMAPnoWhite,
                                    vmin=vmin, vmax=vmax)
                                    #norm=BoundaryNorm(bounds, cmap.N))

                if i == 0:
                    cax = fig.add_axes([0.125, 0.65 - (j * 0.30), 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
                    cbar = plt.colorbar(im, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
                    ticks = np.linspace(vmin, vmax, 6)
                    cbar.set_ticks(ticks)
                    cbar.ax.tick_params(labelsize=16)
                    tick_labels = [tick.get_text() for tick in cbar.ax.get_xticklabels()]
                    tick_labels[-1] += '+'
                    cbar.ax.set_xticklabels(tick_labels) 

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()


        figName = f'fig2_Statistics_Climatology_{ENSEMBLE_QUANTITY}_{metric}_part2'
        # Crear la figura y los ejes
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

        data_to_plot = {'mean': None, 'mean-inter': None, 'mean-sd': None}
        mean_combined = xr.concat(predictands_total_mean, dim='member')
        #inter_combined = xr.concat(predictands_total_inter, dim='member')


        data_to_plot['mean'] = mean_combined.mean(dim='member')
        data_to_plot['mean-inter'] = mean_combined.quantile(0.75, dim='member') - mean_combined.quantile(0.25, dim='member')
        data_to_plot['mean-inter'] = np.log(data_to_plot['mean-inter'] + 1)
        data_to_plot['mean-sd'] = mean_combined.std(dim='member')
        data_to_plot['mean-sd'] = np.log(data_to_plot['mean-sd'] + 1)



        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]


        ax1.set_title(f'Mean', fontsize=16)
        ax2.set_title(f'InterQuartile', fontsize=16)
        ax3.set_title(f'StandarDeviation', fontsize=16)

        ax1.coastlines(resolution='10m')
        ax2.coastlines(resolution='10m')
        ax3.coastlines(resolution='10m')

        im1 = ax1.pcolormesh(data_to_plot['mean']['tasmean'].coords['lon'].values, data_to_plot['mean']['tasmean'].coords['lat'].values,
                            data_to_plot['mean']['tasmean'],
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAPnoWhite,
                            vmin=vminMetric[metric][0], vmax=vmaxMetric[metric][0])

        cax = fig.add_axes([0.28, 0.288, 0.02, 0.425]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
        cbar = plt.colorbar(im1, cax, pad=0.05, spacing='uniform', orientation='vertical')#, extend='both', extendfrac='auto', )
        cbar.set_ticks(np.linspace(vminMetric[metric][0], vmaxMetric[metric][0], 6))
        cbar.ax.tick_params(labelsize=8)

        im2 = ax2.pcolormesh(data_to_plot['mean-inter']['tasmean'].coords['lon'].values, data_to_plot['mean-inter']['tasmean'].coords['lat'].values,
                            data_to_plot['mean-inter']['tasmean'],
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAPnoWhite,
                            vmin=vminMetric[metric][1], vmax=vmaxMetric[metric][1])
        
        cax = fig.add_axes([0.5523, 0.288, 0.02, 0.425]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
        cbar = plt.colorbar(im2, cax, pad=0.05, spacing='uniform', orientation='vertical')#, extend='both', extendfrac='auto', )
        cbar.set_ticks(np.linspace(vminMetric[metric][1], vmaxMetric[metric][1], 6))
        cbar.ax.tick_params(labelsize=8)

        # Desviaciones estandar
        im3 = ax3.pcolormesh(data_to_plot['mean-sd']['tasmean'].coords['lon'].values, data_to_plot['mean-sd']['tasmean'].coords['lat'].values,
                            data_to_plot['mean-sd']['tasmean'],
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAPnoWhite,
                            vmin=vminMetric[metric][2], vmax=vmaxMetric[metric][2])

        cax = fig.add_axes([0.823, 0.288, 0.02, 0.425]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
        cbar = plt.colorbar(im3, cax, pad=0.05, spacing='uniform', orientation='vertical')#, extend='both', extendfrac='auto', )
        cbar.set_ticks(np.linspace(vminMetric[metric][2], vmaxMetric[metric][2], 6))
        cbar.ax.tick_params(labelsize=8)



        plt.subplots_adjust(left=0.05, right=0.82, top=0.95, bottom=0.05, wspace=0.2, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()

    del predictand_data, predictand_data_ensemble, mean_combined, predictands_total_mean, mean_list


    # climatology - CCSIGNAL
    for metric in metrics_2:
        vminMetric = {'Mean': (4, 0, 0), '1Percentile': (1, 0, 0), '99Percentile': (4, 0, 0)}
        vmaxMetric = {'Mean': (9, 0.75, 0.75), '1Percentile': (8, 0.5, 0.5), '99Percentile': (14, 1.25, 1.25)}


        figName = f'fig2_Statistics_CCSignal_{ENSEMBLE_QUANTITY}_{metric}_part1'
        # Crear la figura y los ejes
        fig, axes = plt.subplots(3, 5, figsize=(20, 9), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

        continuousCMAP = plt.get_cmap('hot_r')
        discreteCMAP = ListedColormap(continuousCMAP(np.linspace(0, 1, 10)))
        discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, 11)[1:]))

        predictands_total_mean = []
        #predictands_total_inter = []
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
            predictand_data['sd'] = predictand_data_ensemble.std('member')
            predictand_data['interq'] = predictand_data_75 - predictand_data_25
            predictands_total_mean.append(predictand_data['mean'])
            #predictands_total_inter.append(predictand_data['interq'])

            # INTERQUARTIL A MANO
            for j, (metric_fig, metric_data) in enumerate(predictand_data.items()):
                if metric_fig == 'mean':
                    vmin = vminMetric[metric][0]
                    vmax = vmaxMetric[metric][0]
                elif metric_fig == 'sd':
                    vmin = vminMetric[metric][2]
                    vmax = vmaxMetric[metric][2]
                elif metric_fig == 'interq':
                    vmin = vminMetric[metric][1]
                    vmax = vmaxMetric[metric][1]
                ax = axes[j, i]
                if j == 0:
                    ax.set_title(f'{predictand_name.capitalize()}', fontsize=16)
                if i == 0:
                    ax.text(-0.07, 0.55, metric_fig.capitalize(), va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize=16)

                ax.coastlines(resolution='10m')
                

                dataToPlot = metric_data['tasmean']
                if metric_fig != 'mean':
                    dataToGraph = np.log(dataToPlot + 1)
                else:
                    dataToGraph = dataToPlot
                print(f'{metric}/{metric_fig}-{predictand_name} max: {dataToGraph.max().item()}')
                print(f'{metric}/{metric_fig}-{predictand_name} min: {dataToGraph.min().item()}')

                im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                                    dataToGraph,
                                    transform=ccrs.PlateCarree(),
                                    cmap=discreteCMAPnoWhite,
                                    vmin=vmin, vmax=vmax)
                                    #norm=BoundaryNorm(bounds, cmap.N))

                if i == 0:
                    cax = fig.add_axes([0.125, 0.65 - (j * 0.30), 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
                    cbar = plt.colorbar(im, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
                    ticks = np.linspace(vmin, vmax, 6)
                    cbar.set_ticks(ticks)
                    cbar.ax.tick_params(labelsize=16)
                    tick_labels = [tick.get_text() for tick in cbar.ax.get_xticklabels()]
                    tick_labels[-1] += '+'
                    cbar.ax.set_xticklabels(tick_labels) 

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()


        figName = f'fig2_Statistics_CCSignal_{ENSEMBLE_QUANTITY}_{metric}_part2'
        # Crear la figura y los ejes
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

        data_to_plot = {'mean': None, 'mean-inter': None, 'mean-sd': None}
        mean_combined = xr.concat(predictands_total_mean, dim='member')
        #inter_combined = xr.concat(predictands_total_inter, dim='member')


        data_to_plot['mean'] = mean_combined.mean(dim='member')
        data_to_plot['mean-inter'] = mean_combined.quantile(0.75, dim='member') - mean_combined.quantile(0.25, dim='member')
        data_to_plot['mean-inter'] = np.log(data_to_plot['mean-inter'] + 1)
        data_to_plot['mean-sd'] = mean_combined.std(dim='member')
        data_to_plot['mean-sd'] = np.log(data_to_plot['mean-sd'] + 1)
        print(f'{metric}mean max: {data_to_plot['mean']['tasmean'].max().item()}')
        print(f'{metric}mean-inter max: {data_to_plot['mean-inter']['tasmean'].max().item()}')
        print(f'{metric}mean-sdmax: {data_to_plot['mean-sd']['tasmean'].max().item()}')
        print(f'{metric}mean min: {data_to_plot['mean']['tasmean'].min().item()}')
        print(f'{metric}mean-inter min: {data_to_plot['mean-inter']['tasmean'].min().item()}')
        print(f'{metric}mean-sd min: {data_to_plot['mean-sd']['tasmean'].min().item()}')

        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]


        ax1.set_title(f'Mean', fontsize=16)
        ax2.set_title(f'InterQuartile', fontsize=16)
        ax3.set_title(f'StandarDeviation', fontsize=16)

        ax1.coastlines(resolution='10m')
        ax2.coastlines(resolution='10m')
        ax3.coastlines(resolution='10m')

        im1 = ax1.pcolormesh(data_to_plot['mean']['tasmean'].coords['lon'].values, data_to_plot['mean']['tasmean'].coords['lat'].values,
                            data_to_plot['mean']['tasmean'],
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAPnoWhite,
                            vmin=vminMetric[metric][0], vmax=vmaxMetric[metric][0])

        cax = fig.add_axes([0.28, 0.288, 0.02, 0.425]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
        cbar = plt.colorbar(im1, cax, pad=0.05, spacing='uniform', orientation='vertical')#, extend='both', extendfrac='auto', )
        cbar.set_ticks(np.linspace(vminMetric[metric][0], vmaxMetric[metric][0], 6))
        cbar.ax.tick_params(labelsize=8)

        im2 = ax2.pcolormesh(data_to_plot['mean-inter']['tasmean'].coords['lon'].values, data_to_plot['mean-inter']['tasmean'].coords['lat'].values,
                            data_to_plot['mean-inter']['tasmean'],
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAPnoWhite,
                            vmin=vminMetric[metric][1], vmax=vmaxMetric[metric][1])
        
        cax = fig.add_axes([0.5523, 0.288, 0.02, 0.425]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
        cbar = plt.colorbar(im2, cax, pad=0.05, spacing='uniform', orientation='vertical')#, extend='both', extendfrac='auto', )
        cbar.set_ticks(np.linspace(vminMetric[metric][1], vmaxMetric[metric][1], 6))
        cbar.ax.tick_params(labelsize=8)

        # Desviaciones estandar
        im3 = ax3.pcolormesh(data_to_plot['mean-sd']['tasmean'].coords['lon'].values, data_to_plot['mean-sd']['tasmean'].coords['lat'].values,
                            data_to_plot['mean-sd']['tasmean'],
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAPnoWhite,
                            vmin=vminMetric[metric][2], vmax=vmaxMetric[metric][2])

        cax = fig.add_axes([0.823, 0.288, 0.02, 0.425]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
        cbar = plt.colorbar(im3, cax, pad=0.05, spacing='uniform', orientation='vertical')#, extend='both', extendfrac='auto', )
        cbar.set_ticks(np.linspace(vminMetric[metric][2], vmaxMetric[metric][2], 6))
        cbar.ax.tick_params(labelsize=8)



        plt.subplots_adjust(left=0.05, right=0.82, top=0.95, bottom=0.05, wspace=0.2, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()

    del predictand_data, predictand_data_ensemble, mean_combined, predictands_total_mean, mean_list

    print("Figura 2 completada!")



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

    # GRAFICOS BOXPLOT PARA SHORT, MEDIUM y LONG / CCSIGNAL
    periods = [future_2, future_4, future_3]
    xmin = (2, 1.5)
    xmax = (11, 11.5)
    # CC SIGNAL
    for shape in SHAPE_NAME:
        # Etiquetas
        colors = ['lightgreen', 'lightblue', 'bisque']
        names = ['Short', 'Medium', 'Long']
        legend_handles = []

        figName = f'fig3_boxPlot_ccsignals_Ensemble{ENSEMBLE_QUANTITY}_{shape}'
        # Crear la figura y los ejes
        fig, ax1 = plt.subplots(figsize=(10, 8))

        # Graficar cada set de datos (Short, Medium, Long) en el mismo gráfico
        for i, period in enumerate(periods):
            data_to_plot = []
            data_to_plot_99 = []
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
                ccsignal_array_99 = np.array([ds['tasmean'].values for ds in ccsignal_predictand_99])
                data_to_plot_99.append(ccsignal_array_99)

            ax = ax1.twiny() if i > 0 else ax1  # Crear ejes adicionales solo para Medium y Long
            color = colors[i]
            bplot = ax.boxplot(data_to_plot, positions= 5 + np.arange(len(predictands)), widths=0.35, 
                            patch_artist=True, boxprops=dict(facecolor=color), vert=False, whis=[5, 95],
                            whiskerprops=dict(color=color), flierprops=dict(color=color, markeredgecolor=color))
            ax.set_xlim(xmin[1], xmax[1]-2)
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.tick_top()
            #ax.set_xticks([]) if i!=0 else ax.set_xticks(np.linspace(xmin[0], xmax[0], 7))
            ax.set_xticks(np.linspace(xmin[0], xmax[0]-2, 8))
            #ax.legend('Mean', loc='center right', bbox_to_anchor=(1, 0))


            ax_99 = ax1.twiny() if i > 0 else ax1  # Crear ejes adicionales solo para Medium y Long
            bplot = ax_99.boxplot(data_to_plot_99, positions=np.arange(len(predictands))+0.1 , widths=0.35, 
                            patch_artist=True, boxprops=dict(facecolor=color), vert=False, whis=[5, 95],
                            whiskerprops=dict(color=color), flierprops=dict(color=color, markeredgecolor=color))
            ax_99.set_xlim(xmin[1], xmax[1])
            ax_99.xaxis.set_ticks_position('bottom')
            ax_99.set_xticks([]) if i>0 else ax_99.set_xticks(np.linspace(xmin[0], xmax[0], 10))
            # if i == 2:
            #     ax_99.xaxis.set_ticks_position('top')
            #     ax_99.set_xticks(np.linspace(xmin[0], xmax[0]-2, 10))
            #ax_99.legend('99 Percentile', loc='center right', bbox_to_anchor=(0.5, 0))
            
            
            # Asignar la etiqueta del eje X solo para el primer eje (ax1)
            if i == 0:
                ax.set_xlabel('CC Signal Tasmean')
            # Crear un handle de la leyenda solo en la primera iteración para cada conjunto de datos
            legend_handles.append(bplot["boxes"][0])

        # Etiquetas del eje Y solo en ax1
        ax1.set_yticks(np.arange(len(predictands)*2) )
        ax1.set_yticklabels(predictands*2)

        # Calcular el centro del gráfico
        y_min, y_max = ax1.get_ylim()
        y_center = (y_min + y_max) / 2
        # Dibujar una línea horizontal
        ax1.hlines(y=y_center, xmin=xmin[1], xmax=xmax[1], colors='black', linestyles='dashed', linewidth=1)

        # Agregar una leyenda para cada boxplot
        plt.legend(legend_handles, names, loc='lower right', prop={'size': 10}, frameon=False)


        # Guardar el gráfico
        plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}/{figName}.pdf', bbox_inches='tight')


    # # CLIMATOLOGY
    # xmin = (15, 14)
    # xmax = (40, 41)
    # for shape in SHAPE_NAME:
    #     # Etiquetas
    #     colors = ['lightgreen', 'lightblue', 'lightcoral']
    #     names = ['Short', 'Medium', 'Long']
    #     legend_handles = []

    #     figName = f'fig3_boxPlot_climatology_Ensemble{ENSEMBLE_QUANTITY}_{shape}'
    #     # Crear la figura y los ejes
    #     fig, ax1 = plt.subplots(figsize=(10, 8))

    #     # Graficar cada set de datos (Short, Medium, Long) en el mismo gráfico
    #     for i, period in enumerate(periods):
    #         data_to_plot = []
    #         data_to_plot_99 = []
    #         for predictand_name in predictands:
    #             predictand_data = []
    #             ccsignal_predictand = []
    #             ccsignal_predictand_99 = []
    #             predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]

    #             for predictand_number in predictand_numbered:
    #                 modelName = f'DeepESD_tas_{predictand_number}' 
    #                 loaded_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{period[0]}-{period[1]}.nc')
    #                 grided_data = loaded_data.sel(
    #                     lat=references_grid[shape].lat,
    #                     lon=references_grid[shape].lon,
    #                 ) if shape != 'Iberia' else loaded_data

    #                 grided_data_99 = grided_data.resample(time = 'YE').quantile(0.99, dim = 'time')
    #                 grided_mean = grided_data.mean(dim=['time', 'lat', 'lon']) 
    #                 grided_mean_99 = grided_data_99.mean(dim=['time', 'lat', 'lon'])
    #                 ccsignal_predictand.append(grided_mean)
    #                 ccsignal_predictand_99.append(grided_mean_99)


    #             ccsignal_array = np.array([ds['tasmean'].values for ds in ccsignal_predictand])
    #             data_to_plot.append(ccsignal_array)
    #             ccsignal_array_99 = np.array([ds['tasmean'].values for ds in ccsignal_predictand_99])
    #             data_to_plot_99.append(ccsignal_array_99)

    #         ax = ax1.twiny() if i > 0 else ax1  # Crear ejes adicionales solo para Medium y Long
    #         color = colors[i]
    #         bplot = ax.boxplot(data_to_plot, positions=np.arange(len(predictands)) * 2.0-0.25, widths=0.35, 
    #                         patch_artist=True, boxprops=dict(facecolor=color), vert=False, whis=[5, 95])
    #         ax.set_xlim(xmin[1], xmax[1])
    #         ax.set_xticks([]) if i>0 else ax.set_xticks(np.linspace(xmin[0], xmax[0], 10))
    #         ax.xaxis.set_ticks_position('top')

    #         ax_99 = ax1.twiny() if i > 0 else ax1  # Crear ejes adicionales solo para Medium y Long
    #         bplot = ax_99.boxplot(data_to_plot_99, positions=np.arange(len(predictands)) * 2.0+0.25, widths=0.35, 
    #                         patch_artist=True, boxprops=dict(facecolor=color), vert=False, whis=[5, 95])
    #         ax_99.set_xlim(xmin[1], xmax[1])
            
    #         ax_99.set_xticks([]) if i>0 else ax_99.set_xticks(np.linspace(xmin[0], xmax[0], 10))
    #         ax_99.xaxis.set_ticks_position('bottom')
            
    #         # Asignar la etiqueta del eje X solo para el primer eje (ax1)
    #         if i == 0:
    #             ax.set_xlabel('CC Signal Tasmean')
    #         # Crear un handle de la leyenda solo en la primera iteración para cada conjunto de datos
    #         legend_handles.append(bplot["boxes"][0])

    #     # Etiquetas del eje Y solo en ax1
    #     ax1.set_yticks(np.arange(len(predictands)) * 2.0)
    #     ax1.set_yticklabels(predictands)

    #     # Agregar una leyenda para cada boxplot
    #     plt.legend(legend_handles, names, loc='lower right', prop={'size': 10}, frameon=False)

    #     # Guardar el gráfico
    #     plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
    #     plt.savefig(f'{FIGS_PATH}/{figName}.pdf', bbox_inches='tight')

    print("Figura 3 completada!")
    
### # FIG4 # ####
if '4' in FIGS:
    figName = f'fig4_extremes_ccsignals_Ensemble{ENSEMBLE_QUANTITY}'
    # Crear la figura y los ejes
    fig, axes = plt.subplots(4, 5, figsize=(20, 12), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

    continuousCMAP = plt.get_cmap('hot_r')
    #discreteCMAP = ListedColormap(continuousCMAP(np.linspace(0, 1, 10)))
    discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, 11)[1:]))

    # vmin = 1 if METRIC != '99Percentile' else 3
    # vmax = 11 if METRIC != '99Percentile' else 13

    for i, predictand_name in enumerate(predictands):

        obs_predictand = utils.getPredictand(f'{DATA_PATH_PREDICTANDS_SAVE}', predictand_name, 'tasmean')
        obs_temp = obs_predictand.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
        obs_predictand = utils.maskData(
                    path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                    var='tasmean',
                    to_slice=(yearsTrain[0], yearsTest[1]),
                    objective = obs_predictand.sel(time=slice(*(hist_baseline[0], hist_baseline[1]))),
                    secondGrid = obs_temp)
        obs_predictand_99 = obs_predictand.resample(time = 'YE').quantile(0.99, dim = 'time')
        obs_predictand_mean = obs_predictand.mean(dim='time')
        obs_predictand_mean_99 = obs_predictand_99.mean(dim='time')


        # Future Data
        predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
        predictand_data = {'min_mean': None, 'max_mean': None, 'min_99': None, 'max_99' : None}
        predictand_data_mean = {'min_mean': None, 'max_mean': None, 'min_99': None, 'max_99' : None}
        number_min_max = {'min_mean': None, 'max_mean': None, 'min_99': None, 'max_99' : None}
        grided_mean_list = []
        index = 1
        for predictand_number in predictand_numbered:
            
            modelName = f'DeepESD_tas_{predictand_number}' 
            loaded_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{future_3[0]}-{future_3[1]}.nc')
            loaded_data_99 = loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')

            grided_mean = loaded_data.mean(dim=['time', 'lat', 'lon']) 
            mean_time = loaded_data.mean(dim='time')
            grided_mean_99 = loaded_data_99.mean(dim=['time', 'lat', 'lon']) 
            mean_time_99 = loaded_data_99.mean(dim='time')


            # CHECK MIN AND MAX AND SAVE MIN, MAX, MEAN FOR CCSIGNAL
            if predictand_data_mean['min_mean'] == None or grided_mean['tasmean'].values < predictand_data_mean['min_mean']:
                predictand_data_mean['min_mean']=grided_mean['tasmean'].values
                predictand_data['min_mean']=mean_time
                number_min_max['min_mean'] = index


            if predictand_data_mean['max_mean'] == None or grided_mean['tasmean'].values > predictand_data_mean['max_mean']:
                predictand_data_mean['max_mean']=grided_mean['tasmean'].values
                predictand_data['max_mean']=mean_time
                number_min_max['max_mean'] = index

            if predictand_data_mean['min_99'] == None or grided_mean_99['tasmean'].values < predictand_data_mean['min_99']:
                predictand_data_mean['min_99']=grided_mean_99['tasmean'].values
                predictand_data['min_99']=mean_time_99
                number_min_max['min_99'] = index


            if predictand_data_mean['max_99'] == None or grided_mean_99['tasmean'].values > predictand_data_mean['max_99']:
                predictand_data_mean['max_99']=grided_mean_99['tasmean'].values
                predictand_data['max_99']=mean_time_99
                number_min_max['max_99'] = index
            
            index = index +1

        predictand_data['min_mean'] = predictand_data['min_mean'] - obs_predictand_mean
        predictand_data['max_mean'] = predictand_data['max_mean'] - obs_predictand_mean
        predictand_data['min_99'] = predictand_data['min_99'] - obs_predictand_mean_99
        predictand_data['max_99'] = predictand_data['max_99'] - obs_predictand_mean_99

        for j, (metric, metric_data) in enumerate(predictand_data.items()):

            vmin = 3
            vmax = 13

            ax = axes[j, i]
            if j == 0:
                ax.set_title(f'{predictand_name.capitalize()}', fontsize=16)
            if i == 0:
                ax.text(-0.07, 0.55, f'{metric.capitalize()}', va='bottom', ha='center',
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
            
            number_patch_mean = Patch(color='white', edgecolor='black', label=f'{number_min_max[metric]}')
            ax.legend(handles=[number_patch_mean], loc='lower right', bbox_to_anchor=(1, 0), frameon=False, fontsize=12)

            if i == 0:
                cax = fig.add_axes([0.125, 0.73 - (j * 0.225), 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
                cbar = plt.colorbar(im, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
                cbar.set_ticks(np.linspace(vmin, vmax, 6))
                cbar.ax.tick_params(labelsize=16)

    plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
    plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
    plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
    plt.close()


    figName = f'fig4_extremes_climatology_Ensemble{ENSEMBLE_QUANTITY}'
    # Crear la figura y los ejes
    fig, axes = plt.subplots(4, 5, figsize=(20, 12), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

    continuousCMAP = plt.get_cmap('hot_r')
    #discreteCMAP = ListedColormap(continuousCMAP(np.linspace(0, 1, 10)))
    discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, 11)[1:]))

    # vmin = 1 if METRIC != '99Percentile' else 3
    # vmax = 11 if METRIC != '99Percentile' else 13

    for i, predictand_name in enumerate(predictands):

        # Future Data
        predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
        predictand_data = {'min_mean': None, 'max_mean': None, 'min_99': None, 'max_99' : None}
        predictand_data_mean = {'min_mean': None, 'max_mean': None, 'min_99': None, 'max_99' : None}
        number_min_max = {'min_mean': None, 'max_mean': None, 'min_99': None, 'max_99' : None}
        grided_mean_list = []
        index = 1
        for predictand_number in predictand_numbered:
            
            modelName = f'DeepESD_tas_{predictand_number}' 
            loaded_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{future_3[0]}-{future_3[1]}.nc')
            loaded_data_99 = loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')

            grided_mean = loaded_data.mean(dim=['time', 'lat', 'lon']) 
            mean_time = loaded_data.mean(dim='time')
            grided_mean_99 = loaded_data_99.mean(dim=['time', 'lat', 'lon']) 
            mean_time_99 = loaded_data_99.mean(dim='time')


            # CHECK MIN AND MAX AND SAVE MIN, MAX, MEAN FOR CCSIGNAL
            if predictand_data_mean['min_mean'] == None or grided_mean['tasmean'].values < predictand_data_mean['min_mean']:
                predictand_data_mean['min_mean']=grided_mean['tasmean'].values
                predictand_data['min_mean']=mean_time
                number_min_max['min_mean'] = index


            if predictand_data_mean['max_mean'] == None or grided_mean['tasmean'].values > predictand_data_mean['max_mean']:
                predictand_data_mean['max_mean']=grided_mean['tasmean'].values
                predictand_data['max_mean']=mean_time
                number_min_max['max_mean'] = index

            if predictand_data_mean['min_99'] == None or grided_mean_99['tasmean'].values < predictand_data_mean['min_99']:
                predictand_data_mean['min_99']=grided_mean_99['tasmean'].values
                predictand_data['min_99']=mean_time_99
                number_min_max['min_99'] = index


            if predictand_data_mean['max_99'] == None or grided_mean_99['tasmean'].values > predictand_data_mean['max_99']:
                predictand_data_mean['max_99']=grided_mean_99['tasmean'].values
                predictand_data['max_99']=mean_time_99
                number_min_max['max_99'] = index
            
            index = index +1

        for j, (metric, metric_data) in enumerate(predictand_data.items()):

            vmin = 15
            vmax = 40

            ax = axes[j, i]
            if j == 0:
                ax.set_title(f'{predictand_name.capitalize()}', fontsize=16)
            if i == 0:
                ax.text(-0.07, 0.55, f'{metric.capitalize()}', va='bottom', ha='center',
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
            number_patch_mean = Patch(color='white', edgecolor='black', label=f'{number_min_max[metric]}')
            ax.legend(handles=[number_patch_mean], loc=4, bbox_to_anchor=(1, 0), frameon=False, fontsize=12)

            if i == 0:
                cax = fig.add_axes([0.125, 0.73 - (j * 0.225), 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
                cbar = plt.colorbar(im, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
                cbar.set_ticks(np.linspace(vmin, vmax, 6))
                cbar.ax.tick_params(labelsize=16)

    plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
    plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
    plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
    plt.close()

    print("Figura 4 completada!")

### # FIG5 # ####
if '5' in FIGS:
    for metric in ['Mean', '99Percentile', '1Percentile']:
        for predictand_name in predictands:

            figName = f'fig5_rmse_Mean_Ensemble{ENSEMBLE_QUANTITY}_{predictand_name}_{metric}'
            rmse_test = []
            test_pred = []
            gcm_pred = []

            modelName = f'DeepESD_tas_{predictand_name}' 
            loaded_test_obs = utils.getPredictand(DATA_PATH_PREDICTANDS_SAVE, predictand_name, 'tasmean')
            loaded_test_obs = loaded_test_obs.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
            loaded_test_obs = utils.maskData(
                        path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                        var='tasmean',
                        to_slice=(yearsTrain[0], yearsTest[1]),
                        objective = loaded_test_obs,
                        secondGrid = loaded_test_obs)
            
            predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
            for predictand_number in predictand_numbered:
                modelName = f'DeepESD_tas_{predictand_number}' 
                loaded_test = xr.open_dataset(f'{PREDS_PATH_TEST}predTest_{modelName}.nc')
                rmse = np.sqrt((((loaded_test - loaded_test_obs)**2).mean(dim=['time', 'lat', 'lon']))['tasmean'])
                rmse_test.append(rmse)

                if metric == '99Percentile':
                    loaded_test = loaded_test.resample(time = 'YE').quantile(0.99, dim = 'time')
                elif metric == '1Percentile':
                    loaded_test = loaded_test.resample(time = 'YE').quantile(0.01, dim = 'time')
                test_pred.append(loaded_test.mean(dim=['time', 'lat', 'lon'])['tasmean'])

                
                loaded_pred = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{future_3[0]}-{future_3[1]}.nc')
                if metric == '99Percentile':
                    loaded_pred = loaded_pred.resample(time = 'YE').quantile(0.99, dim = 'time')
                elif metric == '1Percentile':
                    loaded_pred = loaded_pred.resample(time = 'YE').quantile(0.01, dim = 'time')
                gcm_pred.append(loaded_pred.mean(dim=['time', 'lat', 'lon'])['tasmean'])

            # Plotting
            fig, ax1 = plt.subplots(figsize=(8, 6))

            # Cálculo de las líneas de significancia
            coefficients_test = np.polyfit(rmse_test, test_pred, 1)
            coefficients_gcm = np.polyfit(rmse_test, gcm_pred, 1)
            m_test, b_test = coefficients_test
            m_gcm, b_gcm = coefficients_gcm
            pendiente_test = [da * m_test for da in rmse_test]
            pendiente_gcm = [da * m_gcm for da in rmse_test]
            gcm_min, gcm_max = np.min(gcm_pred), np.max(gcm_pred)
            gcm_diff = (gcm_max - gcm_min)*0.2
            test_min, test_max = np.min(test_pred), np.max(test_pred)
            test_diff = (test_max - test_min)*0.1

            # Scatter for gcm_pred on the left Y-axis
            ax1.scatter(rmse_test, gcm_pred, color='red', label='Long (Temperature)', alpha=0.7)
            ax1.set_xlabel('RMSE', fontsize=12)
            ax1.set_ylabel('Long Temperature (°C)', fontsize=12, color='red')
            ax1.set_ylim(gcm_min-gcm_diff, gcm_max+gcm_diff)
            ax1.tick_params(axis='y', labelcolor='red')  # Color de las etiquetas para distinguir
            ax1.grid(True, linestyle='--', alpha=0.5)
            # Línea de significancia para gcm_pred
            ax1.plot(rmse_test, pendiente_gcm + b_gcm, color='red', linestyle='--', label='GCM Regression')

            # Create a second Y-axis for test_pred
            ax2 = ax1.twinx()
            ax2.scatter(rmse_test, test_pred, color='black', label='Test (Temperature)', alpha=0.7)
            ax2.set_ylim(test_min-test_diff, test_max+test_diff)
            ax2.set_ylabel('Test Temperature (°C)', fontsize=12, color='black')
            ax2.tick_params(axis='y', labelcolor='black')  # Color de las etiquetas para distinguir

            # Línea de significancia para test_pred
            ax2.plot(rmse_test, pendiente_test + b_test, color='black', linestyle='--', label='Test Regression')


            # Título y leyendas
            fig.suptitle('RMSE vs Temperature', fontsize=14)
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')

            # Show grid for better readability
            #ax.grid(True, linestyle='--', alpha=0.5)

            plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
            plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
            plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
            plt.close()

            # GRAPH TEMP LONG VS TEST
            figName = f'fig5_temps_Mean_Ensemble{ENSEMBLE_QUANTITY}_{predictand_name}_{metric}'
            # Plotting
            fig, ax = plt.subplots(figsize=(8, 6))
            # xlim = {'Mean': (19, 21), '99Percentile': (20, 35), '1Percentile': (6, 8)}
            # ylim = {'Mean': (13, 14.5), '99Percentile': (15, 25), '1Percentile': (1, 3)}
            # Scatter for dataset 1
            ax.scatter(gcm_pred, test_pred, color='red', label='Long (Temperature)', alpha=0.7)

            # Labels and legend
            ax.set_xlim(np.floor(min([da.values.item() for da in gcm_pred])), np.ceil(max([da.values.item() for da in gcm_pred])))
            ax.set_ylim(np.floor(min([da.values.item() for da in test_pred])), np.ceil(max([da.values.item() for da in test_pred])))
            ax.set_xlabel('Temperature Long', fontsize=12)
            ax.set_ylabel('Temperature Test', fontsize=12)
            ax.set_title('Long vs Test', fontsize=14)
            ax.legend()

            # Show grid for better readability
            ax.grid(True, linestyle='--', alpha=0.5)

            plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
            plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
            plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
            plt.close()

    print("Figura 5 completada!")
### # FIG OPTIONAL VARIANCE# ####
if '0' in FIGS:
    scenario = 'ccsignal'
    yPredLoaded = {scenario: {}}
    yObsLoaded = {scenario: {}}
    for predictand in predictands:
        loaded_test_obs = utils.getPredictand(DATA_PATH_PREDICTANDS_SAVE, predictand, 'tasmean')
        loaded_test_obs = loaded_test_obs.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
        loaded_test_obs = utils.maskData(
                    path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                    var='tasmean',
                    to_slice=(yearsTrain[0], yearsTest[1]),
                    objective = loaded_test_obs,
                    secondGrid = loaded_test_obs)
        yObsLoaded[scenario][predictand] = loaded_test_obs

        predictand_numbered = [f"{predictand}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
        yPredLoaded[scenario][predictand] = {}

        to_sort = {}
        for predictand_number in predictand_numbered:
            modelName = f'DeepESD_tas_{predictand_number}'
            yPredLoaded[scenario][predictand][predictand_number] = xr.open_dataset(f'{PREDS_PATH}predGCM_{modelName}_EC-Earth3-Veg_ssp585_{future_3[0]}-{future_3[1]}.nc')
            to_sort[predictand_number] = yPredLoaded[scenario][predictand][predictand_number].mean(dim=['time', 'lon', 'lat'])['tasmean'].values
        
        sorted_numbers = {}
        for i in range(ENSEMBLE_QUANTITY):
            min_key = min(to_sort, key=to_sort.get)
            sorted_numbers[min_key] = yPredLoaded[scenario][predictand][min_key]
            del to_sort[min_key]


        yPredLoaded[scenario][predictand] = sorted_numbers
            
    yMeanVariances = {scenario: {}}
    y99quanVariances = {scenario: {}}

    # Mean
    yMeanVariances[scenario] = utils.getVariance(yPredLoaded[scenario], yObsLoaded[scenario], metric='mean', percentage=True, type_data='single')
    utils.graphVariances(yMeanVariances, scenario, FIGS_PATH, vmin=0, vmax=100, extra=f'Mean_Ensemble{ENSEMBLE_QUANTITY}_Ordered', extension='png', extra_title=f'Mean')
    utils.graphVariancesMeanSd(yMeanVariances, scenario, FIGS_PATH, vmin=0, vmax=10, extra=f'Mean_Ensemble{ENSEMBLE_QUANTITY}_Ordered', extension='png', extra_title=f'Mean')
    # 99 Quantile
    y99quanVariances[scenario] = utils.getVariance(yPredLoaded[scenario], yObsLoaded[scenario], metric='99quantile', percentage=True, type_data='single')
    utils.graphVariances(y99quanVariances, scenario, FIGS_PATH, vmin=0, vmax=100, extra=f'99Percentile_Ensemble{ENSEMBLE_QUANTITY}_Ordered', extension='png', extra_title=f'99Percentil')
    utils.graphVariancesMeanSd(y99quanVariances, scenario, FIGS_PATH, vmin=0, vmax=10, extra=f'99Percentile_Ensemble{ENSEMBLE_QUANTITY}_Ordered', extension='png', extra_title=f'99Percentil')
    print("Figura Opcional completada!")
