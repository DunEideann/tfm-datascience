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
FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/paper-april/'
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
predictands_map = {'ERA5-Land0.25deg': 'ERA5-Land', 'E-OBS': 'E-OBS','AEMET_0.25deg':'ROCIO-IBEB', 'Iberia01_v1.0':'Iberia01', 'CHELSA': 'CHELSA'}
metrics_1 = ['mean', '99quantile']
predictands_group_1 = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg']
metrics_2 = ['Mean', '99Percentile']
metrics_3 = ['Mean', '99Percentile', '1Percentile']
#metrics_2 = ['Mean', '99Percentile']

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
    total_metrics = {f'{metric_stat}': [] for metric_stat in metrics_1}#{f'{predictand_name}': {f'{metric_stat}': [] for metric_stat in metrics_1} for predictand_name in predictands}
    #group1_metrics = {f'{metric_stat}': [] for metric_stat in metrics_1}


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
        for metric_stat in metrics_1:
            total_metrics[metric_stat].append(whole_obs_metrics['annual'][predictand_name][metric_stat])
            # if predictand_name in predictands_group_1:
            #     group1_metrics[metric_stat].append(whole_obs_metrics['annual'][predictand_name][metric_stat])
 
        # for season_name, months in seasons.items():
        #     whole_obs[season_name][predictand_name] = whole_obs['annual'][predictand_name].isel(time = (whole_obs['annual'][predictand_name].time.dt.season == months))
        #     whole_obs_metrics[season_name][predictand_name] = utils.getMetricsTemp(whole_obs[season_name][predictand_name], short = True)

    
    fig_num = 1
    print(f"WHOLE METRICS")
    for period, data_metrics in whole_obs_metrics.items():
        # Graphs means, p99 and p1
        utils.metricsGraph(datasets_metrics=data_metrics, figs_path=FIGS_PATH, vmin=[5, 15, -8], vmax=[25, 35, 12], pred_type='observation_whole', fig_num = fig_num, period = period, x_map = predictands_map)#, extension='png')
        utils.metricsGraph(datasets_metrics=data_metrics, figs_path=FIGS_PATH, vmin=[5, 15, -8], vmax=[25, 35, 12], pred_type='observation_whole', fig_num = fig_num, period = period, x_map = predictands_map, extension='png')
        fig_num += Decimal('0.1')
    ##### FIGURA PARA VER TAMAÑO DE PIRINEOS
    continuousCMAP2 = plt.get_cmap('cool')    
    discreteCMAPnoWhite2 = ListedColormap(continuousCMAP2(np.linspace(0, 1, 11)[1:]))
    figName = f'pirineos_region'
    # Crear la figura y los ejes
    fig, axes = plt.subplots(1, 1, figsize=(4, 3), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

    lon_min, lon_max = -0.37, 3.37
    lat_min, lat_max = 41.42, 42.80
    data_to_plot = whole_obs_metrics['annual']['E-OBS']['mean']['tasmean']
    # Crear una máscara booleana
    mask = (
        (data_to_plot.lon >= lon_min) & (data_to_plot.lon <= lon_max) &
        (data_to_plot.lat >= lat_min) & (data_to_plot.lat <= lat_max)
    )

    # Aplicar la máscara: los valores fuera de la región se vuelven NaN
    data_to_plot_masked = data_to_plot.where(mask, np.nan)
    # data_to_plot_uncropped = whole_obs_metrics['annual']['E-OBS']['mean']['tasmean']
    # data_to_plot = data_to_plot_uncropped.sel(
    # lon=slice(-0.37, 3.37),
    # lat=slice(41.42, 42.80)
    # )
    im1 = axes.pcolormesh(data_to_plot_masked.coords['lon'].values, data_to_plot_masked.coords['lat'].values,
                        data_to_plot_masked,
                        transform=ccrs.PlateCarree(),
                        cmap=discreteCMAPnoWhite2,
                        vmin=5, vmax=25)
    cax = fig.add_axes([0.125, 0.055, 0.776, 0.065]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
    cbar = plt.colorbar(im1, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
    cbar.set_ticks(np.linspace(5, 25, 6))
    cbar.ax.tick_params(labelsize=15)

    plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.2, hspace=0.002)
    plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
    plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
    plt.close()
    ##### TERMINA FIGURA

    del whole_obs, whole_obs_metrics, obs
    
    # Standard Deviation all and group 1 ()
    total_metrics_concatened = {}
    #group1_metrics_concatened = {}
    std_metrics = {'all': {}}#, 'group1': {}}

    for metric_stat in metrics_1:    
        total_metrics_concatened[metric_stat] = xr.concat(total_metrics[metric_stat], dim='member')
        #group1_metrics_concatened[metric_stat] = xr.concat(group1_metrics[metric_stat], dim='member')
        std_metrics['all'][metric_stat] = total_metrics_concatened[metric_stat].std(dim='member')
        print(f'{metric_stat} min: {std_metrics['all'][metric_stat]['tasmean'].min().item()}')
        print(f'{metric_stat} max: {std_metrics['all'][metric_stat]['tasmean'].max().item()}')
        #std_metrics['group1'][metric_stat] = group1_metrics_concatened[metric_stat].std(dim='member')

    # GRAPHS STANDAR DEVIATION
    utils.stdGraphs(std_metrics=std_metrics, figs_path=FIGS_PATH, vmin=[0, 0], vmax=[1.5, 1.5], pred_type='standard_deviation_3', fig_num=1, period=period, extension='png')
    utils.stdGraphs(std_metrics=std_metrics, figs_path=FIGS_PATH, vmin=[0, 0], vmax=[1.5, 1.5], pred_type='standard_deviation_3', fig_num=1, period=period, extension='pdf')

            
    del total_metrics, total_metrics_concatened, std_metrics#, group1_metrics_concatened, group1_metrics
    print("Figura 1 completada!")

### # FIG2 # ####
if '2' in FIGS:
    stat_metrics = ['rmse', 'bias', 'bias99']
    vminMetric = {'rmse': (0, 0, 20), 'bias': (0, 0, 20), 'bias99': (0.0, 0, 20)}
    vmaxMetric = {'rmse': (2.0, 20, 20), 'bias': (2, 20, 20), 'bias99': (2.0, 20, 20)}
    

    figName = f'fig2_rmse_error_{ENSEMBLE_QUANTITY}'
    # Crear la figura y los ejes
    fig, axes = plt.subplots(3, 5, figsize=(20, 9), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
    

    predictands_total_mean = []
    predictands_group_mean = []
    #predictands_total_inter = []
    stat_realization_mean = {f'{metric}': {} for metric in stat_metrics}
    #stat_realization_group_mean = {f'{metric}': [] for metric in stat_metrics}
    error_std = {f'{stat}': {'all': []} for stat in stat_metrics}
    for i, predictand_name in enumerate(predictands):      
        modelName = f'DeepESD_tas_{predictand_name}' 
        loaded_test_obs = utils.getPredictand(DATA_PATH_PREDICTANDS_SAVE, predictand_name, 'tasmean')
        loaded_test_obs = loaded_test_obs.sel(time=slice(*(yearsTest[0], yearsTest[1])))
        loaded_test_obs = utils.maskData(
                    path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                    var='tasmean',
                    to_slice=(yearsTrain[0], yearsTest[1]),
                    objective = loaded_test_obs,
                    secondGrid = loaded_test_obs)
        
        predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
        
        stat_realizations = {f'{stat}': [] for stat in stat_metrics}

        for predictand_number in predictand_numbered:
            modelName = f'DeepESD_tas_{predictand_number}' 
            loaded_test = xr.open_dataset(f'{PREDS_PATH_TEST}predTest_{modelName}.nc')
            rmse = np.sqrt((((loaded_test - loaded_test_obs)**2).mean(dim=['time']))['tasmean'])
            bias = np.abs(loaded_test.mean(['time'])['tasmean'] - loaded_test_obs.mean(['time'])['tasmean'])
            loaded_test_99 = loaded_test.resample(time = 'YE').quantile(0.99, dim = 'time')
            loaded_test_obs_99 = loaded_test_obs.resample(time = 'YE').quantile(0.99, dim = 'time')
            bias_99 = np.abs(loaded_test_99.mean(['time'])['tasmean'] - loaded_test_obs_99.mean(['time'])['tasmean'])

            stat_realizations['rmse'].append(rmse)
            stat_realizations['bias'].append(bias)
            stat_realizations['bias99'].append(bias_99)

        for stat_name, realization_list in stat_realizations.items():
            stat_concat = xr.concat(realization_list, dim='member')
            stat_realization_mean[stat_name][predictand_name] = stat_concat.mean(dim='member')


        # INTERQUARTIL A MANO
        for j, (stat_name, stat_data) in enumerate(stat_realization_mean.items()):
            if stat_name == 'rmse':
                axisx_name = 'RMSE'
            elif stat_name == 'bias':
                axisx_name = 'Bias-mean'
            elif stat_name == 'bias99':
                axisx_name = 'Bias-99th'
   
            ax = axes[j, i]
            if j == 0:
                ax.set_title(f'{predictands_map[predictand_name]}', fontsize=16)
            if i == 0:
                ax.text(-0.07, 0.55, f'{axisx_name}', va='bottom', ha='center',
                    rotation='vertical', rotation_mode='anchor',
                    transform=ax.transAxes, fontsize=16)
                
            continuousCMAP = plt.get_cmap('viridis_r')
            discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, vmaxMetric[stat_name][2]+1)[vminMetric[stat_name][1]:vmaxMetric[stat_name][1]]))

            ax.coastlines(resolution='10m')
            

            dataToPlot = stat_data[predictand_name]

            im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                                dataToPlot,
                                transform=ccrs.PlateCarree(),
                                cmap=discreteCMAPnoWhite,
                                vmin=vminMetric[stat_name][0], vmax=vmaxMetric[stat_name][0])
                                #norm=BoundaryNorm(bounds, cmap.N))

            if i == 0:
                cax = fig.add_axes([0.125, 0.655 - (j * 0.305), 0.776, 0.02])#DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
                cbar = plt.colorbar(im, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
                ticks = np.linspace(vminMetric[stat_name][0], vmaxMetric[stat_name][0], int(np.floor(vmaxMetric[stat_name][1]+1-vminMetric[stat_name][1])))
                cbar.set_ticks(ticks)
                cbar.ax.tick_params(labelsize=15)
                tick_labels = [tick.get_text() for tick in cbar.ax.get_xticklabels()]
                tick_labels[-1] += '+'
                cbar.ax.set_xticklabels(tick_labels) 

    

    plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
    plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
    plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
    plt.close()



    vminMetric = {'rmse': (0.0, 0, 10), 'bias': (0, 0, 10), 'bias99': (0.0, 0, 10)}
    vmaxMetric = {'rmse': (1.0, 10, 10), 'bias': (1.0, 10, 10), 'bias99': (1.0, 10, 10)}

    continuousCMAP2 = plt.get_cmap('cool')    

    for stat_name in stat_metrics:
        discreteCMAPnoWhite2 = ListedColormap(continuousCMAP2(np.linspace(0, 1, vmaxMetric[stat_name][2]+1)[1:]))
        figName = f'fig2_std_error_{ENSEMBLE_QUANTITY}_{stat_name}'
        # Crear la figura y los ejes
        fig, axes = plt.subplots(1, 1, figsize=(4, 3), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

        rmse_bias_mean_list = {key: list(value.values()) for key, value in stat_realization_mean.items()}
        error_std[stat_name]['all'] = xr.concat(rmse_bias_mean_list[stat_name], dim='member').std(dim='member')   
        #for predictand_name in predictands:
            #if predictand_name in predictands_group_1:
                #stat_realization_group_mean[stat_name].append(stat_realization_mean[stat_name][predictand_name])
        #rmse_bias_mean_group_list = {key: list(value.values()) for key, value in rmse_bias_group_mean.items()}
        #error_std[stat_name]['group1'] = xr.concat(stat_realization_group_mean[stat_name], dim='member').std(dim='member')
        j = 0

        data_to_plot = {'all-std': None}#, 'group1-std': None}

        data_to_plot['all-std'] = error_std[stat_name]['all']
        #data_to_plot['group1-std'] = error_std[stat_name]['group1']

        ax1 = axes

        if stat_name == 'rmse':
            ax1.set_title(f'Std', fontsize=16)
        #ax2.set_title(f'Group 1 Std', fontsize=16)

        ax1.coastlines(resolution='10m')
        #ax2.coastlines(resolution='10m')
        print(f'{stat_name} max: {data_to_plot['all-std'].max().item()}')
        print(f'{stat_name} min: {data_to_plot['all-std'].min().item()}')

        im1 = ax1.pcolormesh(data_to_plot['all-std'].coords['lon'].values, data_to_plot['all-std'].coords['lat'].values,
                            data_to_plot['all-std'],
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAPnoWhite2,
                            vmin=vminMetric[stat_name][0], vmax=vmaxMetric[stat_name][0])

        
        cax = fig.add_axes([0.125, 0.055, 0.776, 0.065]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
        cbar = plt.colorbar(im1, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
        cbar.set_ticks(np.linspace(vminMetric[stat_name][0], vmaxMetric[stat_name][0], 6))
        cbar.ax.tick_params(labelsize=15)


        j += 1


        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.2, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()

    del stat_realization_mean, error_std, data_to_plot, predictands_total_mean, predictands_group_mean, stat_realizations

    print("Figura 2 completada!")
### # FIG3 # ####
if '3' in FIGS:

    # climatology - CCSIGNAL
    vminMetric = {'Mean': (4.6, 0.05, 4, 0), '1Percentile': (1, 0, 1, 1), '99Percentile': (5, 0.25, 2, 2)}
    vmaxMetric = {'Mean': (8.6, 0.85, 14, 8), '1Percentile': (8, 0.5, 26, 14), '99Percentile': (13, 1.25, 22, 12)}
    for metric in metrics_3:

        figName = f'fig3_Statistics_CCSignal_{ENSEMBLE_QUANTITY}_{metric}_part1'
        # Crear la figura y los ejes
        fig, axes = plt.subplots(2, 5, figsize=(20, 6), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
       

        predictands_total_mean = []
        predictands_group_mean = []
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
            predictand_data = {'mean': None, 'std': None}
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

            predictand_data['mean'] = predictand_data_ensemble.mean('member')
            predictand_data['std'] = predictand_data_ensemble.std('member')
            predictands_total_mean.append(predictand_data['mean'])
            # if predictand_name in predictands_group_1:
            #     predictands_group_mean.append(predictand_data['mean'])


            # INTERQUARTIL A MANO
            for j, (metric_fig, metric_data) in enumerate(predictand_data.items()):
                if metric_fig == 'mean':
                    vmin = vminMetric[metric][0]
                    vmax = vmaxMetric[metric][0]
                    num_ticks = 22
                elif metric_fig == 'std':
                    vmin = vminMetric[metric][1]
                    vmax = vmaxMetric[metric][1]
                    num_ticks = 11

                ax = axes[j, i]
                if j == 0:
                    ax.set_title(f'{predictands_map[predictand_name]}', fontsize=16)
                if i == 0:
                    if metric == 'Mean' and j==0:
                        metric_row = metric
                    elif metric == '99Percentile' and j==0:
                        metric_row = '99th'
                    elif metric == '1Percentile' and j==0:
                        metric_row = '1st'
                    else:
                        metric_row = 'Std'
                    ax.text(-0.07, 0.55, f'{metric_row}', va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize=16)

                pos = 2 if metric_fig=='mean' else 3
                continuousCMAP = plt.get_cmap('hot_r') if metric_fig == 'mean' else plt.get_cmap('cool')
                discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, num_ticks+1)[vminMetric[metric][pos]:vmaxMetric[metric][pos]]))

                ax.coastlines(resolution='10m')
                

                dataToPlot = metric_data['tasmean']
                print(f'{metric}/{metric_fig}-{predictand_name} max: {dataToPlot.max().item()}')
                print(f'{metric}/{metric_fig}-{predictand_name} min: {dataToPlot.min().item()}')

                im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                                    dataToPlot,
                                    transform=ccrs.PlateCarree(),
                                    cmap=discreteCMAPnoWhite,
                                    vmin=vmin, vmax=vmax)
                                    #norm=BoundaryNorm(bounds, cmap.N))

                if i == 0:
                    cax = fig.add_axes([0.125, 0.510 - (j * 0.443), 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
                    cbar = plt.colorbar(im, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
                    ticks = np.linspace(vmin, vmax, int(np.floor(vmaxMetric[metric][pos]+1-vminMetric[metric][pos])))# if metric=='mean' else int(np.floor(vmaxMetric[metric][3]+1-vminMetric[metric][3])))
                    cbar.set_ticks(ticks)
                    cbar.ax.tick_params(labelsize=16)
                    tick_labels = [tick.get_text() for tick in cbar.ax.get_xticklabels()]
                    tick_labels[-1] += '+'
                    cbar.ax.set_xticklabels(tick_labels) 

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()


        continuousCMAP2 = plt.get_cmap('cool')
        num_ticks = 11
        discreteCMAP2 = ListedColormap(continuousCMAP2(np.linspace(0, 1, num_ticks)[vminMetric[metric][3]:vmaxMetric[metric][3]]))
        discreteCMAPnoWhite2 = ListedColormap(continuousCMAP2(np.linspace(0, 1, num_ticks+1)[vminMetric[metric][3]:vmaxMetric[metric][3]]))

        figName = f'fig3_Statistics_CCSignal_{ENSEMBLE_QUANTITY}_{metric}_part2'
        # Crear la figura y los ejes
        fig, axes = plt.subplots(1, 1, figsize=(4, 3), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

        data_to_plot = {'all-std': None}#, 'group1-std': None}
        mean_combined = xr.concat(predictands_total_mean, dim='member')

        data_to_plot['all-std'] = mean_combined.std(dim='member')
        print(f'{metric}all std max: {data_to_plot['all-std']['tasmean'].max().item()}')
        print(f'{metric}all std min: {data_to_plot['all-std']['tasmean'].min().item()}')

        ax1 = axes


        ax1.set_title(f'Std', fontsize=16)
        #ax2.set_title(f'Group 1 Std', fontsize=16)

        ax1.coastlines(resolution='10m')
        #ax2.coastlines(resolution='10m')

        im1 = ax1.pcolormesh(data_to_plot['all-std']['tasmean'].coords['lon'].values, data_to_plot['all-std']['tasmean'].coords['lat'].values,
                            data_to_plot['all-std']['tasmean'],
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAPnoWhite2,
                            vmin=vminMetric[metric][1], vmax=vmaxMetric[metric][1])
        print(data_to_plot['all-std']['tasmean'])

        #cax = fig.add_axes([0.28, 0.288, 0.02, 0.425]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
        #cbar = plt.colorbar(im1, cax, pad=0.05, spacing='uniform', orientation='vertical')#, extend='both', extendfrac='auto', )
        #cbar.set_ticks(np.linspace(vminMetric[metric][1], vmaxMetric[metric][1], 6))
        #cbar.ax.tick_params(labelsize=8)

        # im2 = ax2.pcolormesh(data_to_plot['group1-std']['tasmean'].coords['lon'].values, data_to_plot['group1-std']['tasmean'].coords['lat'].values,
        #                     data_to_plot['group1-std']['tasmean'],
        #                     transform=ccrs.PlateCarree(),
        #                     cmap=discreteCMAPnoWhite2,
        #                     vmin=vminMetric[metric][1], vmax=vmaxMetric[metric][1])
        
        cax = fig.add_axes([0.125, 0.08, 0.776, 0.0335]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
        cbar = plt.colorbar(im1, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
        cbar.set_ticks(np.linspace(vminMetric[metric][1], vmaxMetric[metric][1], 5))
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.tick_params(labelsize=16)
        tick_labels = [tick.get_text() for tick in cbar.ax.get_xticklabels()]
        tick_labels[-1] += '+'
        cbar.ax.set_xticklabels(tick_labels) 




        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002) #left=0.05, right=0.82, 
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()

    del predictand_data, predictand_data_ensemble, mean_combined, predictands_total_mean, predictands_group_mean, mean_list

    print("Figura 3 completada!")



### # FIG 4 # ####
if '4' in FIGS:
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
        colors = ['darkgreen', 'darkblue', 'darkred']
        names = ['Short', 'Medium', 'Long']
        legend_handles = []

        figName = f'fig4_boxPlot_ccsignals_Ensemble{ENSEMBLE_QUANTITY}_{shape}'
        # Crear la figura y los ejes
        fig, ax1 = plt.subplots(figsize=(20, 12))

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
                            whiskerprops=dict(color=color), flierprops=dict(color=color, markeredgecolor=color),
                            medianprops=dict(color='snow', linewidth=2))
            ax.set_xlim(xmin[1], xmax[1]-2)
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.tick_top()
            ax.set_xticks(np.linspace(xmin[0], xmax[0]-2, 8))
            ax.tick_params(axis='x', labelsize=16)
 

            ax_99 = ax1.twiny() if i > 0 else ax1  # Crear ejes adicionales solo para Medium y Long
            bplot = ax_99.boxplot(data_to_plot_99, positions=np.arange(len(predictands))+0.1 , widths=0.35, 
                            patch_artist=True, boxprops=dict(facecolor=color), vert=False, whis=[5, 95],
                            whiskerprops=dict(color=color), flierprops=dict(color=color, markeredgecolor=color),
                            medianprops=dict(color='snow', linewidth=1.5))
            ax_99.set_xlim(xmin[1], xmax[1])
            ax_99.xaxis.set_ticks_position('bottom')
            ax_99.set_xticks([]) if i>0 else ax_99.set_xticks(np.linspace(xmin[0], xmax[0], 10))
            ax_99.tick_params(axis='x', labelsize=16)

            
            
            # Asignar la etiqueta del eje X solo para el primer eje (ax1)
            if i == 0:
                ax.set_xlabel(f'CC Signal Tasmean {shape}')
            # Crear un handle de la leyenda solo en la primera iteración para cada conjunto de datos
            legend_handles.append(bplot["boxes"][0])


        # Etiquetas del eje Y solo en ax1
        ax1.set_yticks(np.arange(len(predictands)*2) )
        ax1.set_yticklabels(list(predictands_map.values())*2, fontsize=16)

        # Calcular el centro del gráfico
        y_min, y_max = ax1.get_ylim()
        y_center = (y_min + y_max) / 2
        # Dibujar una línea horizontal
        ax1.hlines(y=y_center, xmin=xmin[1], xmax=xmax[1], colors='black', linestyles='dashed', linewidth=1)

        # Agregar la cuadrícula punteada
        ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax_99.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Agregar etiquetas en la parte superior derecha
        ax1.text(xmax[1] - 1, y_max/2 + 0.5, 'Mean', fontsize=14, fontweight='bold', ha='right', va='top', color='black')
        ax_99.text(xmax[1] - 0.5, + 0.5, '99th Percentile', fontsize=14, fontweight='bold', ha='right', va='top', color='black')

        # Agregar una leyenda para cada boxplot
        plt.legend(legend_handles, names, loc='upper right', prop={'size': 14}, frameon=False)


        # Guardar el gráfico
        plt.savefig(f'{FIGS_PATH}/{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}/{figName}.pdf', bbox_inches='tight')


    print("Figura 4 completada!")
    
### # FIG5 # ####
if '5' in FIGS:
    figName = f'fig5_extremes_ccsignals_Ensemble{ENSEMBLE_QUANTITY}'
    # Crear la figura y los ejes
    fig, axes = plt.subplots(2, 5, figsize=(20, 6), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

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
        predictand_data2 = {}
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
        predictand_data2['Rmax-Rmin (Mean)'] = predictand_data['max_mean'] - predictand_data['min_mean']
        predictand_data2['Rmax-Rmin (99th)'] = predictand_data['max_99'] - predictand_data['min_99']

        vmax = {'Rmax-Rmin (Mean)': 2, 'Rmax-Rmin (99th)': 4}
        vmin = {'Rmax-Rmin (Mean)': 0, 'Rmax-Rmin (99th)': 0}

        colors = {'Rmax-Rmin (Mean)': (1, 6), 'Rmax-Rmin (99th)': (1, 11)}
        for j, (metric, metric_data) in enumerate(predictand_data2.items()):
            
            discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, 11)[colors[metric][0]:colors[metric][1]]))
            ax = axes[j, i]
            if j == 0:
                ax.set_title(f'{predictands_map[predictand_name]}', fontsize=16)
            if i == 0:
                ax.text(-0.07, 0.55, f'{metric}', va='bottom', ha='center',
                    rotation='vertical', rotation_mode='anchor',
                    transform=ax.transAxes, fontsize=16)

            ax.coastlines(resolution='10m')
            

            dataToPlot = metric_data['tasmean']
            im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                                dataToPlot,
                                transform=ccrs.PlateCarree(),
                                cmap=discreteCMAPnoWhite,
                                vmin=vmin[metric], vmax=vmax[metric])
                                #norm=BoundaryNorm(bounds, cmap.N))
            
            #number_patch_mean = Patch(color='white', edgecolor='black', label=f'{metric}')
            #ax.legend(handles=[number_patch_mean], loc='lower right', bbox_to_anchor=(1, 0), frameon=False, fontsize=12)

            if i == 0:
                cax = fig.add_axes([0.125, 0.510 - (j * 0.443), 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
                cbar = plt.colorbar(im, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
                cbar.set_ticks(np.linspace(vmin[metric], vmax[metric], 6))
                cbar.ax.tick_params(labelsize=16)
                tick_labels = [tick.get_text() for tick in cbar.ax.get_xticklabels()]
                tick_labels[-1] += '+'
                cbar.ax.set_xticklabels(tick_labels) 

    plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
    plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
    plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
    plt.close()

    print("Figura 5 completada!")

### # FIG6 # ####
if '6' in FIGS:
    valuesMinMax = {'Mean': (13, 15, 19, 21), '99Percentile': (25, 28, 33, 36), '1Percentile': (0.75, 2.75, 5.75, 7.75)}
    
    for metric in ['Mean', '99Percentile', '1Percentile']:
        observational_mean = {}
        test_pred_total = {}
        gcm_pred_total = {}
        color_list = ['crimson', 'forestgreen', 'royalblue', 'orchid', 'cadetblue']
        color_num = 0
        for predictand_name in predictands:

            figName = f'fig6_rmse_Mean_Ensemble{ENSEMBLE_QUANTITY}_{predictand_name}_{metric}'
            rmse_test = []
            test_pred = []
            gcm_pred = []

            modelName = f'DeepESD_tas_{predictand_name}' 
            loaded_test_obs = utils.getPredictand(DATA_PATH_PREDICTANDS_SAVE, predictand_name, 'tasmean')
            loaded_test_obs = loaded_test_obs.sel(time=slice(*(yearsTest[0], yearsTest[1])))
            loaded_test_obs = utils.maskData(
                        path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                        var='tasmean',
                        to_slice=(yearsTrain[0], yearsTest[1]),
                        objective = loaded_test_obs,
                        secondGrid = loaded_test_obs)
            
            if metric == '99Percentile':
                observed_mean = loaded_test_obs.resample(time = 'YE').quantile(0.99, dim = 'time')
            elif metric == '1Percentile':
                observed_mean = loaded_test_obs.resample(time = 'YE').quantile(0.01, dim = 'time')
            else:
                observed_mean = loaded_test_obs

            observed_mean = observed_mean.mean(dim=['time', 'lat', 'lon'])['tasmean']
            observational_mean[predictands_map[predictand_name]] = observed_mean


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

            color_num += 1
            test_pred_total[predictands_map[predictand_name]] = test_pred
            gcm_pred_total[predictands_map[predictand_name]] = gcm_pred

        figName = f'fig6_temps_Mean_Ensemble{ENSEMBLE_QUANTITY}_{metric}'
        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter for dataset 1
        color_num = 0
        
        for predictand_name in predictands:
            ax.scatter(test_pred_total[predictands_map[predictand_name]], gcm_pred_total[predictands_map[predictand_name]], color=f'{color_list[color_num]}', label=f'{predictand_name}', alpha=0.4)
            ax.plot([valuesMinMax[metric][0], valuesMinMax[metric][3]], [valuesMinMax[metric][0], valuesMinMax[metric][3]], color='black', linestyle='-', linewidth=1.5)
            color_num += 1

        # Labels and legend
        ax.set_xlim(valuesMinMax[metric][0], valuesMinMax[metric][3])#ax.set_xlim(np.floor(min([da.values.item() for da in gcm_pred])), np.ceil(max([da.values.item() for da in gcm_pred])))
        ax.set_ylim(valuesMinMax[metric][0], valuesMinMax[metric][3])#ax.set_ylim(np.floor(min([da.values.item() for da in test_pred])), np.ceil(max([da.values.item() for da in test_pred])))
        ax.set_xlabel('Temperature Test', fontsize=12)
        ax.set_ylabel('Temperature Long', fontsize=12)
        ax.set_title('Long vs Test', fontsize=14)
        ax.legend()

        # Show grid for better readability
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()


        figName = f'fig6_zoom_Ensemble{ENSEMBLE_QUANTITY}_{metric}'
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 9))

        # Scatter for dataset 1
        color_num = 0
        print(f"Observational mean: {metric}")
        print(observational_mean)
        
        for predictand_name in predictands:
            test_values = test_pred_total[predictands_map[predictand_name]]
            gcm_values = gcm_pred_total[predictands_map[predictand_name]]
            ax.scatter(test_values, gcm_values, color=f'{color_list[color_num]}', label=f'{predictands_map[predictand_name]}', alpha=0.7)
            ax.plot([valuesMinMax[metric][0], valuesMinMax[metric][2]], [valuesMinMax[metric][1], valuesMinMax[metric][3]], color='black', linestyle='-', linewidth=1.5)
            ax.axvline(observational_mean[predictands_map[predictand_name]], color=f'{color_list[color_num]}', linestyle='--', linewidth=1)
            #print(f"{predictands_map[predictand_name]}-{metric}: {test_values}")
            test_mean = np.mean([da.data for da in test_values])
            gcm_mean = np.mean([da.data for da in gcm_values])
            ax.scatter(test_mean, gcm_mean, color=f'{color_list[color_num]}', marker='+', s=100)
            color_num += 1

        # Labels and legend
        ax.set_xlim(valuesMinMax[metric][0], valuesMinMax[metric][1])
        ax.set_ylim(valuesMinMax[metric][2],valuesMinMax[metric][3])
        ax.set_xlabel('Temperature Test', fontsize=12)
        ax.set_ylabel('Temperature Long', fontsize=12)
        ax.set_title('Long vs Test', fontsize=14)
        ax.legend()


        # Show grid for better readability
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()



    print("Figura 6 completada!")
    
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
    utils.graphVariances(yMeanVariances, scenario, FIGS_PATH, vmin=0, vmax=100, extra=f'Mean_Ensemble{ENSEMBLE_QUANTITY}_Ordered', extension='pdf', extra_title=f'Mean')
    utils.graphVariancesMeanSd(yMeanVariances, scenario, FIGS_PATH, vmin=0, vmax=10, extra=f'Mean_Ensemble{ENSEMBLE_QUANTITY}_Ordered', extension='pdf', extra_title=f'Mean')
    # 99 Quantile
    y99quanVariances[scenario] = utils.getVariance(yPredLoaded[scenario], yObsLoaded[scenario], metric='99quantile', percentage=True, type_data='single')
    utils.graphVariances(y99quanVariances, scenario, FIGS_PATH, vmin=0, vmax=100, extra=f'99Percentile_Ensemble{ENSEMBLE_QUANTITY}_Ordered', extension='pdf', extra_title=f'99Percentil')
    utils.graphVariancesMeanSd(y99quanVariances, scenario, FIGS_PATH, vmin=0, vmax=10, extra=f'99Percentile_Ensemble{ENSEMBLE_QUANTITY}_Ordered', extension='pdf', extra_title=f'99Percentil')
    print("Figura Opcional completada!")



### # FIG 6 (50 Realizaciones) # ####
# if '6' in FIGS:
#     figName = f'fig6_Climatology_{ENSEMBLE_QUANTITY}_'
#         # Crear la figura y los ejes
#         fig, axes = plt.subplots(ENSEMBLE_QUANTITY, 5, figsize=(20, 3*ENSEMBLE_QUANTITY), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

#         continuousCMAP = plt.get_cmap('hot_r')
#         discreteCMAP = ListedColormap(continuousCMAP(np.linspace(0, 1, 10)))
#         discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, 11)[1:]))


#         predictands_total_mean = []
#         #predictands_total_inter = []
#         for i, predictand_name in enumerate(predictands):

#             # Future Data
#             predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
#             predictand_data = {'mean': None, 'interq': None, 'sd': None}
#             mean_list = []

#             grided_mean_list = []
#             for predictand_number in predictand_numbered:
#                 modelName = f'DeepESD_tas_{predictand_number}' 
#                 loaded_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{future_3[0]}-{future_3[1]}.nc')
#                 if metric == '99Percentile':
#                     loaded_data = loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')
#                 elif metric == '1Percentile':
#                     loaded_data = loaded_data.resample(time = 'YE').quantile(0.1, dim = 'time')
#                 mean_time = loaded_data.mean(dim='time')
#                 mean_list.append(mean_time)