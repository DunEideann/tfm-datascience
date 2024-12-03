import xarray as xr
import torch
from lib import utils, models, data, settings
import sys, time
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm

DATA_PATH_PREDICTORS = '/lustre/gmeteo/PTICLIMA/DATA/PROJECTIONS/CMIP6_PNACC/CMIP6_models/'
DATA_PATH_PREDICTANDS_READ = '/lustre/gmeteo/PTICLIMA/DATA/AUX/GRID_INTERCOMP/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/extremes-interquartile/'
MODELS_PATH = '/oceano/gmeteo/users/reyess/tfm/official-code/models'
DATA_PREDICTORS_TRANSFORMED = '/lustre/gmeteo/WORK/reyess/data/NorthAtlanticRegion_1.5degree/'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/GCM/AEMET/'

METRIC = '1Percentile'
PREDICTANDS_SIZE = 5
ENSEMBLE_QUANTITY = 50
GCM_NAME = 'EC-Earth3-Veg'
MAIN_SCENARIO = 'ssp585'

if PREDICTANDS_SIZE == 6:
    predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'pti-grid', 'CHELSA']
elif PREDICTANDS_SIZE == 5:
    predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0','CHELSA']# 'pti-grid']
elif PREDICTANDS_SIZE == 4:
    predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0']#, 'pti-grid', 'CHELSA']
elif PREDICTANDS_SIZE == 3:
    predictands = ['ERA5-Land0.25deg', 'AEMET_0.25deg', 'pti-grid']

hist_reference = ('1980-01-01', '2014-12-31')
hist_baseline = ('1995-01-01', '2014-12-31') #95-14
future_1 = ('2021-01-01', '2040-12-31')
future_2 = ('2041-01-01', '2060-12-31')
future_3 = ('2081-01-01', '2100-12-31') 
future_4 = ('2061-01-01', '2080-12-31')
yearsTrain = ('1980-01-01', '2003-12-31')
yearsTest = ('2004-01-01', '2015-12-31')
periods = [future_2, future_4, future_3]

vminMetric = {'Mean': 10, '1Percentile': 0, '99Percentile': 20}
vmaxMetric = {'Mean': 30, '1Percentile': 20, '99Percentile': 40}


figName = f'interquartil_projection_Ensemble{ENSEMBLE_QUANTITY}_{METRIC}'
# Crear la figura y los ejes
fig, axes = plt.subplots(3, PREDICTANDS_SIZE, figsize=(20, 9), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

continuousCMAP = plt.get_cmap('hot_r')
discreteCMAP = ListedColormap(continuousCMAP(np.linspace(0, 1, 10)))
discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, 11)[1:]))


predictands_total_mean = []
predictands_total_median = []
for i, predictand_name in enumerate(predictands):

    # Future Data
    predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
    predictand_data = {'mean': None, 'sd': None, 'interq': None}
    mean_list = []

    grided_mean_list = []
    for predictand_number in predictand_numbered:
        modelName = f'DeepESD_tas_{predictand_number}' 
        loaded_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{future_3[0]}-{future_3[1]}.nc')
        if METRIC == '99Percentile':
            loaded_data = loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')
        elif METRIC == '1Percentile':
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
    predictands_total_median.append(predictand_data_ensemble.median('member'))

    # INTERQUARTIL A MANO
    #print(np.sort(predictand_data_ensemble.sel(lat=40, lon=-2, method='nearest')['tasmean'].values)[6] - np.sort(predictand_data_ensemble.sel(lat=40, lon=-2, method='nearest')['tasmean'].values)[2])
    #print(np.percentile(np.sort(predictand_data_ensemble.sel(lat=40, lon=-2, method='nearest')['tasmean'].values), 75)-np.percentile(np.sort(predictand_data_ensemble.sel(lat=40, lon=-2, method='nearest')['tasmean'].values), 25))
    for j, (metric, metric_data) in enumerate(predictand_data.items()):
        if metric == 'mean':
            vmin = vminMetric[METRIC]
            vmax = vmaxMetric[METRIC]
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
plt.close()


figName = f'standarDeviation_{ENSEMBLE_QUANTITY}_{METRIC}'
# Crear la figura y los ejes
fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

data_to_plot = {'mean': None, 'sd mean': None, 'median': None, 'sd median': None}
mean_combined = xr.concat(predictands_total_mean, dim='member')
median_combined = xr.concat(predictands_total_median, dim='member')


data_to_plot['mean'] = mean_combined.mean(dim='member')
data_to_plot['sd mean'] = mean_combined.std(dim='member')
data_to_plot['median'] = median_combined.mean(dim='member')
data_to_plot['sd median'] = median_combined.std(dim='member')


vmin_sd = 0
vmax_sd = 0.8

ax1 = axes[0, 0]
ax2 = axes[1, 0]
ax3 = axes[0, 1]
ax4 = axes[1, 1]


ax1.set_title(f'Mean', fontsize=16)
ax3.set_title(f'Median', fontsize=16)


ax1.text(-0.07, 0.55, 'Mean', va='bottom', ha='center',
    rotation='vertical', rotation_mode='anchor',
    transform=ax1.transAxes, fontsize=16)
ax2.text(-0.07, 0.55, 'StandarDeviation', va='bottom', ha='center',
    rotation='vertical', rotation_mode='anchor',
    transform=ax2.transAxes, fontsize=16)

ax1.coastlines(resolution='10m')
ax2.coastlines(resolution='10m')
ax3.coastlines(resolution='10m')
ax4.coastlines(resolution='10m')

im1 = ax1.pcolormesh(data_to_plot['mean']['tasmean'].coords['lon'].values, data_to_plot['mean']['tasmean'].coords['lat'].values,
                    data_to_plot['mean']['tasmean'],
                    transform=ccrs.PlateCarree(),
                    cmap=discreteCMAPnoWhite,
                    vmin=vminMetric[METRIC], vmax=vmaxMetric[METRIC])

cax = fig.add_axes([0.125, 0.53, 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
cbar = plt.colorbar(im1, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
cbar.set_ticks(np.linspace(vminMetric[METRIC], vmaxMetric[METRIC], 6))
cbar.ax.tick_params(labelsize=16)

im3 = ax3.pcolormesh(data_to_plot['median']['tasmean'].coords['lon'].values, data_to_plot['median']['tasmean'].coords['lat'].values,
                    data_to_plot['median']['tasmean'],
                    transform=ccrs.PlateCarree(),
                    cmap=discreteCMAPnoWhite,
                    vmin=vminMetric[METRIC], vmax=vmaxMetric[METRIC])

# Desviaciones estandar
im2 = ax2.pcolormesh(data_to_plot['sd mean']['tasmean'].coords['lon'].values, data_to_plot['sd mean']['tasmean'].coords['lat'].values,
                    data_to_plot['sd mean']['tasmean'],
                    transform=ccrs.PlateCarree(),
                    cmap=discreteCMAPnoWhite,
                    vmin=vmin_sd, vmax=vmax_sd)

cax = fig.add_axes([0.125, 0.115, 0.776, 0.02]) #DIST DESDE IZQUIERDA/DIST DESDE ABAJO/LARDO HORI/LARGO/VERT
cbar = plt.colorbar(im2, cax, pad=0.05, spacing='uniform', orientation='horizontal')#, extend='both', extendfrac='auto', )
cbar.set_ticks(np.linspace(vmin_sd, vmax_sd, 6))
cbar.ax.tick_params(labelsize=16)

im4 = ax4.pcolormesh(data_to_plot['sd median']['tasmean'].coords['lon'].values, data_to_plot['sd median']['tasmean'].coords['lat'].values,
                    data_to_plot['sd median']['tasmean'],
                    transform=ccrs.PlateCarree(),
                    cmap=discreteCMAPnoWhite,
                    vmin=vmin_sd, vmax=vmax_sd)


plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
plt.close()