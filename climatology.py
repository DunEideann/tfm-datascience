import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from lib import utils, models, data
import time
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap

FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/climatology'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
DATA_PREDICTORS_TRANSFORMED = '/lustre/gmeteo/WORK/reyess/data/NorthAtlanticRegion_1.5degree/'
MODELS_PATH = '/oceano/gmeteo/users/reyess/tfm/official-code/models'

predictands = ['E-OBS', 'AEMET_0.25deg', 'Iberia01_v1.0', 'pti-grid', 'CHELSA', 'ERA5-Land0.25deg']
seasons = {'spring': 'MAM', 'summer': 'JJA', 'autumn': 'SON', 'winter': 'DJF'}
plot_metrics = ['train', 'test', 'whole']
metrics = ['mean', 'std', '99quantile', 'over30', 'over40', 'mean_max_mean']
yearsTrain = ('1980-01-01', '2003-12-31')
yearsTest = ('2004-01-01', '2015-12-31')

predictors = utils.getPredictors(DATA_PREDICTORS_TRANSFORMED)
data_to_plot = {metric: {season_name: {predictand_name: None for predictand_name in predictands} for season_name in seasons.keys()} for metric in plot_metrics}


for predictand_name in predictands:
    predictand = utils.getPredictand(DATA_PATH_PREDICTANDS_SAVE, predictand_name, 'tasmean')
    predictand = utils.maskData(
        path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
        var='tasmean',
        to_slice=(yearsTrain[0], yearsTest[1]),
        objective = predictand,
        secondGrid = predictand)

    # X and Y Train/Test
    xTrain, xTest, yTrain, yTest = utils.getTrainTest(predictors, predictand, (yearsTrain, yearsTest))

    for season_name, months in seasons.items():
        y_train_season = yTrain.isel(time= (yTrain.time.dt.season == months))
        y_test_season = yTest.isel(time= (yTest.time.dt.season == months))
        y_train_metrics = utils.getMetricsTemp(y_train_season)
        y_test_metrics = utils.getMetricsTemp(y_test_season)
        y_metrics = utils.getMetricsTemp(predictand.sel(time=slice(*(yearsTrain[0], yearsTest[1]))).load())
        data_to_plot['train'][season_name][predictand_name] = y_train_metrics
        data_to_plot['test'][season_name][predictand_name] = y_test_metrics
        data_to_plot['whole'][season_name][predictand_name] = y_metrics

cmap = plt.cm.bwr  # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

start_time = time.time()
for graph_type, seasons_value in data_to_plot.items():
    for metric in metrics: 
        #Cambiar a un diccionario TODO
        if graph_type != 'diff':
            if metric == 'over30':
                v_min = 0
                v_max = 500
            elif metric == 'over40':
                v_min = 0
                v_max = 30
            elif metric == 'std':
                v_min = 0
                v_max = 10
            else:
                v_min = -5
                v_max = 35
        else:
            if metric == 'over30' or metric == 'over40':     
                v_min = -50
                v_max = 50
            elif metric == 'std':
                v_min = 0
                v_max = 1.5
            else:
                v_min = -3
                v_max = 3
        bounds = np.linspace(v_min, v_max, 21)
        norm = BoundaryNorm(bounds, cmap.N)
        #intervalos = np.arange(v_min, v_max+0.1, 10)
        nRows, nCols = 6, 4
        #fig = plt.figure(figsize=(20, 18))#, sharex = True, sharey = True)
        fig, axes = plt.subplots(nRows, nCols, figsize=(20, 18), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
        #gs = fig.add_gridspec(4, 6)
        i = 0
        for season_name, predictand_value in seasons_value.items():
            j = 0

            for predictand_name, value in predictand_value.items():     

                ax = axes[j, i]
                if j == 0:
                    ax.set_title(f'{season_name.capitalize()}', fontsize=14)
                if i == 0:
                    ax.text(-0.07, 0.55, predictand_name, va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize=14)
           
                ax.coastlines(resolution='10m')
                if graph_type != 'diff':
                    dataToPlot = value[metric]['tasmean']
                elif metric != 'std':
                    dataToPlot = (data_to_plot['pred'][season_name][predictand_name][metric] - data_to_plot['real'][season_name][predictand_name][metric])['tasmean']
                else:
                    dataToPlot = (data_to_plot['pred'][season_name][predictand_name][metric]/data_to_plot['real'][season_name][predictand_name][metric])['tasmean']

                im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                                    dataToPlot,
                                    transform=ccrs.PlateCarree(),
                                    cmap=cmap,
                                    norm=BoundaryNorm(bounds, cmap.N))
                #ax.grid(True)
                j += 1
            i += 1

        cax = fig.add_axes([0.91, 0.058, 0.04, 0.88])
        cbar = fig.colorbar(im, cax)#, cmap=cmap_discreto, norm=norm, boundaries=intervalos)#, pad = 0.02, shrink=0.8)
        #|fig.supylabel("HAIWHROIWR")

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        #plt.setp(axes[0, 0].get_ylabel, visible=True)
        plt.savefig(f'{FIGS_PATH}/comparisson_{metric}_{graph_type}.pdf')
        plt.close()

total_time = time.time() - start_time
print(f"El código de graficos de test se ejecutó en {total_time:.2f} segundos.")