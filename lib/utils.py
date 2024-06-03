import os
import copy
from time import time
import numpy as np
import xarray as xr
import pandas as pd
import torch
from scipy import signal, stats
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import time


def checkUnitsTempt(data, var):
    """Check units of tempeture and transform them if necessary.

    Args:
        data (xarray.dataset): Dataset de xarray.
        var (string): Variable a transformar.

    Returns:
        xarray.dataset: Sme dataset with tranformed values if necessary.
    """
    is_kelvin = (data[var] > 100).any(dim=('time', 'lat', 'lon'))

    if is_kelvin:
        data[var].data = data[var].data - 273.15

    return data
    

def getMetricsTemp(data, var = None):#, mask=None):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    if var == None:
        var = 'tasmean'
    val_mean = data.mean(dim = 'time')
    val_mean_annual = data.resample(time = 'YE').mean()
    val_st = data.std(dim = 'time')
    val_99 = data.quantile(0.99, dim = 'time')
    over30 = data[var].where(data[var] >= 30).sum(dim='time').to_dataset(name=var)
    over30 = over30.where(over30 != 0, np.nan)
    #over30 = filterByMask(mask=mask, data=over30_prep, var='tasmean').sum(dim='time')
    over40 = data[var].where(data[var] >= 40).sum(dim='time').to_dataset(name=var)
    over40 = over40.where(over40 != 0, np.nan)
    #over40 = filterByMask(mask=mask, data=over40_prep, var='tasmean').sum(dim='time')
    mean_max_mean = data.resample(time = 'YE').max(dim='time').mean(dim='time')

    #over30 = data['tasmean'].where(data['tasmean'] >= 30 and data['tasmean'] != None).sum(dim='time', keep_attrs='all', skipna=True).to_dataset(name='tasmean')
    #over40 = data['tasmean'].where(data['tasmean'] >= 40 and data['tasmean'] != None).sum(dim='time', keep_attrs='all', skipna=True).to_dataset(name='tasmean')

    return {
        'mean': val_mean,
        '99quantile': val_99,
        'std': val_st,
        'trend': val_mean_annual,
        'over30': over30,
        'over40': over40,
        'mean_max_mean': mean_max_mean
        }

def __graphTrend(metrics, season_name, folder_path, pred_name, extra = ''):
    """_summary_

    Args:
        metrics (_type_): _description_
        season_name (_type_): _description_
        folder_path (_type_): _description_
        pred_name (_type_): _description_
        extra (str, optional): _description_. Defaults to ''.
    """
    # Get coordinates from your data
    lons = metrics['trend']['lon'].values
    lats = metrics['trend']['lat'].values

    # Create empty DataArray for each variable (fill with NaNs initially)
    slope = xr.DataArray(np.nan * np.ones((len(lats), len(lons))), dims=('lat', 'lon'), coords={'lat': lats, 'lon': lons})
    intercept = xr.DataArray(np.nan * np.ones((len(lats), len(lons))), dims=('lat', 'lon'), coords={'lat': lats, 'lon': lons})
    r_value = xr.DataArray(np.nan * np.ones((len(lats), len(lons))), dims=('lat', 'lon'), coords={'lat': lats, 'lon': lons})
    p_value = xr.DataArray(np.nan * np.ones((len(lats), len(lons))), dims=('lat', 'lon'), coords={'lat': lats, 'lon': lons})
    std_err = xr.DataArray(np.nan * np.ones((len(lats), len(lons))), dims=('lat', 'lon'), coords={'lat': lats, 'lon': lons})


    # Loop through latitudes and longitudes, performing calculations and filling DataArrays
    #y_pred_metrics
    # TODO : Evitar 2 ciclos for usando zip o map o product
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            slope_val, intercept_val, r_val, p_val, std_err_val = stats.linregress(
                metrics['trend'].isel(lon=j, lat=i)['tasmean'].values,
                np.arange(len(metrics['trend'].time)))
            slope[i, j] = slope_val
            intercept[i, j] = intercept_val
            r_value[i, j] = r_val
            p_value[i, j] = p_val
            std_err[i, j] = std_err_val

    # Combine DataArrays into a single dataset
    linreg_results = xr.Dataset({
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err
    })


    fig = plt.figure(figsize=(4, 4))

    axs = (fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()))
    axs.coastlines(resolution='10m')
    axs.set_title(f'Trend {season_name} {pred_name}')

    XX, YY = np.meshgrid(metrics['trend'].coords['lon'].values, metrics['trend'].coords['lat'].values)
    cs = axs.pcolormesh(XX, YY, slope, cmap='cool', zorder = 1)#, norm=norm)

    zm = np.ma.masked_less_equal(p_value, 0.05)
    axs.pcolor(metrics['trend'].coords['lon'].values, metrics['trend'].coords['lat'].values, zm, hatch='////', alpha=0.0, zorder = 10)
    plt.savefig(f'{folder_path}{pred_name}/trend_{season_name}_{extra}.pdf')
    plt.close()

def getGraphsTemp(val_metrics, pred_metrics, season_name, folder_path, pred_name):
    """_summary_

    Args:
        val_metrics (_type_): _description_
        pred_metrics (_type_): _description_
        season_name (_type_): _description_
        folder_path (_type_): _description_
        pred_name (_type_): _description_
    """
    # Check if target folder to save figs exists, if not creates it
    if not os.path.exists(f'{folder_path}{pred_name}'):
        os.makedirs(f'{folder_path}{pred_name}')

    __graphTrend(val_metrics, season_name, folder_path, pred_name,'real')
    __graphTrend(pred_metrics, season_name, folder_path, pred_name, 'pred')

    # Removemos elemento para poder hacer tranquilamente una iteracino
    del val_metrics['trend']
    #Visualize the prediction for a certain day
    for key, value in val_metrics.items():
        nRows, nCols = 1, 3
        fig = plt.figure(figsize=(15, 4))

        # TEMPERATURA OBJETIVO
        ax = fig.add_subplot(nRows, nCols, 1, projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m')
        ax.set_title(f'Target temperature {season_name}')

        dataToPlot = value['tasmean']
        im = plt.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                            dataToPlot,
                            transform=ccrs.PlateCarree(),
                            cmap='RdBu_r')
        # TEMPERATURA PREDICHA
        ax = fig.add_subplot(nRows, nCols, 2, projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m')
        ax.set_title(f'Predicted temperature {season_name}')

        dataToPlot = pred_metrics[key]['tasmean']
        im = plt.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                            dataToPlot,
                            transform=ccrs.PlateCarree(),
                            cmap='RdBu_r')
        vmin = np.min([np.min(value['tasmean']), np.min(pred_metrics[key]['tasmean'])])
        vmax = np.max([np.max(value['tasmean']), np.max(pred_metrics[key]['tasmean'])])

        plt.colorbar(im, ax=ax)
        plt.clim(vmin, vmax)# AGREGADO PARA LEYENDA

        # DIFERENCIAL DE TEMPERATURA
        ax = fig.add_subplot(nRows, nCols, 3, projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m')
        ax.set_title(f'Diff temperature {season_name}')

        dataToPlot = pred_metrics[key]['tasmean']-value['tasmean'] if key != 'std' else pred_metrics[key]['tasmean']/value['tasmean']
        im = plt.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                            dataToPlot,
                            transform=ccrs.PlateCarree(),
                            cmap='RdBu_r')

        v = np.max([abs(np.min(dataToPlot)), abs(np.max(dataToPlot))])
        plt.colorbar(im, ax=ax)
        plt.clim(-v, v) if key != 'std' else plt.clim(0.5, 1.5)
        plt.tight_layout()
        plt.savefig(f'{folder_path}{pred_name}/predictionComparison_{season_name}_{key}.pdf')
        plt.close()

def getGraphsTempGCM(pred_metrics, scenario, folder_path, pred_name, model_name):
    """_summary_

    Args:
        pred_metrics (_type_): _description_
        scenario (_type_): _description_
        folder_path (_type_): _description_
        pred_name (_type_): _description_
        model_name (_type_): _description_
    """
    # Check if target folder to save figs exists, if not creates it
    if not os.path.exists(f'{folder_path}{pred_name}_pred_{model_name}'):
        os.makedirs(f'{folder_path}{pred_name}_pred_{model_name}')

    __graphTrend(pred_metrics, scenario, folder_path, f'{pred_name}_pred_{model_name}')

    # Removemos elemento para poder hacer tranquilamente una iteracino
    del pred_metrics['trend']
    #Visualize the prediction for a certain day
    for key, value in pred_metrics.items():
        nRows, nCols = 1, 1
        fig = plt.figure(figsize=(5*nCols, 4))


        # TEMPERATURA PREDICHA
        ax = fig.add_subplot(nRows, nCols, 1, projection=ccrs.PlateCarree())
        ax.coastlines(resolution='10m')
        ax.set_title(f'Predicted temperature {scenario}')

        dataToPlot = value['tasmean']
        im = plt.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                            dataToPlot,
                            transform=ccrs.PlateCarree(),
                            cmap='RdBu_r')
        vmin = np.min(value['tasmean'])
        vmax = np.max(value['tasmean'])

        plt.colorbar(im, ax=ax)
        plt.clim(vmin, vmax)# AGREGADO PARA LEYENDA

        plt.tight_layout()
        plt.savefig(f'{folder_path}{pred_name}_pred_{model_name}/prediction_{scenario}_{key}.pdf')
        plt.close()

def checkIndex(dataset):
    """_summary_

    Args:
        dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataset_modificado = dataset
    lat_diff = dataset['lat'].diff('lat')
    lon_diff = dataset['lon'].diff('lon')
    if np.all(lat_diff < 0):
        dataset_modificado = dataset_modificado.reindex(lat=list(reversed(predictand.lat)))
    if np.all(lon_diff < 0):
        dataset_modificado = dataset_modificado.reindex(lon=list(reversed(predictand.lon)))

    return dataset_modificado

def getFileName(data_path, target_name, keyword):
    """_summary_

    Args:
        data_path (_type_): _description_
        target_name (_type_): _description_
        keyword (_type_): _description_

    Returns:
        _type_: _description_
    """
    file_name = None
    predictand_folder_path = f'{data_path}{target_name}'
    files_in_predictand_folder = os.listdir(predictand_folder_path)
    for file in files_in_predictand_folder:
        if target_name.lower() in file.lower() and keyword in file.lower():
            file_name = file
            break
    return file_name

def loadGcm(gcm, scenario, to_slice, gcm_path, optVar=None):
    """_summary_

    Args:
        gcm (_type_): _description_
        scenario (_type_): _description_
        gcm_path (_type_): _description_

    Returns:
        _type_: _description_
    """

    if gcm in ('CNRM-ESM2-1', 'UKESM1-0-LL'):
        gcm_run = 'r2i1p1f2'
    else:
        gcm_run = 'r1i1p1f1'

    years_1 = '19500101-20141231'
    years_2 = '20150101-21001231'

    
    # if optVar != None:
    #     vars_mapping = optVar
    # else:
    vars_mapping = {'ta': 't',
                'hus': 'q',
                'va': 'v',
                'ua': 'u',
                'psl': 'msl'}

    heights_mapping = {500.0: '500',
                    700.0: '700',
                    850.0: '850'}

    varsData = []
    for var in vars_mapping.keys():
        data_1 = xr.open_dataset(f'{gcm_path}/{var}_{gcm}_historical_{gcm_run}_{years_1}.nc')
        data_2 = xr.open_dataset(f'{gcm_path}/{var}_{gcm}_{scenario}_{gcm_run}_{years_2}.nc')
        data = xr.merge([data_1, data_2]).sel(time=slice(*to_slice))
        data = data.drop_dims('bnds')

        if var not in ('psl') and var != 'tas':
            data['plev'] = data['plev'] / 100
            for height in heights_mapping.keys():
                dataAux = data.sel(plev=height)
                dataAux = dataAux.drop_vars('plev')
                dataAux = dataAux.rename({var: f'{vars_mapping[var]}{heights_mapping[height]}'})
                dataAux = dataAux.reindex(lat=list(reversed(dataAux.lat)))
                varsData.append(dataAux)
        else:
            dataAux = data.rename({var: f'{vars_mapping[var]}'})
            dataAux = dataAux.reindex(lat=list(reversed(dataAux.lat)))
            varsData.append(dataAux)

    predictor_gcm = xr.merge(varsData)
    predictor_gcm = predictor_gcm.assign_coords({'time': predictor_gcm.indexes['time'].normalize()})


    return predictor_gcm

def maskArray(masker, to_mask):
    """_summary_

    Args:
        masker (_type_): _description_
        to_mask (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Define target domain (mask)
    predictandMask = masker.isnull().astype('int').mean('time')
    predictandMask = xr.where(predictandMask < 1, 1, 0)

    # Select the valid (to downscale) gridpoints
    predictandMaskValid = predictandMask.stack(gridpoint=('lat', 'lon'))
    predictandMaskValid = predictandMaskValid.where(predictandMaskValid ==
    1, drop=True)

    # Subset the yTrain data
    to_mask_array = to_mask.stack(gridpoint=('lat', 'lon'))
    to_mask_array = to_mask_array.where(to_mask_array['gridpoint'] ==
    predictandMaskValid['gridpoint'],
                                        drop=True)
    to_mask_array = downscaler.toArray(to_mask_array)

    return to_mask_array

def loadSurfaceGcm(gcm, var, scenario, gcm_path):
    """_summary_

    Args:
        gcm (_type_): _description_
        var (_type_): _description_
        scenario (_type_): _description_
        gcm_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    if gcm in ('CNRM-ESM2-1', 'UKESM1-0-LL'):
        gcm_run = 'r2i1p1f2'
    else:
        gcm_run = 'r1i1p1f1'

    if scenario == 'historical':
        years = '19500101-20141231'
    else:
        years = '20150101-21001231'

    data = xr.open_dataset(f'{gcm_path}/{var}_{gcm}_{scenario}_{gcm_run}_{years}.nc')
    data = data.drop_dims('bnds')

    data = data.reindex(lat=list(reversed(data.lat)))
    data = data.assign_coords({'time': data.indexes['time'].normalize()})

    return data

def checkCorrectData(dataset, name_transformation={'latitude': 'lat', 'longitude': 'lon'}):
    """
    Checks dataset dimensions and coordinates names.
    If they are not in the format 'lat', 'lon', 'time', transforms them only if possible.

    Args:
        dataset (xarray.Dataset): The dataset to be checked and potentially transformed.
        name_transformation (dict, optional): A dictionary mapping old names to new names.
            Defaults to {'latitude': 'lat', 'longitude': 'lon'}.

    Returns:
        xarray.Dataset: The original dataset if no transformation is needed,
            otherwise the transformed dataset.

    Raises:
        ValueError: If transformation is not possible due to name conflicts.
    """

    # Check if transformation is needed
    current_dims = set(dataset.dims)
    current_coords = set(dataset.coords.keys())
    for key,value in name_transformation.items():

        coord_transform_req = False
        dim_transform_req = False

        if value not in current_coords:
            coord_transform_req = True

        if value not in current_dims:
            dim_transform_req = True

        if key not in current_coords:
            coord_transform_req = False

        if key not in current_dims:
            dim_transform_req = False

        # If transformation is possible, perform rename
        if dim_transform_req:
            print(f"Dimension transformation done successfully for {key}:{value}!")
            try:
                dataset = dataset.rename_dims({key: value})
            except:
                raise Exception("Dim transform error!")
        else:
            print(f"Transformation for dimensions not needed for {key}:{value}")

        if coord_transform_req:
            print(f"Coordinates transformation done successfully for {key}:{value}!")
            try:
                dataset = dataset.rename_vars({key: value})
            except:
                raise Exception("Coord transform error!")
        else:
            print(f"Transformation for coordinates not needed for {key}:{value}")

    return dataset

def removeNANs(grid, var=None, verbose=1, coordNames={'lat': 'lat', 'lon': 'lon'}):
    '''
    Remove nans or null values from a Dataset by removing observations corresponding
    with a day with one or more null values

    @grid: Grid containing the data to plot. It must be a xarray containing 2D
           variables
    @var: Name of the variable to remove nans from. If None the operation
          will be applied to all the variables of the Dataset
    @verbose: Control the printing of logs
    @coordNames: Dictionary with lat and lon names of the grid
    '''

    if var != None:
        raise ValueError('Not implemented')
    else:

        # Get time indices with zero null values
        nansIndices = grid.isnull()
        nansIndices = nansIndices.sum(dim=(coordNames['lat'], coordNames['lon'])).to_array().values
        nansIndices = np.logical_or.reduce(nansIndices, axis=0)
        nansIndices = ~nansIndices

        # Keep just this observations
        grid = grid.sel(time = nansIndices)

        # Return whether any observation was removed
        if verbose==1:
            if np.sum(nansIndices) == len(nansIndices):
                print('There are no observations containing null values')
            else:
                print('Removing observations contaning null values')

    return grid

def alignDatasets(grid1, grid2, coord):
    '''
    Align two grids taking into account the coordinate coord

    @grid1: First grid to align
    @grid2: Second grid to align
    @coord: Coordinate used to perform the filtering
    '''

    grid1 = grid1.sel(time = np.in1d(grid1[coord].values,
                                     grid2[coord].values))
    grid2 = grid2.sel(time = np.in1d(grid2[coord].values,
                                     grid1[coord].values))

    return grid1, grid2

def toArray(grid, coordNames={'lat': 'lat', 'lon': 'lon'}):
    '''
    Transforms the data from a Dataset into a numpy array, maintaining all the
    variables

    @grid: Dataset to transform
    @coordNames: Dictionary with lat and lon names of the grid
    '''

    # Get name of all the variables of the Dataset
    vars = [i for i in grid.data_vars]

    # Get name of all the dimensions of the Dataset
    dims = [i for i in grid.coords]

    # Check whether we are working with X or Y Datasets
    # In the former case we must have latitude and longitude dimensions. In the
    # latter we must have a gridpoint variable (multindex of latitude and longitude)
    if ('gridpoint' in dims) and (len(vars) == 1): # Y case

        if 'time' in grid.coords:
            npGrid = np.empty((grid['time'].shape[0], grid['gridpoint'].shape[0]))
        else:
            npGrid = np.empty((1, grid['gridpoint'].shape[0]))

        npGrid = grid[vars[0]].values

    elif (len(vars) > 1): # X case

        if 'time' in grid.coords:
            npGrid = np.empty((grid['time'].shape[0], len(vars),
                           grid[coordNames['lat']].shape[0], grid[coordNames['lon']].shape[0]))
        else:
            npGrid = np.empty((1, len(vars),
                           grid[coordNames['lat']].shape[0], grid[coordNames['lon']].shape[0]))
        
        for idx, var in enumerate(vars):
            npGrid[:, idx, :, :] = grid[var].values

    elif ('gridpoint' not in dims)  and(len(vars) == 1): # fully convolutional Y case

        if 'time' in grid.coords:
            npGrid = np.empty((grid['time'].shape[0],
                           grid[coordNames['lat']].shape[0], grid[coordNames['lon']].shape[0]))
        else:
            npGrid = np.empty((1, grid[coordNames['lat']].shape[0], grid[coordNames['lon']].shape[0]))

        npGrid = grid[vars[0]].values

    else:
        raise ValueError('Please provide a Dataset with either gridpoint or ' \
                         'latitude and longitude dimensions')

    return npGrid

class flattenSpatialGrid():
    '''
    Transform a 2D variable into a 1D vector. First it flattens the 2D variable,
    then it removes the observations with NANs values.
    '''

    def __init__(self, grid, var, coordNames={'lat': 'lat', 'lon': 'lon'}):
        '''
        Initialize the flattening operation wit the reference grid. This grid is
        used to compute the NAN mask.

        @grid: Reference grid containing the data to flatten. It must be a Dataset
               containing 2D variables
        @var: Variable to flatten
        '''

        self.latName = coordNames['lat']
        self.lonName = coordNames['lon']

        # Save lat and lon dimensions
        self.lat = grid[self.latName].copy()
        self.lon = grid[self.lonName].copy()

        # Save new dimensions with NANs
        newGrid = grid.stack(gridpoint=(self.latName, self.lonName))

        # Compute and save NANs mask
        self.nanIndices = np.isnan(newGrid[var])
        self.nanIndices = np.any(self.nanIndices, axis=0)
        del newGrid

        # Create refArray for getPosition functions
        self.refArray = self.nanIndices
        self.refArray = self.refArray.where(~self.nanIndices, drop=True)
        self.refArray.values = np.arange(0, self.refArray.values.shape[0])

        # Save grid's template lat
        self.gridTemplate = grid.sel(time=grid['time'].values[0])
        self.gridTemplate = self.gridTemplate.stack(gridpoint=(self.latName, self.lonName))
        self.gridTemplate = self.gridTemplate.expand_dims('time')
        self.gridTemplate = xr.where(cond=self.gridTemplate[var] != np.nan,
                                     x=np.nan, y=np.nan)

    def flatten(self, grid, var):
        '''
        Perform the flattening taking into acount the Dataset provided as reference
        in the __init__ method

        @grid: Grid containing the data to flatten. It must be a Dataset
               containing 2D variables
        @var: Variable to flatten
        '''

        # Check dimensions of grid to flatten
        if np.array_equal(grid[self.latName].values, self.lat.values) and \
           np.array_equal(grid[self.lonName].values, self.lon.values):

           # Flatten grid
           newGrid = grid.stack(gridpoint=(self.latName, self.lonName))

           # Filter NANs
           newGrid = newGrid.where(~self.nanIndices, drop=True)

           return newGrid

        else:
            raise ValueError('Discrepancies found in the latitude and longitude dimensions between grids')

    def unFlatten(self, grid, var, revertLat=False):
        '''
        Unflatten the grid taking into account the grid passed to the __init__ method

        @grid: Grid containing the data to unflatten. It must be a Dataset
               containing 2D variables
        @var: Variable to unflatten
        @var: Whether to revert the latitude coordinate
        '''

        # Create a dataset with all the gridpoints spanning the time of the input grid
        refGrid = self.gridTemplate.reindex({'time': grid['time'].values})
        
        # Merge grids and unstack
        finalGrid = grid.combine_first(refGrid)
        finalGrid = finalGrid.unstack('gridpoint')

        if revertLat:
            finalGrid = finalGrid.reindex(lat=list(reversed(finalGrid.lat)))

        return finalGrid

def validSet_fromArray(Xarray, Yarray, validPerc, seed=None):
    '''
    Split the X and Y arrays into a training and validation array. For this, it takes as
    reference for the split the validPerc argument. If Xarray is a list of arrays, the split
    is performed over all of them, in this case a list of arrays is returned

    @Xarray: X array (or list of arrays) to split intro training and validation set
    @Yarray: Y array to split intro training and validation set
    @validPerc: Percentage of data to destinate to the validation set
    @seed: Seed to fix the random generation of indices
    '''

    if seed != None:
        np.random.seed(seed)

    if isinstance(Xarray, list):

        idxs = list(range(Xarray[0].shape[0]))
        np.random.shuffle(idxs)

        XarrayTrain = list()
        XarrayValid = list()

        for i, elem in enumerate(Xarray):
            Xarray_aux = np.array(elem[idxs], copy = True).copy()
            Yarray_aux = np.array(Yarray[idxs], copy = True)

            split_threshold = round((1 - validPerc) * len(idxs))

            XarrayTrain.append(Xarray_aux[:split_threshold])
            YarrayTrain = Yarray_aux[:split_threshold]

            XarrayValid.append(Xarray_aux[split_threshold:])
            YarrayValid = Yarray_aux[split_threshold:]

    else:

        idxs = list(range(Xarray.shape[0]))
        np.random.shuffle(idxs)

        Xarray_aux = np.array(Xarray[idxs], copy = True)
        Yarray_aux = np.array(Yarray[idxs], copy = True)

        split_threshold = round((1 - validPerc) * len(idxs))

        XarrayTrain = Xarray_aux[:split_threshold]
        YarrayTrain = Yarray_aux[:split_threshold]

        XarrayValid = Xarray_aux[split_threshold:]
        YarrayValid = Yarray_aux[split_threshold:]

    return XarrayTrain, YarrayTrain, XarrayValid, YarrayValid

def train_model(model, modelName, modelPath, device,
                lossFunction, optimizer, numEpochs, patience,
                trainData, validData):

    '''
    Train a deep learning model using a standard training loop (pytorch)

    @model: Model to train
    @modelName: Name of the model used to save it as a file
    @modelPath: Path used to save the trained model
    @device: Device used to train the model (cpu or cuda)
    @lossFunction: Loss function to optimize
    @optimizer: Optimizer used to update the parameters
    @numEpochs: Maximum number of epochs
    @patience: Maximum number of epochs without an improvement in the loss function (early stopping)
    @trainData: Dataloader containing the training data
    @validData: Dataloader containing the validation data
    '''

    # Load model into the corresponding device
    model = model.to(device)

    # Epoch counter for early stopping
    epoch_early_stopping = 1

    # Save losses during training
    avg_train_loss_list = []
    avg_val_loss_list = []

    # Earlystopping
    best_model = copy.deepcopy(model.state_dict())
    lowest_loss = 10000

    # Iterate over epochs
    for epoch in range(1, numEpochs + 1):

        epochStart = time()

        # Training step
        model.train()
        loss_train_list = [] # loss per batch

        # Iterate over batches
        for X, Y in trainData:

            X = X.to(device)
            Y = Y.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X)
            loss = lossFunction(outputs, Y)
            loss_train_list.append(loss.item())

            # Compute gradients
            loss.backward()

            # Update parameters
            optimizer.step()

        # Train loss on epoch
        avg_train_loss_list.append(np.mean(loss_train_list))

        # Validation step
        model.eval()
        avg_val_loss_list_aux = []

        for X_val, Y_val in validData:

            X_val = X_val.to(device)
            Y_val = Y_val.to(device)

            # Forward pass
            outputs = model(X_val)

            # Compute loss
            loss = lossFunction(outputs, Y_val)
            avg_val_loss_list_aux.append(loss.item())

        # Validation loss on epoch
        avg_val_loss_list.append(np.mean(avg_val_loss_list_aux))

        epochEnd = time()

        # Epoch log
        log = 'Epoch {} ({:.2f}s) | Training Loss {:.5f} | Validation Loss {:.5f}'.format(epoch,
                                                                                          epochEnd - epochStart,
                                                                                          avg_train_loss_list[-1],
                                                                                          avg_val_loss_list[-1])

        # Earlystopping checking
        if avg_val_loss_list[-1] < lowest_loss:

            # Update the lowest loss
            lowest_loss = avg_val_loss_list[-1]

            # Restart the epoch counter
            epoch_early_stopping = 1

            # Save model
            torch.save(model.state_dict(),
                       os.path.expanduser(f'{modelPath}/{modelName}.pt'))

            log = log + ' (New best model found)'

        # If patience is surpassed
        elif epoch_early_stopping > patience:

            print(log)
            print('')
            print('{} epochs without loss reduction (Finishing training...)'.format(epoch_early_stopping - 1))
            print('Final loss: {:.5f}'.format(lowest_loss))
            print('')

            # Break training loop
            break

        print(log)
        epoch_early_stopping += 1

    return avg_train_loss_list, avg_val_loss_list

def obtainMask(var, grid = None, path = None, to_slice=None):
    """Function that obtains a mask from dataset to apply to other datasets.

    Args:
        var (string): Variable to use as reference in dataset.
        grid (xarray.object, optional): Dataset to obtain mask from. Defaults to None.
        path (string, optional): Path to load dataset to obtain mask from. Defaults to None.
        to_slice (duple(string, string), optional): Range to slice time in case of being necessary. Defaults to None.

    Returns:
        utils.flattenSpatialGrid: An object of the class 'flattenSpatialGrid'
    """
    if path != None:
        grid = xr.open_dataset(path)
    grid = checkCorrectData(grid) # Transform coordinates and dimensions if necessary
    grid = checkIndex(grid)
    grid=grid.assign_coords({'time': grid.indexes['time'].normalize()})
    if to_slice != None:
        baseMask = flattenSpatialGrid(grid=grid.sel(time=slice(*to_slice)).load(), var=var)
    else:
        baseMask = flattenSpatialGrid(grid=grid.load(), var=var)

    return baseMask

def filterByMask(mask, data, var='tasmean'):
    """TODO

    Args:
        mask (_type_): _description_
        data (_type_): _description_
        var (_type_): _description_

    Returns:
        _type_: _description_
    """
    data_flat = mask.flatten(grid=data, var=var)
    data_flat_array = toArray(data_flat)
    data_flat[var].values = data_flat_array
    data_unflatten = mask.unFlatten(grid=data_flat, var=var)

    return data_unflatten

def __predict(model, device, **kwargs):

    '''
    Internal function to perform inference in pytorch
    '''

    # Prepare model
    model = model.to(device)

    # Move from numpy to torch
    for key, value in kwargs.items():
        kwargs[key] = torch.from_numpy(kwargs[key]).float().to(device)

    # Compute preditions
    model.eval()
    with torch.no_grad():
        y_pred = model(*kwargs.values())

    if device == 'cuda':
        y_pred = y_pred.cpu().numpy()
    else:
        y_pred = y_pred.numpy()

    return y_pred

def predDataset(X, model, device, flattener, var, ref = None):

    '''
    Compute the predictions for a certain dataset and deep learning model

    @X: Numpy array data
    @model: Deep learning model
    @device: Device used to perform inference (cpu or cuda)
    @ref: Dataset used as reference
    @flattener: Flattener used to flatten and remove NANs
    @var: Variable name in the Dataset
    '''
    
    yPred = __predict(model=model, device=device, X=X)

    yPredFlat = flattener.flatten(grid=ref, var=var)
 
    yPredFlat[var].values = yPred
    yPred = flattener.unFlatten(yPredFlat, var)


    return yPred

def getPredictand(data_path, name, var, complete_path = None):
    if complete_path == None:
        file_name = getFileName(data_path, name, keyword = var)
        predictand_path = f'{data_path}{name}/{file_name}'
    else:
        predictand_path = complete_path
    predictand = xr.open_dataset(predictand_path,
                                chunks=-1) # Near surface air temperature (daily mean)
    predictand = checkCorrectData(predictand) # Transform coordinates and dimensions if necessary

    predictand = checkIndex(predictand)
    predictand = checkUnitsTempt(predictand, var)
    predictand = predictand.assign_coords({'time': predictand.indexes['time'].normalize()})

    return predictand

def getPredictors(data_path):
    predictors_vars = ['t500', 't700', 't850', # Air temperature at 500, 700, 850 hPa
    'q500', 'q700', 'q850', # Specific humidity at 500, 700, 850 hPa
    'v500', 'v700', 'v850', # Meridional wind component at 500, 700, 850 hPa
    'u500', 'u700', 'u850', # Zonal wind component at 500, 700, 850 hPa
    'msl']
    data_predictors = []
    for var in predictors_vars:
        data_predictors.append(xr.open_dataset(f'{data_path}/{var}_ERA5.nc'))
    predictors = xr.merge(data_predictors)
    predictors = predictors.reindex(lat=list(reversed(predictors.lat))) 

    return predictors

def getTrainTest(predictors, predictand, years):
    y, x = alignDatasets(grid1=predictand, grid2=predictors, coord='time')

    # Split into train and test set
    yearsTrain = years[0]
    yearsTest = years[1]

    xTrain = predictors.sel(time=slice(*yearsTrain)).load()
    xTest = predictors.sel(time=slice(*yearsTest)).load()

    yTrain = predictand.sel(time=slice(*yearsTrain)).load()
    yTest = predictand.sel(time=slice(*yearsTest)).load()

    return xTrain, xTest, yTrain, yTest

def maskData(var, objective, secondGrid=None, grid = None, path = None, to_slice=None):
    """_summary_

    Args:
        var (_type_): _description_
        objective (_type_): _description_
        secondGrid (_type_, optional): _description_. Defaults to None.
        grid (_type_, optional): _description_. Defaults to None.
        path (_type_, optional): _description_. Defaults to None.
        to_slice (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if path != None:
        grid = xr.open_dataset(path)
    grid = checkCorrectData(grid) # Transform coordinates and dimensions if necessary
    grid = checkIndex(grid)
    grid=grid.assign_coords({'time': grid.indexes['time'].normalize()})
    if to_slice != None:
        baseMask = flattenSpatialGrid(grid=grid.sel(time=slice(*to_slice)).load(), var=var)
    else:
        baseMask = flattenSpatialGrid(grid=grid.load(), var=var)

    objectiveFlat = baseMask.flatten(grid=objective, var=var)
    objectiveFlat_array = toArray(objectiveFlat)
    objectiveFlat[var].values = objectiveFlat_array
    objectiveUnflatten = baseMask.unFlatten(grid=objectiveFlat, var=var)
    
    if np.isnan(objectiveUnflatten).sum() > 0 and secondGrid != None:
        secondMask = obtainMask(grid = secondGrid, var = var)
        secondFlat = secondMask.flatten(grid=objectiveUnflatten, var=var)
        secondFlat_array = toArray(secondFlat)
        secondFlat[var].values = secondFlat_array
        objectiveUnflatten = secondMask.unFlatten(grid=secondFlat, var=var)

    return objectiveUnflatten


def biasYear_x(yTest, yPred, var = None, season_months=None):
    
    if season_months != None:
        yTest = yTest.isel(time=(yTest.time.dt.season == season_months))
        yPred = yPred.isel(time=(yPred.time.dt.season == season_months))
    
    yTest = yTest.groupby('time.year').max('time')
    yPred = yPred.groupby('time.year').max('time')    
    metric = (yPred.mean('year') - yTest.mean('year'))   

    return metric

def getMontlyMetrics(data, to_slice = None):

    if to_slice != None:
        data = data.sel(time=slice(*to_slice))

    data = data.groupby('time.month')
    mean = data.mean()
    std = data.std()

    return {'mean': mean, 'std': std}

def __operationStandarBias(data, hist_mean, hist_std, future_mean, observational_mean, observational_std, mes):

    delta = future_mean - hist_mean
    result = (data - delta - hist_mean)*(observational_std/hist_std) + observational_mean + delta

    return result

def standarBiasCorrection(dataset, hist_metric, future_metric, observational_metric):

    future_mean = future_metric['mean']
    hist_mean = hist_metric['mean']
    observational_mean = observational_metric['mean']
    hist_std = hist_metric['std']
    observational_std = observational_metric['std']

    dataset_corrected = dataset.copy(deep=True)
    for mes in range(1, 13):        
        for var in dataset.keys():
            dataset_corrected[var][dataset_corrected.time.dt.month == mes] = __operationStandarBias(
                dataset.sel(time=dataset.time.dt.month == mes),
                hist_mean.sel(month=mes),
                hist_std.sel(month=mes),
                future_mean.sel(month=mes),
                observational_mean.sel(month=mes),
                observational_std.sel(month=mes),
                mes)[var]

    return dataset_corrected

def scalingDeltaCorrection(grid, refHist, refObs):    
    '''
    Perform a scaling delta mapping following https://gmd.copernicus.org/preprints/gmd-2022-57/    @grid: Dataset to correct (GCM predictors)
    @refHist: Historical predictors (GCM predictors on historial period)
    @refObs: Observational predictors
    '''    
    gridAux = grid.copy(deep=True)
    refHistAux = refHist.copy(deep=True)
    refObsAux = refObs.copy(deep=True)    
    for month in range(1, 12+1):        # Compute monthly means and standard deviations
        refHist_monthMean = refHistAux.sel(time=refHistAux.time.dt.month.isin(month)).mean('time')
        refObs_monthMean = refObsAux.sel(time=refObsAux.time.dt.month.isin(month)).mean('time')
        grid_monthMean = gridAux.sel(time=gridAux.time.dt.month.isin(month)).mean('time')        
        refHist_monthSD = refHistAux.sel(time=refHistAux.time.dt.month.isin(month)).std('time')
        refObs_monthSD = refObsAux.sel(time=refObsAux.time.dt.month.isin(month)).std('time')        # Select data from a specific month
        grid_month = gridAux.sel(time=gridAux.time.dt.month.isin(month))        # Perform the correction
        seasonalDelta = grid_monthMean - refHist_monthMean
        grid_month = ((((grid_month - seasonalDelta -  refHist_monthMean)/refHist_monthSD) * \
                        refObs_monthSD) + refObs_monthMean + seasonalDelta)        # Iterate over vars and assign the value to the grid dataset
        for var in grid.keys():
            gridAux[var][gridAux.time.dt.month.isin(month)] = grid_month[var]    
    
    return gridAux


def multiMapPerSeason(data_to_plot, metrics, plot_metrics, FIGS_PATH, extra_path = '', color_number = None, color_map = None,
                values = {'diff': {'over30': (-50, 50), 'over40': (-10, 10), 'std': (0, 10), 'else': (-3, 3)},
                          'noDiff': {'over30': (0, 500), 'over40': (0, 30), 'std': (0, 1.5), 'else': (-5, 35)}
                }):
    cmap = plt.cm.bwr  # define the colormap
    color_number = cmap.N if color_number == None else color_number
    cmaplist = [cmap(i) for i in range(color_number)]
    cmap = LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, color_number)
    start_time = time.time()
    for graph_type, seasons_value in data_to_plot.items():
        for metric in metrics: 
            #Cambiar a un diccionario TODO
            if graph_type != 'diff':
                if metric == 'over30':
                    v_min = values['noDiff'][metric][0]
                    v_max = values['noDiff'][metric][1]
                elif metric == 'over40':
                    v_min = values['noDiff'][metric][0]
                    v_max = values['noDiff'][metric][1]
                elif metric == 'std':
                    v_min = values['noDiff'][metric][0]
                    v_max = values['noDiff'][metric][1]
                else:
                    v_min = values['noDiff']['else'][0]
                    v_max = values['noDiff']['else'][1]
            else:
                if metric == 'over30' or metric == 'over40':     
                    v_min = values['diff'][metric][0]
                    v_max = values['diff'][metric][1]
                elif metric == 'std':
                    v_min = values['diff'][metric][0]
                    v_max = values['diff'][metric][1]
                else:
                    v_min = values['diff']['else'][0]
                    v_max = values['diff']['else'][1]
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
                        dataToPlot = (data_to_plot[plot_metrics[0]][season_name][predictand_name][metric] - data_to_plot[plot_metrics[1]][season_name][predictand_name][metric])['tasmean']
                    else:
                        dataToPlot = (data_to_plot[plot_metrics[0]][season_name][predictand_name][metric]/data_to_plot[plot_metrics[1]][season_name][predictand_name][metric])['tasmean']

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
            plt.savefig(f'{FIGS_PATH}/comparisson_{metric}_{graph_type}{extra_path}.pdf')
            plt.close()

    total_time = time.time() - start_time
    print(f"El código de graficos de test se ejecutó en {total_time:.2f} segundos.")

def graphsBaseGCM(objective, reference, save_path):
    diff = {}
    for key in objective.keys():
        if key == 'std':
            diff[key] = objective[key] / reference[key]
        else:
            diff[key] = objective[key] - reference[key]

    # Configurar el tamaño de la figura y crear subgráficos con GeoAxes
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(15, 30), subplot_kw={'projection': ccrs.PlateCarree()})

    cmap = (ListedColormap(['royalblue', 'cyan', 'yellow', 'orange'])
            .with_extremes(over='red', under='blue'))

    # Generar y mostrar datos en cada subgráfico
    for i, key in enumerate(objective.keys()):
        for j in range(3):
            if j == 0:
                data = objective[key]['tas']
            elif j == 1:
                data = reference[key]['tas']
            else:
                data = diff[key]['tas']

            if j == 0 and i == 0:
                print(data)
            # Obtener el subgráfico actual
            v_min = data.min()
            v_max = data.max()
            bounds = np.linspace(v_min, v_max, 5)
            ax = axes[i, j]

            # Mostrar la imagen en el subgráfico actual
            im = ax.pcolormesh(data.coords['lon'].values, data.coords['lat'].values, data,
                               transform=ccrs.PlateCarree(), cmap=cmap,
                               norm=BoundaryNorm(bounds, cmap.N), shading='auto')

            # Agregar una barra de color individual
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
            im.set_clim(vmin=v_min, vmax=v_max)

            # Opcional: establecer título, etiquetas, etc.
            ax.set_title(f'Subgráfico {i+1},{j+1}')
            ax.coastlines()  # Añadir líneas de costa para contexto geográfico

    # Ajustar el layout para que no haya solapamientos
    plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.2, hspace=0.4)
    plt.savefig(save_path)
    plt.close()