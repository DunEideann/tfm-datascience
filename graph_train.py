import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from lib import utils, models, data
import time
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap

FIGS_PATH = '/lustre/gmeteo/WORK/reyess/figs/'
PREDS_PATH = '/lustre/gmeteo/WORK/reyess/preds/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'
MODELS_PATH = '/oceano/gmeteo/users/reyess/tfm/official-code/models'

predictands = ['E-OBS', 'AEMET_0.25deg', 'Iberia01_v1.0', 'pti-grid', 'CHELSA', 'ERA5-Land0.25deg']

yPred = {}
yRealTest = {}
# TODO Pasar a archivo unico cosas repetidas
yearsTrain = ('1980-01-01', '2003-12-31')
yearsTest = ('2004-01-01', '2015-12-31')