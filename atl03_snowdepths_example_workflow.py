#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 19:27:20 2023

@author: desireet
"""

import os

import numpy as np
#from pyproj import CRS
#import icepyx as ipx
import matplotlib.pyplot as plt
#import h5py
import pandas as pd
import geopandas as gpd
import xdem
# maps
#import contextily as ctx
#from pyproj import Transformer


#import sliderule
#from sliderule import icesat2,  io #ipysliderule,
#import ipywidgets as widgets
#import logging
#import warnings
#import datetime

#import socket
import sys
# import custom modules
if os.name == 'posix':
    basepath = '/uio/hypatia/geofag-felles/projects/snowdepth/'
if os.name == 'nt':
    basepath = 'K:/projects/snowdepth/'

# XXX currently still needed, TODO: merge functions from IC2tools into the atl03_functions
mytoolspath = basepath + 'code/tools/own_repos/snowdepth/tools/'
sys.path.append(mytoolspath)
import IC2tools

atl03functionspath = basepath+ 'desireet/atl03_snowdepth_paper/git_code/'
sys.path.append(atl03functionspath)
from atl03_functions import downloadATL, loadATLdata, getUTMcrslatlon, llAOI2bb
from atl03_functions import addATL08class, strongorweak, coregpts, coregpts_newversion
from atl03_functions import interpolateATL03_alongtrack, atl03snowdepths, plotsnowtransect


#%% stuff that should go outside the functions eventually

wd = basepath+'desireet/atl03_snowdepth_paper/'
os.chdir(wd)
datapath = basepath+'data/ICESat-2/granules/'
wgs84 = 'EPSG:4326'
# ic2crs= 'EPSG:4326' # https://nsidc.org/sites/nsidc.org/files/ATL03-V005-UserGuide_1.pdf
utm32n = 'EPSG:32632'
utm33n = 'EPSG:32633' # the Norwegian DEMs are in this projection
etrs89 = 'EPSG:4937'


#%% settings for AOI HW west
# --------
AOIname = 'Dyranut'
lats = [60.25, 60.5]
lons= [7.4, 7.65]
AOI = [417483.6, 6693046.9, 419957.1,6697172.9]
datestart='2022-05-05'
datestop = '2022-05-08'

geoidfile = basepath+'data/DEM/Norway/no_kv_HREF2018B_NN2000_EUREF89.tif'
demfile= basepath+'fieldwork/Hardangervidda_2022_winter/dtm1/data/dtm1_33_112_119_UTM32N.tif'
roadshp = basepath+'fieldwork/Hardangervidda_2022_winter/data_analysis/gis_data/HW_road/HW_road.shp'

uavfile = basepath+'fieldwork/Hardangervidda_2022_winter/data/processed_2022-05-16_RGT671-694_Hardanger_west/2022-05-16_RGT671-694_Hardanger_DEM_geotif_AGISOFT_1m.tif'
uavfileY= basepath+'fieldwork/Hardangervidda_2022_winter/data/processed_2022-05-16_RGT671-694_Hardanger_west/processedByYvesBuehler/DSM_Hardanger_West_1m_UTM32N.tif'


#%% run  processing for this AOI
# -------
coords = [lons[0], lats[0],lons[1], lats[1]]
crs = getUTMcrslatlon(np.mean(lats),np.mean(lons))
dates = [datestart, datestop]
email = '' # set your earthdata email!
uid = '' # set your earthdata user name!


# ATL03
#---
# download ATL03 (standard data, not sliderule)
downloadATL('ATL03',coords, dates, uid, email, datapath, AOIname)


# load into a geodataframe
if AOIname=='Dyranut':  # v6 version of the Dyranut data somehow turns out corrupted...
    gdf, ancillary=loadATLdata('ATL03',datapath, 'Dyranut_v5',crs)
else:
    gdf, ancillary =loadATLdata('ATL03', datapath, AOIname,crs) # does not include all data for Dyranut??


# drop data according to photon quality classification and outside the AOI
# check if the area has land ice quality flags, if not, use the standard signal confidence flag
if sum(np.isnan(gdf.signal_conf_ph_landice)) > 0.9*len(gdf) or sum(gdf.signal_conf_ph_landice) < -0.9*len(gdf):  
    gdf = gdf[(gdf.signal_conf_ph>1)] 
else: 
    gdf = gdf[(gdf.signal_conf_ph_landice>1)] 



# ATL08
#---
# download and load ATL08 data
downloadATL('ATL08',coords, dates, uid, email, datapath, AOIname)
gdf08 =loadATLdata('ATL08', datapath, AOIname,crs)

# save to shp to view in GIS
gdf08.to_file(basepath+'fieldwork/Hardangervidda_2022_winter/data/processed_granules/ATL08_Dyranut_20220507.shp')



# merge the two and do some filtering etc
#---
# add some ATL08 info to ATL03
gdf=addATL08class(gdf, gdf08)
gdf=strongorweak(gdf)


# in case there are gaps in the photon data along the transect:
# add dummy footprints (interpolate gaps within the subset) where we still want to have DEM/UAV elevations
subset= (gdf.x>=AOI[0]) & (gdf.x<=AOI[2]) & (gdf.y>=AOI[1]) & (gdf.y<=AOI[3])
gdf= interpolateATL03_alongtrack(gdf, subset)
gdf=gdf.sort_values('delta_time')

# re-define the subset XXX not the most elegant way to do this... 
subset= (gdf.x>=AOI[0]) & (gdf.x<=AOI[2]) & (gdf.y>=AOI[1]) & (gdf.y<=AOI[3])
origpts= (gdf.ID>0) 
    


# load dems and add data to gdf
#---
geoid = xdem.DEM(geoidfile)
DEM = xdem.DEM(demfile)


# add DEM data to the ATL03 gdf: get geoid undulation and DEM elevation for each point
gdf['geoid'] = geoid.interp_points(np.array((gdf['lon_ph'].values, gdf['lat_ph'].values)).T, input_latlon=True, order=1) 
gdf['DEMh_orig'] = DEM.interp_points(np.array((gdf['x'].values, gdf['y'].values)).T, input_latlon=False, order=1) 

# set unreasonable values to nans
gdf.loc[gdf['DEMh_orig'] <0,'DEMh_orig']=np.nan


## find which points lie on the road - use this to snap/to check coregistration quality
road = gpd.read_file(roadshp).to_crs(crs)
road_buffer = road.buffer(5)

#road_buffer.plot() # check how that looks like
gdf['onroad']=gdf.geometry.within(road_buffer.iloc[0])



# coregistration
#---
outpath=wd+f'/output/{AOIname}/coreg_'+AOIname
gdf, coregcoeff = coregpts(gdf, DEM, outpath) # my own old xdem version adapted for point clouds
#gdf, coregcoeff = coregpts_newversion(gdf, DEM, outpath) # the new official 
# xdem version that takes point clouds - XXX does not yet work!

# XXX note that for some areas, I needed to coregister beams individually as 
# there seemed to be a shift between the strong and weak beam 
# (they are not acquired at the same time)



# load UAV dem(s) - this is assuming the UAV data as interpolated raster data, not point clouds
# ---
uav = xdem.DEM(uavfile)
uavY = xdem.DEM(uavfileY)

# get data
gdf['uav'] = uav.interp_points(np.array((gdf['x'].values, gdf['y'].values)).T, input_latlon=False, order=1) 
gdf.loc[gdf.uav<100,'uav']=np.nan # remove no data values



# plot all - XXX add uav/extent
# ---
# create a hillshade from the DEM
hs = xdem.terrain.hillshade(DEM) # very slow for big DEMs!
    
# plot
fig, ax = plt.subplots(figsize=(5,10))
clim = np.nanpercentile(gdf['h_ph'].values, (2, 98))
try:     # this works with the newest official version of xdem
    hsplot=hs.show(ax=ax, cmap='Greys_r',add_cbar=False)#, )
except:     
    # this works for my old version of xdem
    hsplot=hs.show(ax=ax, cmap='Greys_r')#, )
    hsplot[1].remove() 

road.plot(ax=ax)
# plot only the subset and original points (not the interpolated ones, in case there were gaps)
sc=ax.scatter(gdf[subset&origpts].x,gdf[subset&origpts].y, s=0.5, c=gdf[subset&origpts].h_ph)

plt.colorbar(sc)
ax.set_xlim([AOI[0],AOI[2]])
ax.set_ylim([AOI[1],AOI[3]])

fig.suptitle('ICESat-2 ATL03 data in '+AOIname)
plt.savefig(wd+f'output/{AOIname}/{AOIname}_ATL03map'+'_h_ph.png')



# get ICESat-2 snowdepths: running median,  loop through individual beams:
#--------------
DEMh_col = 'DEMh_orig'
uav_col = 'uav'
dhcutoff = 15
window_m=7 # 7m = 10 shots: original shot frequency 0.1ms corresponds to 0.7m

# names for new columns with unfiltered snow depths
sd_col = 'sd_atl03'
sd_uav = 'sd_uav'
gdf, SDdict = atl03snowdepths(gdf, DEMh_col, uav_col, dhcutoff, window_m, sd_col, sd_uav)


    
# plot the result as transects, and analyse them
#--------------------    
for pb in np.unique(gdf.pb):
    beam = gdf[(gdf.pb==pb)&subset]
    SD = SDdict[pb]
    SDsubset = (SD.x>=AOI[0]) & (SD.x<=AOI[2]) & (SD.y>=AOI[1]) & (SD.y<=AOI[3])
    SD=SD[SDsubset]
    

    if len(beam)>100:
        fig2, ax = plt.subplots(2, 1, figsize=(15, 15), sharex=True) # transects
        snowax = ax[1]
        demax = ax[0]
        plotsnowtransect(beam, SD, snowax=snowax, demax=demax, refsd=True, sd_mean_col='sd_med',sd_std_col='sd_med_rolling_std')  
        
        plt.savefig(wd+f'output/{AOIname}/{AOIname}_sdtransect_window{window_m}m_dhcutoff{dhcutoff}m.png')

# XXX analysis function still in the making... 
