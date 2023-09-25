#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to process ICESat-2 ATL03 data for snow depths

Created on Thu Aug 10 11:42:10 2023

@author: Désirée Treichler, desiree.treichler@geo.uio.no
"""

import pyarrow.feather as feather
import os

import numpy as np
from pyproj import CRS
import icepyx as ipx
from glob import glob
import matplotlib.pyplot as plt
import h5py
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
import warnings
import datetime

#import socket
import sys
# import custom modules
if os.name == 'posix':
    basepath = '/uio/hypatia/geofag-felles/projects/snowdepth/'
if os.name == 'nt':
    basepath = 'K:/projects/snowdepth/'
mytoolspath = basepath + 'code/tools/own_repos/snowdepth/tools/'
sys.path.append(mytoolspath)
import IC2tools



# %% functions

def downloadATL(dataversion, coords, dates, uid, email, datapath, AOIname):
    """Usage: downloadATL(dataversion, coords, dates, datapath)
    Uses the icepyx library to download ATL03/ATL08 data of the latest version.
    ATLversion: 'ATL03' or 'ATL08'
    coords:     [lonmin, latmin, lonmax, latmax]
    dates:      start and stop dates as list of strings, ['YYYY-mm-dd', 'YYYY-mm-dd']
    uid:        earthdata login user name
    email:      and email
    datapath:   data folder in which a new folder for the current download will be created
    AOIname:    AOI name for the data folder, output format: dataversion_AOIname
        """
    region_S = ipx.Query(dataversion, 
                          np.array(coords), #+np.array([-0.05,-0.05,0.05,0.05]), 
                          dates)
                          #version='005') # specify version if you want to download something else than latest
    region_S.avail_granules(ids=True)         #get a list of granule IDs for the available granules
    region_S.earthdata_login(uid, email =email)
    region_S.download_granules(datapath+dataversion+'_'+AOIname+'/')


def loadATLdata(dataversion, datapath, AOIname,crs=''):
    """ Usage: gdf, ancillary = loadATLdata(dataversion, datapath, AOIname,crs='')
        Input data folder follows the same logic as downloadATL():
        dataversion: 'ATL03' or 'ATL08' - different parameters will be loaded
        datapath:   root folder for data folder
        AOIname:    name of AOI -> h5 files are in datapath/ATL03_AOIname/*.h5
        crs:        projection crs, if specified, the data will be converted and 
                    x / y coordinates added to the gdf. Default is none: crs=''
        Output: geodataframe with point data and attributes, 
                [only for ATL03: ancillary dataframe with parameters per input file]
                    """
    ATL_list = sorted(glob(datapath+dataversion +'_'+AOIname + '/*.h5'))
    datav= int(ATL_list[0][-7])
    
    def get_ancillary_data(fn, ancillary_dict):
        ancillary_data = pd.DataFrame()
        with h5py.File(fn,'r') as h5f:
            for group in ancillary_dict:
                for key in ancillary_dict[group]:
                    DS='%s/%s' % (group, key)
                    temp = np.array(h5f[DS])
                    ancillary_data[key] = temp
        #ancillary_data = pd.DataFrame(ancillary_data)
        return ancillary_data
    
    if dataversion == 'ATL03':
        # dict containing data entries to retrive
        if datav>5:
            dataset_dict = {'heights': ['delta_time', 'lon_ph', 'lat_ph', 'h_ph', 'signal_conf_ph', 'weight_ph'],
                            'geolocation': ['segment_id', 'segment_ph_cnt', 'knn']}
        
            ancillary_dict = {'/ancillary_data/altimetry': ['win_h','win_x','min_knn']}
        else:  # before version 6, some parameters were not available
            dataset_dict = {'heights': ['delta_time', 'lon_ph', 'lat_ph', 'h_ph', 'signal_conf_ph', 'weight_ph'],
                            'geolocation': ['segment_id', 'segment_ph_cnt']}
            ancillary_dict = {}
    
        # Convert the list of hdf5 files into more familiar Pandas Dataframe
        if len(ATL_list)>1:
            gdf_list = [(IC2tools.ATL03_2_gdf(fn, dataset_dict)) for fn in ATL_list] #aoicoords=False, filterbackground=False, utmzone=False, v=False
            ancillary_list = [(get_ancillary_data(fn, ancillary_dict)) for fn in ATL_list] 
    
            # concatenate
            gdf = IC2tools.concat_gdf(gdf_list)
            ancillary = pd.concat([a for a in ancillary_list]).pipe(pd.DataFrame)
        else: 
            gdf = IC2tools.ATL03_2_gdf(ATL_list[0], dataset_dict)
            ancillary = None
    
    elif dataversion =='ATL08':
        dataset_dict = {'land_segments': ['delta_time', 'longitude', 'latitude', 'dem_h', 'dem_flag', 'msw_flag', 'n_seg_ph',
                                          'night_flag', 'rgt', 'segment_id_beg', 'segment_id_end', 'segment_landcover', 'segment_snowcover',  # 'surf_type',
                                          'segment_watercover', 'sigma_h', 'sigma_topo', 'quality', 'terrain_flg'],
                        'land_segments/terrain': ['h_te_best_fit', 'h_te_best_fit_20m', 'h_te_interp', 'h_te_max', 'h_te_mean', 'h_te_median', 'h_te_min',
                                                  'h_te_mode',
                                                  'h_te_skew', 'h_te_std', 'h_te_uncertainty', 'n_te_photons', 'terrain_slope']}
    
        # Convert the list of hdf5 files into Pandas Dataframe
        if len(ATL_list)>1:
            gdf_list = [(IC2tools.ATL08_2_gdf(fn, dataset_dict)) for fn in ATL_list] #aoicoords=False, filterbackground=False, utmzone=False, v=False
            # concatenate
            gdf = IC2tools.concat_gdf(gdf_list)
        else: 
            gdf = IC2tools.ATL08_2_gdf(ATL_list[0], dataset_dict)
    
    # project and add projected coordinates
    if crs != '':
        wgs84 = 'EPSG:4326'
        orig_crs=wgs84
        gdf=gdf.to_crs(crs)
        gdf['x'] = gdf.geometry.x
        gdf['y'] = gdf.geometry.y
        # back to lat/lon
        #gdf=gdf.to_crs(orig_crs)  
        
    # add ID, date 
    gdf['ID'] = np.arange(1, len(gdf) + 1)
    
    deltatime = np.floor(gdf['delta_time'].to_numpy() / 3600/24)
    currentdates, datecounts = np.unique(deltatime, return_counts=True)
    date = [datetime.date(2018, 1, 1)+datetime.timedelta(d) for d in deltatime]
    gdf['date'] = [10000*x.year + 100*x.month + x.day for x in date]
    gdf['month'] = [x.month for x in date]

    #gdf['pb'] = gdf.pair*10+gdf.beam

    # drop any non-numeric columns for a df / easy saving
    #df = pd.DataFrame(gdf.drop(columns={'geometry'}))
    if dataversion == 'ATL03':
        return gdf, ancillary#, df
    else: 
        return gdf


def getUTMcrslatlon(lat,lon): # copy from RGTtools
    """ USAGE: crs = getUTMcrslatlon(lat,lon) - crs in correct UTM zone for data in lat/lon"""
    utm = np.round((183 + lon)/6)
    if utm > 60:
        print('Coordinates need to be in lat/lon.')
        raise Exception(('UTM zone cannot be computed. Already in UTM..?'))
    if lat<0: # south
       crs = CRS.from_string('+proj=utm +zone='+str(int(utm))+'+south')
    else:
       crs = CRS.from_string('+proj=utm +zone='+str(int(utm))) 
    return crs


def llAOI2bb(lats, lons, utm=True, outfile=''):
    """ Usage: clipshp = llAOI2bb(lats, lons, utm=True, outfile='') 
    lats: min/max lat
    lons: min/max lon
    utm: convert to utm or not (from wgs84)
    outfile: if a file path is provided, the aoi polygon is written to disk as shp"""
    
    wgs84 = 'EPSG:4326'
    coords = [lons[0], lats[0],lons[1], lats[1]]

    clipshp_S_ll = IC2tools.makebboxgdf(
        coords[0], coords[1], coords[2], coords[3])
    clipshp_S_ll.crs = wgs84
    clipshp_S_utm = clipshp_S_ll.to_crs(getUTMcrslatlon(np.mean(lats),np.mean(lons)))
    if utm:
        clipshp=clipshp_S_utm
    else: 
        clipshp=clipshp_S_ll
    if (outfile=='')==False:
        clipshp.to_file(outfile)
    return clipshp


def addATL08class(gdf, gdf08):
    
    # We first create a ranges variable that contains all the segment indices for each coarse resolution range using numpy's arange function. We then use np.repeat to replicate the start and end indices for each segment in the ranges variable, and create a dataframe df_map that maps each segment index to the corresponding start and end indices.
    ranges = np.concatenate([np.arange(s, e+1)[:,None] for s, e in zip(gdf08['segment_id_beg'], gdf08['segment_id_end'])])
    df_map = pd.DataFrame({'segment_id': ranges.ravel(), 
            'segment_id_beg': np.repeat(gdf08['segment_id_beg'], gdf08['segment_id_end']-gdf08['segment_id_beg']+1), 
            'segment_id_end': np.repeat(gdf08['segment_id_end'], gdf08['segment_id_end']-gdf08['segment_id_beg']+1),
            'pb': np.repeat(gdf08['pb'], gdf08['segment_id_end']-gdf08['segment_id_beg']+1),
            'date': np.repeat(gdf08['date'], gdf08['segment_id_end']-gdf08['segment_id_beg']+1)})

    # merge the mapped dataframe with the original dataframe to add the segment id sections to the first dataframe
    df_merged = pd.merge(gdf, df_map, on=['segment_id', 'pb','date'], how='left')
     
    gdf08_selected = gdf08[['segment_id_beg','segment_id_end', 'pb','date','RGT','cycle','segment_landcover', 'segment_snowcover', 'sigma_h',
                'sigma_topo', 'terrain_flg', 'h_te_best_fit', 'h_te_interp','h_te_mean',
                'h_te_median','h_te_std','h_te_uncertainty', 'n_seg_ph','n_te_photons',# 'n_ca_photons','centroid_height', 'photon_rate_can',,  'h_canopy_abs'
                'terrain_slope']].copy()
    #gdf08_selected= gdf08_selected.rename(columns={'dateint':'date'})
  
    # merge with the coarse-resolution dataframe to add the coarse info 
    df_final = pd.merge(df_merged, gdf08_selected, on=['segment_id_beg', 'segment_id_end', 'pb','date'], how='left')
    return df_final


def strongorweak(gdf):
    # add strong/weak beam identifier (based on which beam of a pair has more photons per segment)
    gdf['strongbeam']=0
    dates, counts=np.unique(gdf.date,return_counts=True) # 
    for d in dates: 
        for pb in [10,20,30]: 
            currsubsetI=(gdf['date'] == d) & (gdf['pb'] == pb)   #(np.isnan(gdf['DEM05sbcoreg_all'])) 
            currsubsetJ=(gdf['date'] == d) & (gdf['pb'] == pb+1)   #(np.isnan(gdf['DEM05sbcoreg_all'])) 
    
            I = np.nanmean(gdf.loc[currsubsetI,'n_seg_ph'])
            J = np.nanmean(gdf.loc[currsubsetJ,'n_seg_ph'])
            
            if I>J:
                gdf.loc[currsubsetI,'strongbeam']=1
                if I<(2*J):
                    print(f'difference is little: {I} vs {J} for {d}, {pb}/{pb+1}')
            else:
                gdf.loc[currsubsetJ,'strongbeam']=1
                if J<(2*I):
                    print(f'difference is little: {I} vs {J} for {d}, {pb}/{pb+1}')
    return gdf



def coregpts(gdf, DEM, outpath=''):
    # make a xdem-compatible dataframe. It requires lat, lon and z columns. 
    gdf_coreg = gdf.rename(columns={'x':'lon','y':'lat'})
    gdf_coreg['z']= gdf_coreg['h_ph']-gdf_coreg['geoid']
    
    # make coreg object
    nuth_kaab = xdem.coreg.NuthKaab()
    nuth_kaab.fit_pts(gdf_coreg, DEM, verbose=True)
    #print(nuth_kaab._meta) # in pixel
    coregcoeff = nuth_kaab.to_matrix()

    # save the coreg shift info
    if ~(outpath==''):
        IC2tools.savecoreginfo(outpath, nuth_kaab)
        
    # store the coregistration info in the gdf    
    gdf_coreg['DEMh_coreg'] = DEM.interp_points(np.vstack([ \
                                np.array(gdf_coreg['lon'].values +coregcoeff[0,3]),\
                                np.array(gdf_coreg['lat'].values + coregcoeff[1,3])]).T, input_latlon=False, order=1)                 
    # return the z shift also added as a constant to the gdf?
    #gdf_coreg['zshift'] = coregcoeff[2,3]
    
    # rename columns back to original
    gdf = gdf_coreg.rename(columns={'lon':'x','lat':'y'}).drop(columns={'z'})
 
    return gdf, coregcoeff


def coregpts_newversion(gdf, DEM, outpath=''):
    # make a xdem-compatible dataframe. It requires lat, lon and z columns. 
    gdf_coreg = gdf.rename(columns={'x':'E','y':'N'})
    gdf_coreg['z']= gdf_coreg['h_ph']-gdf_coreg['geoid']
    
    # make coreg object
    nuth_kaab = xdem.coreg.NuthKaab()
    nuth_kaab.fit_pts(gdf_coreg, DEM, verbose=True)
    #print(nuth_kaab._meta) # in pixel
    coregcoeff = nuth_kaab.to_matrix()

    # save the coreg shift info
    if ~(outpath==''):
        IC2tools.savecoreginfo(outpath, nuth_kaab)
        
    # store the coregistration info in the gdf    
    gdf_coreg['DEMh_coreg'] = DEM.interp_points(np.vstack([ \
                                np.array(gdf_coreg['E'].values +coregcoeff[0,3]),\
                                np.array(gdf_coreg['N'].values + coregcoeff[1,3])]).T, input_latlon=False, order=1)                 
    # return the z shift also added as a constant to the gdf?
    #gdf_coreg['zshift'] = coregcoeff[2,3]
    
    # rename columns back to original
    gdf = gdf_coreg.rename(columns={'E':'x','N':'y'}).drop(columns={'z'})
 
    return gdf, coregcoeff



# def ctxmap(ax, crs, source=ctx.providers.Stamen.Terrain):
#     try:
#         ctx.add_basemap(ax=ax, crs=crs, source=source)
#     except:
#         wgs84 = 'EPSG:4326'
#         xlim = ax.get_xlim()
#         ylim = ax.get_ylim()
#         dx = xlim[1]-xlim[0]
#         dy = ylim[1]-ylim[0]
#         if dy > (dx*3):
#             xlim = [xlim[0]-dx/3, xlim[1]+dx/3]
#             ax.set_xlim(xlim)
#         transformer = Transformer.from_crs(crs, wgs84)
#         xlimll, ylimll = transformer.transform(xlim, ylim)
#         zoom = ctx.tile._calculate_zoom(
#             xlimll[0], ylimll[0], xlimll[1], ylimll[1])  # (w, s, e, n)
#         e = True
#         while (e == True) & (zoom > 0):
#             try:  # try to increase the zoom level until it works
#                 ctx.add_basemap(ax=ax, crs=crs, zoom=zoom, source=source)
#                 e = False
#             except:
#                 zoom = zoom-1


# XXX get rid of the warning this produces
def interpolateATL03_alongtrack(gdf, subset):
    # first, interpolate to get placeholders for gaps in the original data
    gdf_new = pd.DataFrame(gdf[subset].copy().drop(columns={'geometry'}))
    # add a timestamp index for resampling missing laser shots
    gdf_new['Timestamp']=[pd.to_datetime(datetime.datetime(2018,1,1,0,0,0)+datetime.timedelta(d)) for d in gdf_new['delta_time'].to_numpy()/3600/24]
    gdf_new.set_index("Timestamp", inplace=True)  
    # process each beam individually
    gdf_resampled= gpd.GeoDataFrame()
    for pb in np.unique(gdf_new.pb):
        gdf_subset = gdf_new[gdf_new.pb==pb]
        # resample to the original laser shot frequency (mean location per footprint returns) and linarly interpolate missing footprints
        gdf_subset=gdf_subset.resample('0.1ms').mean().interpolate(method='linear')
        # collect the different beams in one dataframe
        gdf_resampled=pd.concat([gdf_resampled, gdf_subset])
    
    # back to range index
    gdf_resampled=gdf_resampled.reset_index(drop=True)
    
    # select only those that are missing in the original dataframe
    add_to_orig = [((x in gdf.delta_time.values)==False) for x in gdf_resampled.delta_time.values]
    gdf_add_to_orig=gdf_resampled.loc[add_to_orig,['x','lon_ph','y','lat_ph','delta_time','date','pb']]
    # and add them back
    gdf_combined=pd.concat([gdf.reset_index(drop=True), gdf_add_to_orig.reset_index(drop=True)])
    gdf_combined.crs=gdf.crs
    gdf_combined=gdf_combined.reset_index(drop=True)

    return gdf_combined



def atl03snowdepths(gdf, DEMh_col, uav_col, dhcutoff, window_m, sd_col='sd_atl03', sd_uav = 'sd_uav'):
    """ usage: gdf, SDdict = atl03snowdepths(gdf, DEMh_col, uav_col, dhcutoff, 
    window_m, sd_col='sd_atl03', sd_uav = 'sd_uav')
    Move along tracks, filter, average and resample ICESat-2 snow depths.
     """ 
    
    # compute unfiltered snow depths 
    gdf.loc[:,sd_col] = gdf.h_ph - gdf[DEMh_col] - gdf.geoid #+ currzshift  # REMOVE the z shift from the DEM - add it to the differences
    gdf.loc[:,sd_uav] = gdf[uav_col] - gdf[DEMh_col] - gdf.geoid #+ currzshift  # REMOVE the z shift from the DEM - add it to the differences
    
    # get things right according to input
    # 1ms corresponds to 7m, i.e. 10 bursts (of 0-8 photons each)
    windowlength = str(window_m)+'m'
    
    windowsize = str(window_m/7)+'ms'
        
    SDdict = {}
    
    for pb in np.unique(gdf.pb):   
        # current beam
        I=(gdf.pb==pb) & ~np.isnan(gdf[DEMh_col])
        # symmetric dh cutoff boundaries
        medsd = np.nanmedian(gdf.loc[I,sd_col])
        # only use those data points that are within the dh cutoff boundaries:
            # but also include points where h_ph is nan, as we have uav/DEM data in these holes
        I=(gdf.pb==pb) & ~np.isnan(gdf[DEMh_col]) & ( ((gdf[sd_col]<(dhcutoff+medsd)) &  (gdf[sd_col]>(medsd-dhcutoff))) | np.isnan(gdf[sd_col]))
        # plt.plot(gdf.loc[I,'y'],gdf.loc[I,'sd_orig'])
        # current beam with selected photons as dataframe (without geometry)
        beam = gdf.loc[I].copy().drop(columns={'geometry'})
        
        
        if len(beam) < 20:  # don't bother if there are barely any points
            print(f'not enought data points in beam {pb}')
            continue
    
        
        # add a timestamp index for rolling time filter window
        beam['Timestamp']=[pd.to_datetime(datetime.datetime(2018,1,1,0,0,0)+datetime.timedelta(d)) for d in beam['delta_time'].to_numpy()/3600/24]
        beam.set_index("Timestamp", inplace=True)    
    
        
        # rolling window stats
        # -----------------------
        beam.loc[:, 'sd_med'] = beam[sd_col].rolling(window=windowsize, center=True).median()
        beam.loc[:, 'sd_mean'] = beam[sd_col].rolling(window=windowsize, center=True).mean()
        beam.loc[:, 'sd_std'] = beam[sd_col].rolling(window=windowsize, center=True).std()
        beam.loc[:, 'sd_sem'] = beam[sd_col].rolling( window=windowsize, center=True).sem()
    
        # also add to gdf[I]
        for col in ['sd_med','sd_mean','sd_std','sd_sem']:
            gdf.loc[I, col] = beam.loc[:, col].values
        
        # resampling stats
        # -----------------------
        sdn = beam.resample(windowsize).size() # how many samples
        sdn_with_h_ph = beam[~np.isnan(beam.h_ph)].resample(windowsize).size() # how many samples
    
        
        if len(sdn_with_h_ph) <1:  # MOVE ON IF THERE ARE NO POINTS
            print(f'not enought data points in beam {pb}')
            continue
        
        sdmed = beam.resample(windowsize).median()
        sdmean = beam.resample(windowsize).mean()
        sdstd = beam.resample(windowsize).std()
        sdsem= beam.resample(windowsize).sem()
    
    
    
        # collect the relevant data in a new resampled dataframe
        #---------------------------
        # first, location/beam identifier parameters
        SD= pd.DataFrame()
        #SD['Timestamp']=sdmed.index
        for col in ['date','pb','month','lat_ph','lon_ph','x','y',
                    'strongbeam','segment_snowcover','segment_landcover','terrain_slope',
                    'h_ph','geoid',DEMh_col,uav_col]:
            SD[col]=sdmean[col]
        
        # number of contributing samples
        SD['n_with_nans']= sdn
        SD['n']= sdn_with_h_ph
    
        # median snow depths, with and without rolling median filtering
        SD['sd_med']=sdmed['sd_med']
        SD['sd_med_nonfiltered']=sdmed[sd_col]
        
        # mean, with and without rolling median filtering
        SD['sd_mean']=sdmean['sd_med']
        SD['sd_mean_nonfiltered']=sdmean[sd_col]
    
        # error measurements: SEM and std of rolling median, median/mean of rolling std
        SD['sd_sem']=sdsem['sd_med']
        SD['sd_std_of_rolling_median']=sdstd['sd_med']
        SD['sd_mean_rolling_std']=sdmean['sd_std']
        SD['sd_med_rolling_std']=sdmed['sd_std']
        
        # corresponding uav snow depths (no rolling median applied)
        SD['sd_uav_med']=sdmed[sd_uav]
        SD['sd_uav_mean']=sdmean[sd_uav]
        SD['sd_uav_std']=sdstd[sd_uav]
    
    
        # add the resulting data frame for this beam to a dict 
        SDdict[pb]=SD   
        
    return gdf, SDdict


def plotsnowtransect(beam, SD, snowax='', demax='', refsd=True, sd_mean_col='sd_med',sd_std_col='sd_med_rolling_std', sd_col='sd_atl03',sd_uav='sd_uav',uav_col='uav',DEMh_col='DEMh_orig'):    
    if 'med' in sd_mean_col:
        refsd_mean_col = 'sd_uav_med'
    else:         
        refsd_mean_col = 'sd_uav_mean'

    if snowax == '':
        fig2, ax = plt.subplots(2, 1, figsize=(15, 15), sharex=True) # transects
        snowax = ax[1]
        demax = ax[0]

    # plot settings: colors
    #snowlims=[-0.5, 3] # ylim for the snow depth transects
    icptc=[0.8,0.5,0.2]  # icesat points
    #icmedcroll = [0.7,0.4,0.1] # icesat median
    icmedc = [0.6,0.2,0.1] # icesat median
    #icmedcref = [0.9,0.4,0.3] # icesat median
    icstdc = [0.8,0.8,0.6] # icesat std fill
    lidarptc=[0,1,1]        # drone point cloud 
    lidarmedc = [0,0.7,0.7] # drone median
    minsdy = min(np.nanpercentile(beam['sd_med'],(2)),0)-1
    maxsdy = max(np.nanpercentile(beam['sd_med'],(98)),3)+1
        
    # plot the elevation transect if demax was provided or created above
    if ~(demax==''):
        # DEM elevation plot
        ### uav
        ### photons
        demax.scatter(beam.y/1000, beam['h_ph']-beam['geoid'], s=0.2, color=icptc, label='ATL03 photons')
        #try: ax[0].plot(curr1.y/1000, curr1['DEMh'],'k-', markersize=0.1, lw=0.5, label='DEM - no coreg')
        if refsd:
            demax.plot(beam.y/1000, beam[uav_col]-beam['geoid'], '.', color=lidarptc,markersize=0.1, lw=0.5, label='UAV elevations')       

        ### DEM
        demax.plot(beam.y/1000, beam[DEMh_col], '-', markersize=0.1, lw=0.5, label=DEMh_col)       
        demax.set_ylabel('Elevation masl [m]')
        demax.legend()
        miny = np.nanmin(beam[DEMh_col])-5
        maxy = np.nanmax(beam[DEMh_col])+10
        demax.set_ylim([miny, maxy])
        demax.set_title(f"Elevation transect", fontsize='medium')  # verticalalignment: va
        demax.set_xlabel('Northing [km]')

        
        
    # snow depths 
    ###
    # original photons
    snowax.scatter(beam.y/1000, beam[sd_col], s=0.15, color=icptc, label='ATL03 photons')#, label=currcol+' - photons')
    if refsd: # reference snow depth measurements for each photon
        snowax.scatter(beam.y/1000, beam[sd_uav], s=0.15, color=lidarptc, label='UAV')#, label=currcol+' - photons')
        
    # check if standard deviation is at photon or resampled level
    if sd_std_col in ['sd_std', 'sd_sem']: # photon level, very nervous plot
        snowax.fill_between(beam.y/1000,beam[sd_mean_col]-beam[sd_std_col], 
                           beam[sd_mean_col]+beam[sd_std_col],alpha=0.5, edgecolor=None, 
                           facecolor=icstdc, label='stdev range', zorder=0)
    else: 
        snowax.fill_between(SD.y/1000,SD[sd_mean_col]-SD[sd_std_col], 
                           SD[sd_mean_col]+SD[sd_std_col],alpha=0.5, edgecolor=None, 
                           facecolor=icstdc, label='stdev range', zorder=0)

    notresampled=True
    if notresampled: # median, not resampled. sd from both currcol and without coregistration:
        snowax.plot(beam.y/1000, beam['sd_med'], '.-',markersize=0.1, color=icptc,
                        label=f"ATL03 rolling median")


    # averaged and resampled snow depths
    try:
        snowax.plot(SD.y/1000, SD[sd_mean_col], '.-',color=icmedc,
                   markersize=0.5, lw=1, label=f"ATL03 resampled")
#            ax[nn+1].plot(currsdvmed.y/1000, currsdvmed['sd_med_nocoreg'], '--',color=icmedcref,
#                       markersize=0.5, lw=1, label=f"median ({windowlength}) - no coreg")
        if refsd:
            snowax.plot(SD.y/1000, SD[refsd_mean_col], '.-',markersize=0.1, color=lidarmedc,
                           label=f"UAV resampled")
    except: pass

        
    snowax.axhline(0,lw=1,c='grey') # the ground
    snowax.set_title('Snow depths from ICESat-2 and UAV')
    snowax.set_ylim([minsdy, maxsdy])
    snowax.legend()
    snowax.set_ylabel('Snow depth [m]')

    snowax.set_xlabel('Northing [km]')
        
    return
        










#%% unfinished functions


# def load08data(datapath, AOIname,crs=''):
#     """ Usage: ggdf_08 = load08data(datapath, AOIname,crs='')
#         Input data folder follows the same logic as downloadATL():
#         datapath:   root folder for data folder
#         AOIname:    name of AOI -> h5 files are in datapath/ATL08_AOIname/*.h5
#         crs:        projection crs, if specified, the data will be converted and 
#                     x / y coordinates added to the ggdf. Default is none: crs=''
#         Output: geodataframe with point data and attributes
#                     """
#     ATL08_list = sorted(glob(datapath+'ATL08_' +AOIname + '/*.h5'))
#     # dict containing data entries to retrive
#     dataset_dict = {'land_segments': ['delta_time', 'longitude', 'latitude', 'dem_h', 'dem_flag', 'msw_flag', 'n_seg_ph',
#                                       'night_flag', 'rgt', 'segment_id_beg', 'segment_id_end', 'segment_landcover', 'segment_snowcover',  # 'surf_type',
#                                       'segment_watercover', 'sigma_h', 'sigma_topo', 'quality', 'terrain_flg'],
#                     'land_segments/terrain': ['h_te_best_fit', 'h_te_best_fit_20m', 'h_te_interp', 'h_te_max', 'h_te_mean', 'h_te_median', 'h_te_min',
#                                               'h_te_mode',
#                                               'h_te_skew', 'h_te_std', 'h_te_uncertainty', 'n_te_photons', 'terrain_slope']}

#     # Convert the list of hgdf5 files into Pandas Dataframe
#     if len(ATL08_list)>1:
#         gdf_list = [(IC2tools.ATL08_2_gdf(fn, dataset_dict)) for fn in ATL08_list] #aoicoords=False, filterbackground=False, utmzone=False, v=False
#         # concatenate
#         gdf_08 = IC2tools.concat_gdf(gdf_list)
#     else: 
#         gdf_08 = IC2tools.ATL08_2_gdf(ATL08_list[0], dataset_dict)
    
#     # project and add projected coordinates
#     if crs != '':
#         wgs84 = 'EPSG:4326'
#         orig_crs=wgs84
#         gdf_08=gdf_08.to_crs(crs)
#         gdf_08['x'] = gdf_08.geometry.x
#         gdf_08['y'] = gdf_08.geometry.y
    
#     # add ID, date 
#     gdf_08['ID'] = np.arange(1, len(gdf_08) + 1)
    
#     deltatime08 = np.floor(gdf_08['delta_time'].to_numpy() / 3600/24)
#     currentdates08, datecounts08 = np.unique(deltatime08, return_counts=True)
#     date08 = [datetime.date(2018, 1, 1)+datetime.timedelta(d) for d in deltatime08]
#     gdf_08['date'] = [10000*x.year + 100*x.month + x.day for x in date08]
#     gdf_08['month'] = [x.month for x in date08]

#     return gdf_08#, df_08
    



# coords_S = [-0.53, 42.659, -0.17, 42.84]
# #In order to facilitate other formats, the sliderule.toregion function can be used to convert polygons from the GeoJSON and Shapefile formats to the format accepted by SlideRule.
# poly=[{'lon': -0.53, 'lat': 42.659},
#  {'lon': -0.17, 'lat': 42.659},
#  {'lon': -0.17, 'lat': 42.84},
#  {'lon': -0.53, 'lat': 42.84},
#  {'lon': -0.53, 'lat': 42.659}]



# def get_sliderule_yapc(poly, time_start, time_end):
    
#     # turn off warnings for demo
#     #warnings.filterwarnings('ignore')
    
#     url= "icesat2sliderule.org"
#     url = "slideruelearth.io"
    
#     icesat2.init(sliderule.service_url,verbose=False)

#     wgs84 = 'EPSG:4326'
#     #utm30n = 'EPSG:25830'

#     asset = 'atlas-s3'

#     resources = icesat2.cmr(short_name='ATL03', polygon=poly, time_start='2022-06-01',
#         time_end='2022-08-20', asset=asset) 

#     parms = {
#         "poly": poly,
#         "srt": icesat2.SRT_LAND,
#         #minimum confidence level
#         "cnf": 1, # tutorial: -2 - all photons
#         # "pass_invalid": True, # tutorial settings, to get all photons
#         "quality_ph":0, # marco's settings
        
#         #uncomment next line if you want only ground classified photons
#         #"atl08_class":"atl08_ground",
        
        
#         "yapc":{#filter parameters - # marco's settings
#                 "score":0,
#                 "knn": 0,
#                 "min_ph":4, # 0
#                 #parameters for the yapc score window (where to look for neighboors)
#                 "win_h":6, #3
#                 "win_x":11 #5
#                 },
#         #filter parameters
#         "cnt": 3,
#         #definition of a segment
#         "len": 5.0, # tutorial: 20
#         "res": 5.0, 
#         "ats":1,
#     }

#     """rqst={"atl03-asset":"atlas-s3",
#          "resource":resources,
#          "params":parms}"""
#     atl03 = icesat2.atl03sp(parms)



#     atl03= atl03.to_crs(epsg='25830')
#     atl03['x'] = atl03.geometry.x
#     atl03['y'] = atl03.geometry.y

#     atl03 = (atl03.drop(columns={'geometry'}))

#     with open('ATL03_yapc.feather', 'wb') as f:
#         feather.write_feather(atl03, f)
