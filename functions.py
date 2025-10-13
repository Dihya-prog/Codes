#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 15:49:42 2025

@author: dihyachaal
"""


import numpy as np
import xarray as xr
import os
from scipy.signal import butter, filtfilt
from pyproj import Geod


def _normalized_ds(ds, lon_min, lon_max):
    lon = ds.longitude.values
    lon[lon < lon_min] += 360
    lon[lon > lon_max] -= 360
    ds.longitude.values = lon
    return ds
    
def _subset_ds(file, variables, lon_range, lat_range, output_dir):
    #print(f"Subsetting dataset: {file}")
    swot_ds = xr.open_dataset(file)
    swot_ds = swot_ds[variables]
    swot_ds.load()

    ds = _normalized_ds(swot_ds.copy(), -180, 180)
    
    mask = (
        (ds.longitude <= lon_range[1])
        & (ds.longitude >= lon_range[0])
        & (ds.latitude <= lat_range[1])
        & (ds.latitude >= lat_range[0])
    ).compute()
    
    swot_ds_area = swot_ds.where(mask, drop=True)

    if swot_ds_area.sizes['num_lines'] == 0:
        print(f'Dataset {file} not matching geographical area.')
        return None

    for var in list(swot_ds_area.keys()):
        swot_ds_area[var].encoding = {'zlib':True, 'complevel':5}

    #filename = "subset_"+file[10:]
    filename = "subset_" + os.path.basename(file)
    filepath = os.path.join(output_dir, filename)
    swot_ds_area.to_netcdf(filepath)
    #print(f"Subset file created: {filename}")
        
    return filepath



############ manage files ################
 
def remove_subset_files(filenames, variables, lon_range, lat_range, output_dir):
    for file in os.listdir(output_dir):
        if file.startswith("subset"):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)





def subset_files(filenames, variables, lon_range, lat_range, output_dir):
    """ Subset datasets with geographical area.
    Args:
        filenames
            the filenames of datasets to subset
        variables
            variables to select
        lon_range
            the longitude range
        lat_range
            the latitude range
        output_dir
            output directory
    Returns:
        The list of subsets files.
    """
    #Remove old subset files based on input filenames
    
    #for file in filenames:
     #   subset_filename = f"subset{os.path.basename(file)}"
      #  subset_path = os.path.join(output_dir, subset_filename)
       # if os.path.isfile(subset_path):
        #    os.remove(subset_path)
            
    

    return [subset_file for subset_file in [_subset_ds(f, variables, lon_range, lat_range, output_dir) for f in filenames] if subset_file is not None]


####################### Computing functions #########################


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    R = 6371
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = c * R
    return distance    

###################### Temporal filtering ###########################


def butter_filter_velocity(time_data, velocity_data, cutoff_days=2.1, order=5):
    fs = 1   # 1 sample per day
    nyquist = 0.5 * fs
    normal_cutoff = (1 / cutoff_days) / nyquist

    b, a = butter(N=order, Wn=normal_cutoff, btype='low')
    
    return filtfilt(b, a, velocity_data)
    
    
    fs = 1   # sampling frequency in Hz (1/days)

    nyquist = 0.5 * fs
    
    normal_cutoff = (1 / cutoff_days ) / nyquist  

    b, a = butter(N=order, Wn=normal_cutoff, btype='low')

   
    filtered_velocity = filtfilt(b, a, velocity_data)
    
    return filtered_velocity

def filter_velocity(velocity_data, cutoff_hours=48, order=5):
    fs = 1   # 1 sample per hour
    nyquist = 0.5 * fs
    normal_cutoff = (1 / cutoff_hours) / nyquist

    b, a = butter(N=order, Wn=normal_cutoff, btype='low')
    
    return filtfilt(b, a, velocity_data)





def drop_num_nadir(ds):
    vars_to_drop = [var for var in ds.data_vars if 'num_nadir' in ds[var].dims]
    return ds.drop_vars(vars_to_drop)




def point_to_line_distance(point_lat, point_lon, line_start_lat, line_start_lon, line_end_lat, line_end_lon):
    """Calculate the minimum distance from a point to a line segment using vector projection"""

    x1, y1 = line_start_lon, line_start_lat
    x2, y2 = line_end_lon, line_end_lat
    x0, y0 = point_lon, point_lat
    
    # Line vector
    dx = x2 - x1
    dy = y2 - y1

    line_length_sq = dx**2 + dy**2
    
    # If line segment is actually a point
    #if line_length_sq == 0:
        #return haversine_distance(point_lat, point_lon, line_start_lat, line_start_lon)
    
    # Calculate projection parameter
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / line_length_sq
    
    # Clamp t to line segment
    if t < 0:
        nearest_x, nearest_y = x1, y1  # Use start point
    elif t > 1:
        nearest_x, nearest_y = x2, y2  # Use end point
    else:
        # Nearest point on line
        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy
    
    
    
    # Return distance to nearest point using haversine
    return haversine_distance(point_lat, point_lon, nearest_y, nearest_x)



def sample_transect(lat, lon, dx=2500):
    geod = Geod(ellps = 'WGS84')
    line_lats = []
    line_lons = []
    
    for i in range(len(lat)-1):
        lon1, lat1 = lon[i], lat[i]
        lon2, lat2 = lon[i+1] , lat[i+1]
        
        az1, az2, seg_len = geod.inv(lon1, lat1, lon2, lat2)
        n_dx = max(1, int(np.ceil(seg_len / dx)))
        n_intermediate = max(0, n_dx - 1)
        seg_points = geod.npts(lon1, lat1, lon2, lat2, n_intermediate)
        seg_all = [(lon1, lat1)] + seg_points + [(lon2, lat2)]
        
        # Skipping the first point of each seg
        for j, (lonp, latp) in enumerate(seg_all):
            if i > 0 and j == 0:
                continue
            line_lons.append(lonp)
            line_lats.append(latp)

    line_lats = np.array(line_lats)
    line_lons = np.array(line_lons)

    # compute the cumulative distance
    cumdist = np.zeros(len(line_lons), dtype=float)
    for k in range(1, len(line_lons)):
        _, _, d = geod.inv(line_lons[k-1], line_lats[k-1], line_lons[k], line_lats[k])
        cumdist[k] = cumdist[k-1] + d

    return line_lats, line_lons, cumdist
        




    
    
    

