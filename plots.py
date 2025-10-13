#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:32:36 2025

@author: dihyachaal
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.path as mpath #creating custom map boundaries 
import cartopy.feature as cfeat
import cartopy.mpl.ticker as ctk
from matplotlib.lines import Line2D



#""""""""""""""""""""plotting functions"""""""""""""""""


def figure(figsize, nrows, ncols, region):
   
    lon_min, lon_max, lat_min, lat_max = region
    
    rect = mpath.Path([[lon_min, lat_min], [lon_max, lat_min],
    [lon_max, lat_max], [lon_min, lat_max], [lon_min, lat_min]]).interpolated(100)

    proj=ccrs.NearsidePerspective(central_longitude=(lon_min+lon_max)*0.5,
    central_latitude=(lat_min+lat_max)*0.5)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, constrained_layout=True, figsize=figsize, subplot_kw={'projection': proj}, dpi=300)
                        
    return fig, axes, rect

def axis_shape(ax, rect, addCoastline, addCheckerboard):
    
    proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
    rect_region = proj_to_data.transform_path(rect)
    ax.set_boundary(rect_region)
    ax.set_xlim(rect_region.vertices[:,0].min(), rect_region.vertices[:,0].max())
    ax.set_ylim(rect_region.vertices[:,1].min(), rect_region.vertices[:,1].max())
    ax.spines['geo'].set_edgecolor('black')
    ax.spines['geo'].set_linewidth(0.5) 
    
    if addCoastline:
        ax.coastlines(zorder=3, linewidth=2)
        ax.add_feature(cfeat.LAND, color='lightgray', zorder=3)
        #ax.add_feature(cfeat.OCEAN)


    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linestyle='dashed', linewidth=0.1)
    gl.right_labels=False
    gl.top_labels=False
    gl.rotate_labels=False
    gl.xlocator=ctk.LongitudeLocator(6)
    gl.ylocator=ctk.LatitudeLocator(6)
    #gl.xformatter=ctk.LongitudeFormatter(zero_direction_label=True)
    gl.yformatter=ctk.LatitudeFormatter()
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    try:
        for artist in gl.right_label_artists:
            artist.set_visible(False)
    except AttributeError:
        pass

    if addCheckerboard:
        
        ax.spines['geo'].set_visible(True)  # removing or keep the default frame

        # Take boundary vertices
        verts = rect_region.vertices  # boundary vertices

        dists = np.sqrt(np.sum(np.diff(verts, axis=0)**2, axis=1))  # total length of boundary
        total_length = np.sum(dists)

        num_segments = 60  

        segment_length = total_length / num_segments  # same space between the points

        ###### creating black/white segments ####
        cum_dist = np.concatenate([[0], np.cumsum(dists)])
        new_points = [verts[0]]
        for i in range(1, num_segments):
            target_dist = i * segment_length
            idx = np.searchsorted(cum_dist, target_dist) - 1
            ratio = (target_dist - cum_dist[idx]) / dists[idx]
            new_point = verts[idx] + ratio * (verts[idx+1] - verts[idx])
            new_points.append(new_point)
        new_points.append(verts[-1])  # ensure we close

        # Now draw each small segment
        for i in range(len(new_points)-1):
            color = 'black' if i % 2 == 0 else 'white'
            x0, y0 = new_points[i]
            x1, y1 = new_points[i+1]
            ax.add_line(Line2D([x0, x1], [y0, y1], color=color, linewidth=4, transform=ax.transData, zorder=6))
  
            
  
def plot_datasets(datasets, variable, vminmax, title, extent=None):
    cb_args = dict(
        add_colorbar=True,
        cbar_kwargs={"shrink": 0.3},
    )
    
    plot_kwargs = dict(
        x="longitude",
        y="latitude",
        cmap="bwr",
        vmin=vminmax[0],
        vmax=vminmax[1],
    )

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection=ccrs.PlateCarree()))
    if extent: ax.set_extent(extent)
    
    for ds in datasets:
        ds[variable].plot.pcolormesh(
            ax=ax,
            **plot_kwargs,
            **cb_args)
        cb_args=dict(add_colorbar=False)

    ax.set_title(title)
    ax.coastlines()
    gls = ax.gridlines(draw_labels=True)
    gls.top_labels=False
    gls.right_labels=False

    return ax  
  
    
  
    
  
    
  
    
  
    
  
    
            