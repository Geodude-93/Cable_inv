#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 14:09:34 2023

@author: keving
"""
import numpy as np
from matplotlib import pyplot as plt, ticker as mticker


plot_params_paper = {
         'figure.titlesize': 11,
         'axes.labelsize': 9,
         'axes.titlesize': 9,
         'xtick.labelsize': 9,
         'ytick.labelsize':9,''
         'xtick.major.size': 3.0,
         'ytick.major.size': 3.0, 
         'legend.fontsize': 8}

def cm2inches(cms, as_tuple=True):
    """convert cm to inches """
    inches = np.array(cms)/2.54
    if as_tuple: 
        return tuple(inches)
    else: 
        return inches

def label_subfigs(axes, x=0.05, y=0.95, labels=None, kw_text=None, zorder=5, halfbracket=True, 
                  fontsize=12, xy_shifts=None, bbox=False, bbox_props=None):
    """
    plot figure labels on subfigures
    """
    if kw_text is None: 
        kw_text = dict(fontsize=fontsize, fontweight='bold', ha="left", va="top")
    
    if labels is None: 
        labels = ["a","b","c","d","e","f","g,""h"]  #('A','B','C','D','E','F','G','H')
    if halfbracket: 
        labels = [l+")" for l in labels]
        
   
    if bbox_props is None: 
        if bbox: 
            bbox_props=dict(boxstyle="round4", fc="w", ec="k", pad=0.25, alpha=0.75)
        
    for k,ax in enumerate(axes):
        dx_tmp, dy_tmp = 0.0, 0.0
        if isinstance(xy_shifts, dict) and  str(k) in xy_shifts.keys():
            dx_tmp, dy_tmp = [xy_shifts[str(k)][idx] for idx in (0,1)]
        
        ax.text(x+dx_tmp, y+dy_tmp, labels[k], transform=ax.transAxes, zorder=zorder,  
                bbox=bbox_props, **kw_text)
    return

def window_coords(xlims,ylims):
    """
    create overlay window for plot according to limits
    """
    return np.array([ (xlims[0],xlims[0],xlims[1],xlims[1],xlims[0]),\
                              (ylims[1],ylims[0],ylims[0],ylims[1],ylims[1]) ] ).T
        
        
        
def det_ax_ratio_map(figsize, width_ratios, height_ratios=None, col_idx=-1, row_idxes=0):
    "compute ax ratio of the map w.r.t figure size, width and height ratios"
    if height_ratios is None: 
        height_ratios=(1,)
    if isinstance(row_idxes,(list,tuple,np.ndarray))==False:
        row_idxes=(row_idxes,row_idxes)
    
    width_map =  (np.sum(width_ratios[col_idx])/np.sum(width_ratios)) * figsize[0]
    height_map = (np.sum(height_ratios[row_idxes[0]:row_idxes[-1]+1]) / np.sum(height_ratios)) * figsize[1]
    return width_map / height_map


def get_map_lims_by_ratio(coord_lims, ax_coords='X', ax_ratio=None, figsize=None):
    """
    determine the map coodinate difference in one axis providing the coordinate limits and the axis ratio of the other axis

    Parameters
    ----------
    coord_lims : tuple or list
        coordinate limits of provided axis.
    ax_coords : str or int, optional
        axis type of provided axis: either element of (X,x,0). for X-axis or (Y,y,1) for Y-axis. The default is 'X'.
    ax_ratio : float, optional
        ax_ratio for output figure in terms of X/Y. The default is None.
    figsize : tuple, optional
        used to compute ax_ratio if ax_ratio is not provided. The default is None.

    Returns
    -------
    diff_target : float
        the coordinate difference of the other axis.

    """
    
    if ax_ratio is None and figsize is not None: 
        ax_ratio = figsize[0] / figsize[1]
    if coord_lims:
        diff = coord_lims[1]-coord_lims[0]
    if ax_coords in ['X','x',0]:
        diff_target = diff  / ax_ratio
    elif ax_coords in ['Y','y',1]:
        diff_target = diff * ax_ratio
        
    return diff_target
        

def get_positions_secondary_axis(offsets, offsets_label):
    """
    create position for target labels on secondary axis (e.g. offset axis)

    Parameters
    ----------
    offsets : 1d np.array
        array of offsets around reference location.
    offsets_label : 1d np.array
        array of target labels for secondary axis.

    Returns
    -------
    1d np.array
        array of relative positions on new (secondary) axis.

    """
    
    offsets_pos = offsets + np.abs(offsets.min())
    offsets_labels_pos = offsets_label + np.abs(offsets.min())
    return offsets_labels_pos / offsets_pos.max()

def modify_xyticks(axes, axis='x', minor_tick_interval=2, width=1, length=4):
    for ax in axes:
        if axis in ("x","xy","yx"): 
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(minor_tick_interval))
        if axis in ("y","xy","yx"): 
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(minor_tick_interval))
        ax.tick_params(which='minor', width=width, length=length)
    return

def hide_yaxis(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    ax.yaxis.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    return

def add_north_arrow(ax, xpos_rel=0.9, ypos_rel=0.9, fontsize=16, npixels_arrow=60, arrowprops=None, dpi=150):
    
  if arrowprops is None:
      arrowprops=dict(arrowstyle="-|>", facecolor="black" ,
                      linewidth=2)
 
  npixels_arrow_norm = npixels_arrow * dpi/150

  px = xpos_rel * ax.figure.bbox.width
  py = ypos_rel * ax.figure.bbox.height

  # Draw an arrow with a text "N" above it using annotation
  ax.annotate("N", xy=(px, py), fontsize=fontsize, fontweight="bold", xycoords="figure pixels")
  ax.annotate("",  xy=(px,  py), xytext=(px, py-npixels_arrow_norm),xycoords="figure pixels",
          arrowprops=arrowprops)
  return

def ax_rel_coods(ax, dx_label=100, dy_label=100, dtype='int32'):
    """substitute the real axis coordinates with relative ones """
    xlims_orig = ax.get_xlim(); ylims_orig = ax.get_ylim()
    diff_x, diff_y = [arr[1]-arr[0] for arr in (xlims_orig,ylims_orig)]
    
    if dx_label:
        xticks_new = np.arange(0,diff_x, dx_label).astype(dtype)
        xtick_locs = xticks_new + xlims_orig[0]
        ax.set_xticks(xtick_locs, xticks_new)
    if dy_label:
        yticks_new = np.arange(0,diff_y, dy_label).astype(dtype)
        ytick_locs = yticks_new + ylims_orig[0]
        ax.set_yticks(ytick_locs, yticks_new)
    
    return 
    
    