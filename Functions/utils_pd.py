#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 19:17:51 2024

@author: keving
"""

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d



def df_remove_duplicates(df, sort_labels_by, subset_duplicates, ascending_sort=None, duplicates_keep='last' ):
    """
    remove duplicate rows from a dataframe 
    """
    if ascending_sort is not None:
        assert len(ascending_sort)==len(sort_labels_by)
    else:
        ascending_sort = [True]*len(sort_labels_by)
    
    print("DEBUG: ascending_sort: ",ascending_sort)
    df.sort_values(by=sort_labels_by, ascending=ascending_sort, inplace=True, ignore_index=True)
    df.drop_duplicates(subset=subset_duplicates, keep=duplicates_keep,  inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def sub_from_df(df, lims, col_name, reset_idx=True):
    """ extract a subset of a dataframe according to limits of a specific column"""
    if lims[0] is None: 
        lims = ( df[col_name].values.min(), lims[1] )
    elif lims[1] is None: 
        lims = ( lims[0],df[col_name].values.max() )
        
    df_sub = df.sort_values(by=col_name, ascending=True, ignore_index=True, inplace=False)
    df_sub = df_sub[ (df_sub[col_name] >= lims[0]) & (df_sub[col_name] <= lims[1]) ]
    if reset_idx:
        df_sub.reset_index(drop=True, inplace=True)
    return df_sub


def cut_df_by_coordinates(df, xlims, ylims=None, labels_coords=("UTM_X","UTM_Y"), reset_idx=True ):
        
        """extract subset DataFrame from spatial coordinates """
        if xlims is None:
            xlims = (df[labels_coords[0]].values.min(),df[labels_coords[0]].values.max() ) 
        if ylims is None: 
            ylims = (df[labels_coords[1]].values.min(),df[labels_coords[1]].values.max() ) 

        df_new = df[(df["UTM_X"]>=xlims[0]) & (df["UTM_X"]<=xlims[1]) &
                          (df["UTM_Y"]>=ylims[0]) & (df["UTM_Y"]<=ylims[1])]
        
        if reset_idx:
            df_new.reset_index(drop=True, inplace=True)
        return df_new
    
    
def scale_cols(df, colnames, scale_factors, inplace=True):
    """ scale columns of dataframe"""
    
    if inplace:
        df_mod=df
    else:
        df_mod=df.copy()
    
    if isinstance(colnames, str): 
        colnames=[colnames]
    if isinstance(scale_factors, (int,float)): 
        scale_factors = np.ones(len(colnames))*scale_factors
        
    for colname,fac in zip(colnames,scale_factors):
        df_mod[colname] *= fac
        
    return df_mod
    
def average_cols(df, colnames, colname_out, drop=True):
    """ average columns in pd Dataframe (and drop input columns)"""
    
    df[colname_out] = df[colnames].mean(axis=1)
    if drop: 
        df.drop(columns=colnames, inplace=True)
    return df

def average_cols_merge(df, colnames, colnames_out=None, drop=True):
    """ average (multiple) duplicated columns after dataframe merge"""
    for c, colname in enumerate(colnames): 
        cnames = [colname + "_x", colname + "_y"]
        
        if colnames_out is not None and isinstance(colnames_out,(list,tuple)):
            cname_out = colnames_out[c]
        else: 
            cname_out = colname
        df = average_cols(df, cnames, cname_out, drop=drop)
        
    return df
    
    
def interp_df(df, delta=None, indexes_interp=None, column_name_idx=None, dtype='float64'):
    """
    interpolate all colums of dataframe using one column as index
    """
    if indexes_interp is None:
        min_val, max_val = [df.iloc[idx][column_name_idx] for idx in (0,-1)]
        indexes_interp = np.arange(min_val,max_val+delta,delta)
    
    if column_name_idx is None:
        indexes = df.index.values
    else: 
        indexes = df[column_name_idx].values
    dict_new={}
    cols = df.columns.drop(column_name_idx); #cols=cols.drop(column_name_idx)
    for col in cols:
        print(f"DEBUG: interp col:{col}")
        dict_new[col] = np.interp(indexes_interp, indexes, df[col].values)
    dict_new[column_name_idx] = indexes_interp
    
    return pd.DataFrame(dict_new, dtype=dtype)


def common_items(arr1, arr2, return_array=True):
    """ return the common elements between two arrays"""
    arr3 = list(set(arr1) & set(arr2))
    if return_array:
        return np.array(arr3)
    else:   return arr3

def common_items_df(df, separator='shotnumber', label_item="channel_abs" ):
    """ keep only elements of df that are common among sections"""
    sections = np.unique(df[separator].values)
    
    for i, section in enumerate(sections):
        if i==0: 
            print("Do nothing in first round")
        else:
            
            if i == 1: 
                items = common_items( df.loc[df[separator]==sections[i-1], label_item].values, 
                                   df.loc[df[separator]==section, label_item].values, return_array=True)
            else: 
                items = common_items( items, df.loc[df[separator]==section, label_item].values, return_array=True)
                
    df_common = df[df[label_item].isin(items)].sort_values(by=[separator,label_item], ignore_index=True)    

    return df_common 

def df2data(df, separator="shotnumber", label_data="time", check_len=True):
    """ create numpy data array from dataframe"""
    
    sections = np.unique(df[separator].values)
    nsec=len(sections)
    
    if check_len:
        lengths=np.zeros(nsec, dtype='int32')
        for i, section in enumerate(sections):
             lengths[i] = len( df[df[separator]==section] )
        assert np.array_equal(lengths-lengths[0], np.zeros(len(lengths), dtype='int32') )
         
    for i, section in enumerate(sections):
        df_tmp = df[df[separator]==section]
        if i==0: 
            data = np.zeros([len(sections),len(df_tmp)])
        data[i,:] = df_tmp[label_data].values
        
    return data

def data2df_2d(data, axis0, axis1 ,label_ax0="shotnumber", label_ax1="channel_abs", label_data="time", dtypes=('int32','int32',None) ):
    """ create pd dataframe from 2d numpy data array from"""
    
    assert data.shape[0]==len(axis0)
    
    count=0
    for i in range(data.shape[0]):
        df_tmp = pd.DataFrame( {label_ax0:[axis0[i]]*data.shape[1], label_ax1: axis1, 
                               label_data:data[i,:]} )
        if len(df_tmp)>0:
            if count==0:
                df = df_tmp.copy()
            else: 
                df = pd.concat([df,df_tmp])
            count+=1
            
    if dtypes is not None:
        dict_dtypes={}
        for l,label in enumerate( (label_ax0,label_ax1,label_data) ):
            if dtypes[l] is not None: 
                dict_dtypes[label]=dtypes[l]
        df = df.astype(dict_dtypes)
    df.reset_index(drop=True, inplace=True)
            
    return df

def mean_std_by(df, colname_group, colname_data, colname_out_mean="mean", colname_out_std="std"):
    """ group dataframe by certain column and compute mean and std to create new dataframe"""
    df_grouped = df.groupby(colname_group)[colname_data]
    mean_by = df_grouped.mean()
    std_by = df_grouped.std()
    df_mean = pd.DataFrame( {colname_group: mean_by.index.values, 
                                colname_out_mean: mean_by.values, 
                                colname_out_std:std_by.values} )
    return df_mean


def gaussfilt_df(df, cname_sep="shotnumber", cname_data="traveltime", cname_filt="traveltime_gaussfilt", 
                 sigma=3):
    """ apply 1D gaussian  filter to separated section of DF"""
    for i,val in enumerate(np.unique(df[cname_sep])):
        df_tmp = df[df[cname_sep]==val]
        df_tmp[cname_filt] = gaussian_filter1d(df_tmp[cname_data].values , sigma) 
        if i==0: 
            df_pro = df_tmp.copy()
        else: 
            df_pro = pd.concat([df_pro,df_tmp])
    return df_pro
    

