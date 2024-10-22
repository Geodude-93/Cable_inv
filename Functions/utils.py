#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 17:08:58 2021

@author: kevin
"""
import os, sys
import numpy as np, pandas as pd
import math


def done(mess="done"):
    print(mess); 
    sys.exit()
    
def without_keys(d, keys):
    """ remove keys from dict"""
    return {x: d[x] for x in d if x not in keys}

def array_size(arr, unit='gb'):
    """ memory size of np array in different units"""
    if unit.lower()=='gb':
        fac=1e-09
    elif unit.lower()=='mb':
        fac=1e-06
    elif unit.lower()=='kb':
        fac = 1e-03
    elif unit.lower() in ('b','byte','bytes'):
        fac=1
        
    return arr.size * arr.itemsize * fac 


def is_pos_def(x, warning_lvl=2):
    """ check if matrix is positive definite"""
    boolx = np.all(np.linalg.eigvals(x) > 0)
    
    if boolx==False: 
        if warning_lvl==0:
            pass
        elif warning_lvl==1:
            print("WARNING: matrix is not positive definite. Continue")
            return boolx
        else: 
            raise ValueError("WARNING: matrix is not positive definite. Abort")
            
    return boolx

def check_dir(path, levels_check=1):
    """
    helper function that checks whether subdirectories of path already exists and creates them if missing

    Parameters
    ----------
    path : str
        complete path including subdirectories.
    levels_check : int, optional
        defines how many subdirectories are checked/created. The default is 1 and only checks the last subdirectory.
        Note that it is counted reversed, hence from the end of the path 
    Returns
    -------
    None.

    """
    path_elements = path.split('/')
    if levels_check<1: 
        levels_check=1
    elif levels_check >= len(path_elements):
        raise ValueError("!! subdirs to check exceeds number of path elements")
    
    path_elements = path.split('/')
    path_elements_base = path_elements[0:-levels_check]
    paths_elements_check = path_elements[-levels_check::]
    for e,el in enumerate(path_elements_base): 
        if e==0: 
            path_base =  el
        else: 
            path_base += '/' + el
    
    for k,item in enumerate(paths_elements_check):
        if k==0: 
            pathx =  path_base + '/' + item
        else: 
            pathx += '/' + item
        if not os.path.isdir(pathx):
            os.makedirs(pathx); print(f"created path: {pathx}")
        else: 
            print(f"path: {pathx} exists already")

def check_dicts_consitency_scalars(list_dicts, dict_check, return_result=False, warning=1, verbose=True):
    """
    check multiple dictionaries whether they are consitent in values for given parameters,
    the values should be scalars

    Parameters
    ----------
    list_dicts : list of dicts
        list of dicts to check.
    dict_check : dict
        dict containing key-value pairs of parameters to check for all the handed dicts.
    return_result : bool, optional
        return all checked values. Not implemented yet. The default is False.
    warning : int, optional
        gives out warning or raises an error for >=2 when parameters are inconsitent. The default is 1.
    verbose : bool, optional
        activates verbose mode. The default is True.

    Raises
    ------
    ValueError
        when inconsistencies in the dicts are detected.

    Returns
    -------
    None.

    """
    vals=np.zeros(len(list_dicts))
    # dict_check={"general":["ch_stack","dec_chs","dec_chs_extr","dt"], \
    #             "rms":["dt_rms","ns_winhalf_rms","overlap_perc"]}
    for key in dict_check:  
        if verbose:     print("key,vals: ",key,dict_check[key])
        for label in dict_check[key]: 
            for i,info in enumerate(list_dicts):
                vals[i]=info[key][label]; 
            if (vals==vals[0]).all()==False: 
                if warning==1: 
                    print("!! WARNING: parameters of different Dicts are not consistent: \
                      key: {}, label: {}, values: {}".format(key,label,vals))  
                elif warning>=2: 
                    raise ValueError("!! ABORT: parameters of different Dicts are not consistent: \
                      key: {}, label: {}, values: {}".format(key,label,vals))
            else: 
                if verbose:     print("key={}, label={}, vals={}".format(key,label,vals))

    if return_result: 
        print("!! return of results not implemented yet")
    else:   return 
    
def check_dicts_consitency(list_dicts, dict_check, return_result=False, warning=1, verbose=True):
    """
    check multiple dictionaries whether they are consistent in values for given parameters, 
    the values can be scalars, list or np.arrays

    Parameters
    ----------
    list_dicts : list of dicts
        list of dicts to check.
    dict_check : dict
        dict containing key-value pairs of parameters to check for all the handed dicts.
    return_result : bool, optional
        return all checked values. Not implemented yet. The default is False.
    warning : int, optional
        gives out warning or raises an error for >=2 when parameters are inconsitent. The default is 1.
    verbose : bool, optional
        activates verbose mode. The default is True.

    Raises
    ------
    ValueError
        when inconsistencies in the dicts are detected.

    Returns
    -------
    None.

    """
   
    # dict_check={"general":["ch_stack","dec_chs","dec_chs_extr","dt"], \
    #             "rms":["dt_rms","ns_winhalf_rms","overlap_perc"]}
    for key in dict_check:  
        if verbose:     print("\nkey,vals: ",key,dict_check[key])
        for label in dict_check[key]: 
            list_vals=[]
            for i,info in enumerate(list_dicts):
                list_vals.append(info[key][label]); 
            if isinstance(list_vals[0],list):
                 boolx = np.all([listx==list_vals[0] for listx in list_vals[1::]])
            elif isinstance(list_vals[0],np.ndarray):
                boolx = np.all([np.array_equal(arr,list_vals[0]) for arr in list_vals[1::]])
            else: 
                boolx = np.all((list_vals==list_vals[0]))
            if boolx==False: #(vals==vals[0]).all()
                if warning==1: 
                    print("!! WARNING: parameters of different Dicts are not consistent: \
                      key: {}, label: {}, values: {}".format(key,label,list_vals))  
                elif warning>=2: 
                    raise ValueError("!! ABORT: parameters of different Dicts are not consistent: \
                      key: {}, label: {}, values: {}".format(key,label,list_vals))
            else: 
                if verbose:     print("key={}, label={}, vals={}".format(key,label,list_vals))

    if return_result: 
        print("!! return of results not implemented yet")
    else:   return 
    
    
    
def concat_csvs(list_csvs, path=None, label_col='label', labels_sections=None, outfile=None, sep='\t', header=0, idx_out=False):
    """
    concatenate several csvs and create a dataframe

    Parameters
    ----------
    list_csvs : list of strings
        list of filenames.
    path : str, optional
        path to csvs. The default is None.
    label_col : str, optional
        label for new column for section distinction. The default is 'label'.
    labels_sections : list of stings, optional
        labels for each section/csv. The default is None.
    outfile : str, optional
        output file. The default is None.
    sep : str, optional
        column separator for in and output files. The default is '\t'.
    header : int, optional
        rownumber of headerrow. The default is 0.
    idx_out : bool, optional
        save index in output file. The default is False.

    Returns
    -------
    df_main : pd.dataframe
        concatenated dataframe.

    """
    for i,infile in enumerate(list_csvs):
        if infile[-4::] not in ['.csv','.txt']:
            infile = infile + '.csv'
        if path: 
            infile = path + infile 
        df_tmp = pd.read_csv(infile, sep=sep, header=header)
        if labels_sections is not None:
            df_tmp[label_col] = [labels_sections[i]]*df_tmp.shape[0]
        if i==0:
            df_main = df_tmp.copy()
        else: 
            df_main = pd.concat([df_main,df_tmp], ignore_index=True)
    if outfile:
        df_main.to_csv(outfile, sep=sep, idx=idx_out)
    return df_main
    



def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]



    
def get_idx_closest_value(arr, val, full_output=True):
    """ get idx for closest value in array 
    """
    diff = arr-val
    idx = np.argmin(np.abs(diff))
    val_min = arr[idx]
    diff_min = diff[idx] 
    
    if full_output:    
        return idx, val_min, diff_min
    else:
        return idx
     
def get_idx_closest_smaller_value(arr, val, full_output=False):
    """ get idx for closest value in list 
    """
    diffs = (arr - val)*(-1)
    val_min =min([i for i in diffs if i >= 0])  # find the smallest positive value
    idx =diffs.tolist().index(val_min)    # get index of determined value -> index of file for the shot
    
    if full_output:     
        return idx, val_min, arr[idx]
    else: 
        return idx, val_min
        

def get_idxes_closest_smaller_values(arr, list_vals):
    """ get list of indexes for closest value in list 
    """
    if isinstance(list_vals, int) or isinstance(list_vals, float):
        list_vals = [list_vals]
    if isinstance(arr,list):
        arr=np.array(arr)
        
    vals_min=[]; list_idxs=[];
    for v,val in enumerate(list_vals):
        diffs = (arr - val)*(-1)
        vals_min.append(min([i for i in diffs if i >= 0]))  # find the smallest positive value
        list_idxs.append(diffs.tolist().index(vals_min[v]))    # get index of determined value -> index of file for the shot
        
    return list_idxs, vals_min



def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, math.sqrt(variance)
    


