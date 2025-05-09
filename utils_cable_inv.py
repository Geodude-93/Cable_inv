#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:10:11 2024

@author: keving
"""
import time
import numpy as np 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#import matplotlib.patheffects as mpl_pe
from scipy.ndimage import gaussian_filter1d

from Functions import utils, utils_pd, utils_plot

DEC_CHS=6; #DX_PHY=1.02; DX=DEC_CHS*DX_PHY
TIMESTAMP_FIRST_SHOT_LINE2=1656604417.861809
CHANNEL_REF=7661


def dists_2d(coords, target):
    """ distance between coordinate pairs and target """
    return ((target[0] - coords[:,0])**2 + (target[1] - coords[:,1])**2 )**0.5
    

def data2datavec(d):
    """ flatten data array to 1D data vector"""
    return d.flatten()

def datavec2data(dvec, nrows): 
    """ generate data array (src x rec) from data vector"""
   
    N = len(dvec)
    assert N % nrows == 0
    ncols = int(N/nrows) 
    
    return dvec.reshape((nrows, ncols))

def model2paras(x_rec, y_rec):
    """ generate 1d parameter vector from model"""
    return  np.concatenate([x_rec,y_rec])

def paras2model(parameters, n_rec):
    """ convert model parameter vector to coordinate pairs"""
    
    assert len(parameters) == 2*n_rec
    x_rec = parameters[0:n_rec]
    y_rec = parameters[n_rec:2*n_rec]
    return (x_rec, y_rec)


def data2df(df, data, colname_data="traveltime_pred", residual=True, colname_obs="traveltime", colname_res="res_time", 
            iteration=None): 
    """ update dataframe of data with new traveltimes and residuals """
    df_data = df.copy()
    df_data[colname_data] = data
    if residual:
        df_data[colname_res] = df_data[colname_data] - df_data[colname_obs]
    if isinstance(iteration, int): 
        df_data["iter"] = (np.ones(len(df_data))*iteration).astype('int32') 
        
    return df_data


def df_from_data(d, shots, channels):
    """ create a dataframe from the 1D data array, the shots and channels"""
    n_rec = channels.shape[0]
    for s, shot in enumerate(shots):
        df_tmp = pd.DataFrame(dict(shotnumber=(np.ones(n_rec)*shot).astype('int32'),
                                   chidx_local=np.arange(0,n_rec), channel_idx=channels,
                              traveltime=d[s*n_rec:s*n_rec+n_rec]))
        if s==0: 
            df = df_tmp.copy()
        else: 
            df = pd.concat([df, df_tmp])
        
    return df


    
def gen_receivers(orig, m, n_rec=100, d=6.0, noise=0.0, sign_x=1, sign_y=-1):
    """ generate receiver line based on origin and slope"""
    
    dx = np.sqrt(d**2 / (m**2+1))
    dy = np.sqrt(d**2 -dx**2)
    xs_rec = orig[0] + sign_x* np.arange(0,n_rec)*dx
    ys_rec = orig[1] + sign_y* np.arange(0,n_rec)*dy
    
    return xs_rec, ys_rec

def gen_cable_init_apexes(df_shots, dec_chs=6, label_channel_apex="channel_idx_apex",
                          flipflop_indexes=(0,1) ):
    """ generate initial cable position from apexes of flipflop source positions"""
    
    for i, remainder in enumerate( flipflop_indexes ): # even and odd shotnums
        df_shots_tmp = df_shots[df_shots["shotnumber"]%2==remainder]
        chlims_apex = (df_shots_tmp[label_channel_apex].min(), df_shots_tmp[label_channel_apex].max())
        chs_interp = np.arange(chlims_apex[0],chlims_apex[1], dtype='int32')
        xrec_init, yrec_init = [ np.interp(chs_interp, df_shots_tmp[label_channel_apex].values, df_shots_tmp[label].values) \
                                for label in ("UTM_X","UTM_Y")]
   
        df_tmp = pd.DataFrame(dict(UTM_X=xrec_init, UTM_Y=yrec_init, channel_idx=chs_interp \
                                         ) ).sort_values(by="channel_idx", ignore_index=True)
        if i==0: 
            df_cable = df_tmp.copy()
        elif i==1: 
            df_cable = df_cable.merge(df_tmp[["channel_idx","UTM_X","UTM_Y"]], on="channel_idx")
    df_cable = utils_pd.average_cols_merge(df_cable, ("UTM_X","UTM_Y"), colnames_out=None, drop=True)
            
    return df_cable

## helper functions for forward equation

def _path_lengths(coords, coords_sec, coords_src, z_src):
    """ compute path lengths for straight rays"""
    return  ( (coords-coords_src[0])**2 + (coords_sec-coords_src[1])**2 + z_src**2)**0.5 

def _traveltimes_rays(coords, coords_sec, coords_src, z_src, v=1500):
    """ compute traveltimes for straight rays"""
    return _path_lengths(coords, coords_sec, coords_src, z_src) / v


### derivatives of forward equation 

def _ddm_coord(coords, coords_sec, coords_src, z_src, v=1500):
    """ derivative of forward equation w.r.t. receiver coordinates """
    return (coords-coords_src[0]) / (v*_path_lengths(coords, coords_sec, coords_src, z_src))
    #return (coords-coords_src[0]) / (vel_water*( (coords-coords_src[0])**2 + (coords_sec-coords_src[1])**2 + z_src**2)**0.5)

def _ddm2_coord(coords, coords_sec, coords_src, z_src, v=1500):
    """ second derivative of forward equation w.r.t. receiver coordinates """
    
    return (coords_src[1]**2 -2*coords_sec*coords_src[1] + coords_sec**2 + z_src**2) \
        / (v* (coords_src[0]**2 -2*coords*coords_src[0] +coords_src[1]**2 \
               -2*coords_src[1]*coords_sec +coords**2 +coords_sec**2 + z_src**2)**1.5 )
            
def _dxdy_coord(coords, coords_sec, coords_src, z_src, v=1500):
    """ cross derivative of forward equation w.r.t. receiver coordinates """
    return (-1)* ( ( (coords-coords_src[0])*(coords_sec-coords_src[1]) ) \
                  / (v* ( (coords-coords_src[0])**2 +(coords_sec-coords_src[1])**2 +z_src**2 )**1.5 ) )
          


### derivatives for squared forward equation 

def _ddm_ttsq_coord(coords, coords_sec, coords_src, z_src, tau0, htau, delta_t, v=1500):
    """ derivative of squared forward equation w.r.t. receiver coordinates """
    
    return (2*(coords-coords_src[0])* ( _traveltimes_rays(coords, coords_sec, coords_src, z_src, v=v) 
                                       +tau0 + htau*delta_t ) ) \
        / (v* _path_lengths(coords, coords_sec, coords_src, z_src))
        
        
def _ddm2_ttsq_coord(coords, coords_sec, coords_src, z_src, tau0, htau, delta_t, v=1500 ):
    """ second derivative of squared forward equation w.r.t. receiver coordinates """
    
    dists = _path_lengths(coords, coords_sec, coords_src, z_src)
    dists_sq = dists**2
    
    A = (-1)*(coords-coords_src[0])**2 *( dists + v*htau*delta_t + v*tau0 )
    B =  dists_sq *( dists + htau*delta_t*v + tau0*v )
    C =  (coords-coords_src[0])**2 * dists
    D = v**2 * dists_sq**1.5
    
    result = (2*(A+B+C)) / D
    
    return result

def _dxdy_ttsq_coord(coords, coords_sec, coords_src, z_src, tau0, htau, delta_t, v=1500 ):
    """ cross derivatives of squared forward equation w.r.t. receiver coordinates"""
    
    return ( -2*(coords_src[0]-coords)*(coords_src[1]-coords_sec)*(htau*delta_t+tau0) ) \
        / (v*((coords_src[0]-coords)**2 + (coords_src[1]-coords_sec)**2 + z_src**2)**1.5 )


def _ddmxy_ttsq_coord(coords, coords_sec, coords_src, z_src, tau0, htau, delta_t, v=1500):
    """cross derivative dxdy of squared forward equation """  

    return ( 2*(coords-coords_src[0])* (coords_src[1]-coords_sec)* (htau*delta_t*+tau0) ) \
        / (v* (coords_src[0]**2 -2*coords_src[0]*coords + coords_src[1]**2 -2*coords_src[1]*coords_sec \
             + z_src + coords**2 + coords_sec**2  ) )**1.5
    

def _ddm_ttsq_tau0(coords, coords_sec, coords_src, z_src, tau0, ht, delta_t, v=1500):
    """ 1. derivative of squared forward Eq. w.r.t tau0"""
    return 2*( _traveltimes_rays(coords, coords_sec, coords_src, z_src, v=v) \
              + ht*delta_t + tau0)
        
def _ddm2_ttsq_tau0(n):
    """ 2. derivative of squared forward Eq. w.r.t tau0"""
    return np.ones(n, dtype='float32')*2

def _ddm_ttsq_ht(coords, coords_sec, coords_src, z_src, tau0, ht, delta_t, v=1500):
    """ 1. derivative of squared forward Eq. w.r.t htau"""
    return 2*delta_t*( _traveltimes_rays(coords, coords_sec, coords_src, z_src, v=v) \
              + ht*delta_t + tau0)
        
def _ddm2_ttsq_ht(n, delta_t):
    """ 2. derivative of squared forward Eq. w.r.t ht"""
    return np.ones(n, dtype='float32')*2* delta_t**2

def _dtau_dh_ttsq(n, delta_t):
    """cross derivative of squared forward Eq. w.r.t tau and ht"""
    return np.ones(n, dtype='float32')*2* delta_t



def forward_single(xy_src, xcoords_rec, ycoords_rec, t_shift=0, vel_water=1500, wdepth=65):
   """ forward model arrival times for a single sources """
   distsh = dists_2d(np.array([xcoords_rec,ycoords_rec]).T, xy_src) 
   return ((distsh**2 + wdepth**2)**0.5) / vel_water + t_shift


def forward_multi(parameters, n_rec, x_src, y_src, z_src, vel_water=1500, paras_tshift_ext=(0,),
                   weights=None, bools_rec=None, delta_ts=None, 
                   sigma_noise=0, squared=False):
    """ forward model from multiple sources to multiple receivers"""
    
    paras_tmp = paras2model(parameters, n_rec)
    x_rec, y_rec = paras_tmp[0], paras_tmp[1]
    nparas_tshift = len(paras_tshift_ext)
        
    assert n_rec==len(x_rec)==len(y_rec)
    n_src = len(x_src); assert len(y_src)==n_src
    if bools_rec is not None: 
        if bools_rec.ndim==2:
            assert bools_rec.shape == (n_src ,n_rec)
        
    for s in range(n_src):

        # select channels used for shot        
        if n_src==1: 
            bools_rec_tmp = bools_rec
        else: 
            bools_rec_tmp = bools_rec[s,:]
        xrec_tmp, yrec_tmp = x_rec[bools_rec_tmp], y_rec[bools_rec_tmp]
        
        # determine time shift 
        if nparas_tshift==2: 
            tshift = paras_tshift_ext[0] + paras_tshift_ext[1]* delta_ts[s]
        else: 
            tshift = paras_tshift_ext[0]
        
        d_tmp = _traveltimes_rays(xrec_tmp, yrec_tmp, (x_src[s],y_src[s]), z_src, v=vel_water) + tshift
        
        if s==0: 
            d = d_tmp.copy()
        else: 
            d = np.concatenate((d,d_tmp), axis=0)
    if squared:
        d = d**2
    if sigma_noise >0:
        d +=  np.random.normal(scale=sigma_noise, size=d.shape[0])
    if weights is not None:
        d = d @ weights
        
    return d



def rms_misfit(d_res):
    """ compute basic RMS error"""
    return np.sqrt( np.mean(d_res**2))

def rms_misfit_weighted(d_res, sigmas):
    """ compute basic RMS error"""
    return np.sqrt( np.mean(d_res**2 / sigmas**2))


def lsqs_mfit_weighted(d_res, W=None):
    """ compute CHI for each component """
    assert d_res.ndim==1
    N = len(d_res)
    if W is None: 
        W = np.identity(N)
    else: 
        assert W.shape==(N,N)
        
    return np.linalg.norm( d_res.T @ W.T @ W @ d_res )

def check_weights(W, N):
    """ check dimensions of weight mat"""
    if W is None: 
        W = np.identity(N)
    else: 
        assert W.shape[0]==W.shape[1]==N
    return W
        

def lsqr_misfit_model(d_res, parameters, alpha=1, Wd=None, Wm=None, check_weights=False):
    """ objective function including data misfit term and deviation from startin model"""
    
    # prep
    assert d_res.ndim == parameters.ndim == 1
    N, M = len(d_res), len(parameters)
    
    if check_weights:
        Wd, Wm = [ check_weights(W, num) for W,num in zip( [Wd,Wm],[N,M] )]
        
    return 0.5 * d_res.T @Wd.T @Wd @d_res + (alpha/2)* parameters.T @Wm.T @Wm @parameters
    


def gen_linear_weights(arr, ref_max, lims_weights=(1,100) ):
    """ generate linear weights with respect to reference"""
    diffs_abs = np.abs(arr - ref_max)
    max_diff = diffs_abs.max()
    weight_per_x = (lims_weights[1]-lims_weights[0]) / max_diff
    weights = lims_weights[1] - diffs_abs* weight_per_x
    return weights

def get_apexes(df, label_time="timestamp", label_channel="channel_idx", labels_out=("channel_apex","time_apex"), 
               nchs_edge=3):
    """ get apex from smallest times"""
    shotnums = np.unique(df.shotnumber)
    times_apex = np.zeros(len(shotnums))
    chs_apex = np.zeros(len(shotnums), dtype='int32')
    flags_apex = np.ones(len(shotnums), dtype='int32')
    for s, shot in enumerate(shotnums):
              
        df_tmp = df[df["shotnumber"]==shot].reset_index(drop=True)
        
        time_min = df_tmp[label_time].min()
        idxes_min = np.where(df_tmp[label_time]==time_min)
        idx_min = int(np.mean(idxes_min))
        if idx_min >= len(df_tmp)-nchs_edge or idx_min <= nchs_edge:
            flags_apex[s]=0
        
        times_apex[s] = df_tmp.iloc[idx_min][label_time]
        chs_apex[s] = int(df_tmp.iloc[idx_min][label_channel])
        
    df_apex = pd.DataFrame({"shotnumber":shotnums, labels_out[0]:chs_apex, labels_out[1]:times_apex, 
                            "flag_apex":flags_apex})
    
    return df_apex
    


def add_offset2dfpicks(df, df_apex, label_channel="channel_idx", label_apex_channel="channel_idx_apex"):
    """ add" offset from apex to dataframe with picks"""
    
    shots_apex = np.unique(df_apex.shotnumber)
    count=0
    for s, shot in enumerate(np.unique(df.shotnumber)):
        
        if shot in shots_apex:
            df_tmp = df[df["shotnumber"]==shot].reset_index(drop=True)
            #info_apex = df_apex[df_apex["shotnumber"]==shot].iloc[0]
            
            offset_channels = df_tmp[label_channel] - df_apex.loc[df_apex["shotnumber"]==shot, label_apex_channel].item()
            df_tmp["offset_channel"] = offset_channels
            df_tmp["offset_channel_abs"] = np.abs(offset_channels)
            if len(df_tmp)>0:
                if count==0: 
                    df_pro=df_tmp.copy()
                else: 
                    df_pro = pd.concat([df_pro,df_tmp])
                count+=1
        else:   print(f"WARNING!! Shot {shot} not in df_apex")
            
    return df_pro

def gen_uncertainty_offset(lims_linear=(0,400), dec_chs=6, offset_max=250, lims_unc=(0.003,0.015), label_offset="offset_channel_abs", 
                           offset_crit_angle=130, unc_crit=0.005):
        """ generate df with uncertainty as function of offset"""    
        
        lims_linear = [round(val/dec_chs) for val in lims_linear ]
        
        offsets_chs_all = np.arange(0,round(offset_max/dec_chs)+1) 
        df = pd.DataFrame({label_offset:offsets_chs_all, "uncertainty":np.ones(len(offsets_chs_all))*lims_unc[1]})
        
        offsets_tmp = np.arange(lims_linear[0],lims_linear[1]+1)
        uncertainties = np.linspace(lims_unc[0], lims_unc[1], len(offsets_tmp)) #+ sigma_noise
        df_tmp = pd.DataFrame({label_offset:offsets_tmp, "uncertainty":uncertainties})
        
        df =  pd.concat([df,df_tmp]).drop_duplicates(subset=label_offset, keep="last").sort_values(\
            by=label_offset).reset_index(drop=True)

        if offset_crit_angle and unc_crit:
            df.loc[df["offset_channel_abs"]> round(offset_crit_angle/dec_chs), "uncertainty"] += unc_crit
        
        return df
            


def create_weight_mat(d_weights, flatten=True):
    """ create weight matrix of uncorrelatated data weights  """
    if d_weights.ndim>1 and flatten: 
        weights = d_weights.flatten()
    else:   weights = d_weights
    N = len(weights)
    
    W = np.zeros([N,N])
    for i in range(N):
        W[i,i]=weights[i]
    return W

def fd_matrix(nparams):
    """
    Create the finite difference matrix for regularization.
    """
    fdmatrix = np.zeros((nparams - 1, nparams))
    for i in range(fdmatrix.shape[0]):
        fdmatrix[i, i] = -1
        fdmatrix[i, i + 1] = 1
    return fdmatrix
    
def fd_matrix_square(nparams):
    """
    Create second order finite difference matrix for regularization.
    """
    fdmatrix = np.zeros((nparams - 1, nparams))
    for i in range(fdmatrix.shape[0]):
        fdmatrix[i, i] = -1
        fdmatrix[i, i + 1] = 1
    return fdmatrix.T @ fdmatrix

def fdmat_xy_sec(n):
    """ create second order regularization matrix for x,y"""
    nparams = 2* n 
    
    fdmat = fd_matrix_square(n)
    fdmat_all = np.zeros([nparams,nparams])
    fdmat_all[0:n,0:n] = fdmat
    fdmat_all[n:2*n,n:2*n] = fdmat
    
    return fdmat_all


def fdmat_xy(n):
    """ create regularization matrix for x,y"""
    nparams = 2* n 
    fdmat = fd_matrix(n)
    fdmat_all = np.zeros([nparams,nparams])
    fdmat_all[0:n,0:n] = fdmat
    fdmat_all[n:2*n,n:2*n] = fdmat
    
    return fdmat_all
    

def smooth_params_gauss(params, n_rec, sigma): 
    """ smooth the model parameters """
    return np.concatenate( [gaussian_filter1d(params[q*n_rec : q*n_rec+n_rec], sigma=sigma) for q in range(2)] )


def get_time_shifts_shots(tshift_paras, delta_ts, use_mean=True): 
    """ compute the total time shift for each shot using a linear time shift function"""
    if tshift_paras.ndim==1:
        tshift_tmp = np.zeros([1,tshift_paras.shape[0]])
        tshift_tmp[0,:]=tshift_paras
        tshift_paras=tshift_tmp
    
    tshifts_shots=np.zeros((tshift_paras.shape[0], delta_ts.shape[0])) 
    for s, delta_t in enumerate(delta_ts):
        tshifts_shots[:,s] = tshift_paras[:,0] + tshift_paras[:,1]* delta_t
    if use_mean: 
        return  np.mean(tshifts_shots, axis=1)
    else:
        return  tshifts_shots


### plotting functions

def plot_hessian_iter(hessian, hessian_inv, grad, delta_paras, title=None):
    """ plot hessian, gradient and delta paras"""
    fig, axes = plt.subplots(2,2, figsize=(10,8))
    ax00 = axes[0,0]; ax01 = axes[0,1]; 
    mesh = ax00.imshow(hessian)
    plt.colorbar(mesh, ax=ax00, location='right', fraction=0.025)
    ax00.set(xlabel='m', ylabel='m', title=f"hessian")
    mesh=ax01.imshow(hessian_inv)
    plt.colorbar(mesh, ax=ax01, location='right', fraction=0.025)
    ax01.set(xlabel='m',ylabel='m' , title=f"hessian_inv")
    
    ax10 = axes[1,0]; ax11 = axes[1,1]
    ax10.plot(grad)
    ax10.set(xlabel='m', title="gradient")
    ax11.plot(delta_paras)
    ax11.set(xlabel='m', title="delta parameters")
    fig.suptitle(title)
    plt.show()
    return 




def plot_inv_iter(df_cable, df_shots, df_data, misfits, tshifts, iters_plot=(2,12), tshift_paras_true=None, 
                  df_cable_init=None, df_cable_orig=None, shots_plot=None, label_shots=True, xlims_chs=None,
                  rec_init=True, rec_true=True, rec_iter=True, title=None, shot_interval_plot=5, xlims_map=(-300, 800), 
                  y_center=-200, cbar=True, ylims_tt=(0,0.5), ylims_tshift=None, ylims_htau=None, ylims_misfit=None, 
                  figname=None, label_subs=False, shot_interval_annotate=5, figsize=(10,10), dpi=150, 
                  plot_channels_ref=True, channels_ref=None, interval_chs_ref=500, xlabel_tt="channel_idx", 
                  show_fig=True, ylabel_misfit="Weighted RMSE", yscale_misfit="linear", 
                  ylabel_tt=r"Traveltime + Time Shift [s]", plot_inset_zoom=True, paras_inset=None, 
                  file_format='png', kargs_subfig_labels=None, width_ratios=(1,1), height_ratios=(1.75,1.75, 1.5)):
    """ plot results of cable inversion, including positions, and data misfit"""
    
    # prep paras
    plotparas = {"cable_true":{"linestyle":'-', "linewidth":5.5, "c":"gray",
                               "label":r'$cable_{true}$', "zorder":2, "alpha":0.75}, 
                 "cable_init":{"linestyle":'-', "linewidth":2, "c":"k",
                                            "label":r'$cable_{init}$', "zorder":1, "alpha":1.0}, 
                 "cable_iter":{"linestyle":'-', "zorder":3, "alpha":1.0}, 
                 "cable_orig":{"linestyle":'-', "linewidth":3.5, "c":"gray", 
                                             "label":r'$cable_{orig}$', "zorder":1, "alpha":0.75}, 
                 "shots":{"marker":'*', "c":"r", "s":12,
                                             "label":r'source', "zorder":4, "alpha":0.75},
                 "channels_ref_iter":{"marker":'X', "c":"k", "s":50, "edgecolor":None,
                                            "zorder":7, "alpha":1.0}, 
                 "channels_ref_true":{"marker":'s', "c":"w", "s":110, "edgecolor":'k', "linewidths":1.5,
                                            "zorder":6, "alpha":0.75}
                 }
    
    if plot_inset_zoom:
        paras_inset_default = {"orig":(0,0), "dxdy":(150,150), "width":0.45, "xy_anker":(0.02,0.02) }
        if paras_inset is not None:
            paras_inset_default.update(paras_inset)
        paras_inset = paras_inset_default
    
    #height_ratios=(1.75,1.75, 1.5)
     
    colors_cable=('darkorange', 'red') if len(iters_plot)==2 else ("red")
    ax_ratio_map = utils_plot.det_ax_ratio_map(figsize, width_ratios,
                                                  height_ratios=height_ratios, col_idx=0, row_idxes=[0,1])
    #diff_x = xlims_map[1]-xlims_map[0]
    diff_y = utils_plot.get_map_lims_by_ratio(xlims_map, ax_coords='x', ax_ratio=ax_ratio_map)
    ylims_map = (y_center-diff_y/2,y_center+diff_y/2)
    
     
    fig = plt.figure(figsize=figsize, layout='constrained', dpi=dpi)
    gs = GridSpec(3,2, figure=fig, width_ratios=width_ratios, height_ratios=height_ratios, hspace=0.005)
    ax00 = fig.add_subplot(gs[0:2,0]) #
    ax01 = fig.add_subplot(gs[0,1])
    ax02 = fig.add_subplot(gs[1,1])
    #ax01 = fig.add_subplot(gs[0,1]) #
    ax1 = fig.add_subplot(gs[2,:]) # 
    axes_tts = [ax1]
    axes_all = [ax00,ax01,ax1]
  
    if tshifts is not None:
        if tshifts.ndim==2: 
            tshifts_plot = tshifts[:,0]
        axes_all.insert(2, ax02)
    
    ## plot map
    list_df_cables=[]; labels_cables=[]
    if df_cable_orig is not None and rec_true==False:
        ax00.plot(df_cable_orig.UTM_X, df_cable_orig.UTM_Y, **plotparas["cable_orig"])
        #linestyle='-', linewidth=3.5, c='purple', label=r'$cable_{orig}$', zorder=3, alpha=0.6
    
    if df_cable_init is not None and rec_init:
        ax00.plot(df_cable_init["UTM_X_init"], df_cable_init["UTM_Y_init"], **plotparas["cable_init"])
        #linestyle='-', linewidth=2, c='k', label=r'$cable_{init}$', zorder=3
        list_df_cables.append(df_cable_init); labels_cables.append("init")
        
    if df_cable_init is not None and rec_true: 
        ax00.plot(df_cable_init["UTM_X_true"], df_cable_init["UTM_Y_true"],  **plotparas["cable_true"]) #tab:blue
        #linestyle='-', linewidth=3.5, c='gray', label=r'$cable_{true}$', zorder=3, alpha=0.75
        #path_effects=[mpl_pe.Stroke(linewidth=6.5, foreground='k'), mpl_pe.Normal()]
        list_df_cables.append(df_cable_init); labels_cables.append("true")
        # ax00.scatter(xy_coords[0], xy_coords[1], c='k', marker="X", s=80)
    
    if rec_iter:
        linewidths_iter=(1.5,2.5)
        for i, iterx in enumerate(iters_plot):
            df_tmp = df_cable[df_cable["iter"]==iterx]
            ax00.plot(df_tmp["UTM_X"], df_tmp["UTM_Y"], c=colors_cable[i], label=f'cable i={iterx}',
                      **plotparas["cable_iter"], linewidth=linewidths_iter[i]  ); 
            list_df_cables.append(df_tmp); labels_cables.append("iter")
            # xy_coords= df_tmp.loc[df_tmp["channel_abs"]==CHANNEL_CROSSING, ["UTM_X", "UTM_Y"] ].values.T
            # ax00.scatter(xy_coords[0], xy_coords[1], c='k', marker="X", s=80)
    
    # plot reference channels
    if plot_channels_ref:
        labels_chs=("A","B","C") #markersizes=(20,35,20);
        dx_tmp=45; 
        if channels_ref is None:
            channels_ref = np.array([ CHANNEL_REF + fac*(int(interval_chs_ref/DEC_CHS) ) for fac in (-1,0,1)], dtype='int32' ) 
        #print(f"DEBUG: channels_ref: {channels_ref}")
        xy_chs_init=[]; count_iter=0
        for k, (label,df) in enumerate( zip(labels_cables,list_df_cables) ):
            if label=="iter":
                labelx, labely = "UTM_X", "UTM_Y"; count_iter+=1
            else:
                labelx, labely = [coord +"_" + label for coord in ("UTM_X","UTM_Y")]
                
            if label=="true":
                #color = 'w'; marker='s'; edgecolor='k'; markersize=70
                plotparas_tmp = plotparas["channels_ref_true"]
            else: 
                plotparas_tmp = plotparas["channels_ref_iter"]
            # else:
            #     color = 'k'; marker='X'; edgecolor=None; markersize=50
            
            for c,ch in enumerate(channels_ref): 
                label_leg=r"$channel_{ref}$" if (k==0 and c==0) else None 
                xy_coords= df.loc[df["channel_idx"]==ch, [labelx, labely] ].values.T
                ax00.scatter(xy_coords[0], xy_coords[1], label=label_leg, **plotparas_tmp)
                
                if label=='init':
                    xy_chs_init.append(xy_coords)
                
                if (rec_true==True and label=="true") or (rec_true==False and label=='iter' and count_iter<=1):
                        if len(xy_chs_init)>0:
                            if xy_coords[0] > xy_chs_init[c][0]:
                                ha='left'; fac=1; 
                            else: 
                                ha='right'; fac=-1
                            #if label=='iter' and count_iter<=1:
                            ax00.annotate(labels_chs[c], (xy_coords[0]+fac*dx_tmp,xy_coords[1]), ha=ha, 
                                          fontsize=11, fontweight='bold', zorder=7)
        
    # plot shots
    ax00.scatter(df_shots["UTM_X"], df_shots["UTM_Y"], **plotparas["shots"])
    for s, info_shot in df_shots.iterrows():
        if info_shot.shotnumber % 2 == 0: 
            ha='left'; va='bottom'
        else: 
            ha='right'; va='top'
        #ax00.scatter(info_shot.UTM_X, info_shot.UTM_Y, marker='*', c='r', s=40)
        
    # annotate shots
    if shots_plot is not None: 
        df_shots_annotate = df_shots[df_shots["shotnumber"].isin(shots_plot)]
    elif shot_interval_annotate>0:
        df_shots_annotate = df_shots.iloc[::shot_interval_annotate]
    else:   df_shots_annotate=pd.DataFrame()
        
    dx_tmp, dy_tmp = 10, 10
    for _, info_shot in df_shots_annotate.iterrows():
        if info_shot.shotnumber % 2 == 0: 
            ha='left'; va='bottom'; fac=1
        else: 
            ha='right'; va='top'; fac=-1
        ax00.annotate(f"{int(info_shot.shotnumber)}", 
                          (info_shot.UTM_X+fac*dx_tmp, info_shot.UTM_Y+fac*dy_tmp), c='r', zorder=5,
                          ha=ha, va=va , fontsize=8, bbox=dict(boxstyle="round4", fc="w", ec="k", pad=0.3))
    
    ax00.set(xlabel="X [m]", ylabel="Y [m]", title=title, xlim=xlims_map, ylim=ylims_map)
    ax00.legend(loc="upper right")
    
    if plot_inset_zoom:
        fac_linewidth = 1.5
        fac_markersize=1.3
        xlims_inset, ylims_inset = [ tuple([ paras_inset["orig"][idx] + fac*paras_inset["dxdy"][idx] for fac in (-1,1) ]\
                                           ) for idx in (0,1)]
        diffx_tmp, diffy_tmp = [arr[1]-arr[0] for arr in (xlims_inset, ylims_inset)]
        axratio_inset = diffx_tmp/diffy_tmp
        height_inset = paras_inset["width"] *ax_ratio_map * (1/axratio_inset)
        axins = ax00.inset_axes( [paras_inset["xy_anker"][0], paras_inset["xy_anker"][1], paras_inset["width"], height_inset], \
                                xlim=xlims_inset, ylim=ylims_inset, \
                                  xticklabels=[], yticklabels=[] , xticks=[], yticks=[] ) #xticklabels=[], yticklabels=[]  # zoom = 6
        if rec_true:
            axins.plot(df_cable_init["UTM_X_true"].values,df_cable_init["UTM_Y_true"].values, 
                       linewidth=plotparas["cable_true"]["linewidth"]*fac_linewidth, \
                       **utils.without_keys(plotparas["cable_true"], ("linewidth","label") ) )
        elif df_cable_orig is not None and rec_true==False:
            axins.plot(df_cable_orig.UTM_X, df_cable_orig.UTM_Y, linewidth=plotparas["cable_orig"]["linewidth"]*fac_linewidth,\
            **utils.without_keys(plotparas["cable_orig"], ("linewidth","label") ))
        if rec_init:
            axins.plot(df_cable_init["UTM_X_init"].values,df_cable_init["UTM_Y_init"].values, 
                       linewidth=plotparas["cable_init"]["linewidth"]*fac_linewidth, 
                       **utils.without_keys(plotparas["cable_init"], ("linewidth","label") ))
        if rec_iter:
            for i, iterx in enumerate(iters_plot):
                df_tmp = df_cable[df_cable["iter"]==iterx]
                axins.plot(df_tmp["UTM_X"], df_tmp["UTM_Y"], linewidth=linewidths_iter[i]*fac_linewidth,\
                 c=colors_cable[i], **utils.without_keys(plotparas["cable_iter"], ("linewidth")) ) 
        
        axins.scatter(df_shots["UTM_X"],df_shots["UTM_Y"],  s=plotparas["shots"]["s"]*fac_markersize,\
                      **utils.without_keys(plotparas["shots"], ("s")) )
        if plot_channels_ref:
            idx_tmp=1; dx_tmp=35; 
            xy_chs_init=[]; count_iter=0
            for k, (label,df) in enumerate( zip(labels_cables,list_df_cables) ):
                if label=="iter":
                    labelx, labely = "UTM_X", "UTM_Y"; count_iter+=1
                else:
                    labelx, labely = [coord +"_" + label for coord in ("UTM_X","UTM_Y")] 
                if label=="true":
                    #color = 'w'; marker='s'; edgecolor='k'; markersize=75 #50
                    plotparas_tmp = plotparas["channels_ref_true"]
                    plotparas_tmp.update({"s":plotparas_tmp["s"]*fac_markersize})
                else:
                    plotparas_tmp = plotparas["channels_ref_iter"]
                    plotparas_tmp.update({"s":plotparas_tmp["s"]*fac_markersize})
                    #color = 'k'; marker='X'; edgecolor=None; markersize=50 #35
                    
                xy_coords= df.loc[df["channel_idx"]==channels_ref[idx_tmp], [labelx, labely] ].values.T
                axins.scatter(xy_coords[0], xy_coords[1], **plotparas_tmp) #label=label_leg
                if label=='init':
                    xy_chs_init=xy_coords
                if (rec_true==True and label=="true") or(rec_true==False and label=='iter' and count_iter>1) :
                        if len(xy_chs_init)>0:
                            if xy_coords[0] > xy_chs_init[0]:
                                ha='left'; fac=1; 
                            else: 
                                ha='right'; fac=-1
                            axins.annotate(labels_chs[idx_tmp], (xy_coords[0]+fac*dx_tmp,xy_coords[1]), ha=ha, 
                                          fontsize=12, fontweight='bold', zorder=6)
        ax00.indicate_inset_zoom(axins, edgecolor="black", linewidth=2)
    
    
    ## plot misfit and time shift
    idxes_iter = np.arange(0,max(iters_plot)+1); 
    #niter=len(idxes_iter)
    ax01.axhline(1, c='gray', label=r"$\epsilon_{target}$", linewidth=4, zorder=1, alpha=0.75)
    ax01.plot(idxes_iter, misfits[idxes_iter], '-o', markersize=3, zorder=2)
    ax01.set(ylabel=ylabel_misfit, yscale=yscale_misfit, ylim=ylims_misfit)
    ax01.legend(loc="upper right")
    if tshifts is None:
        ax02.axis("off")
    else:
        if tshifts.ndim==2:
            label_tshift = r"$\tau_0$"; label_tshift_true=r"$\tau_{0,true}$"
        else:
            label_tshift = r"$\tau$"; label_tshift_true=r"$\tau_{true}$"
            
            
       # label_tshift = r"$\tau_0$" if tshifts.ndim==2 else r"$\tau$"
        ax02.plot(idxes_iter, tshifts_plot[idxes_iter], '-o', c='darkblue', markersize=3, 
                       zorder=2, label=label_tshift )
        if tshift_paras_true is not None:
            label = label_tshift_true if tshifts.ndim==1 else None
            ax02.axhline(tshift_paras_true[0], c='darkblue', label=label, linewidth=1.5, 
                         linestyle='--', zorder=1,  alpha=0.75)
            #loc_leg = "lower right" if tshift_paras_true[0]>=0 else "upper right"
       
        
        if tshifts.ndim==2: 
            fac_ax_htau = 1.1 if rec_true else 1.25
            ylims_htau = (None, tshifts[idxes_iter,1].max()*fac_ax_htau) if ylims_htau is None else ylims_htau
            
            ax022 = ax02.twinx()
           
            #print(f"DEBUG: idxes_iter={idxes_iter}, mus={tshifts[[idxes_iter],1]}")
            if tshift_paras_true is not None: 
                ax022.axhline(tshift_paras_true[1], c='red', linestyle='--', linewidth=1.5) #label=r"$h_{\tau,true}$"
            ax022.plot(idxes_iter, tshifts[idxes_iter,1], '-o', markersize=3, zorder=2, 
                       c='red', label=r"$h_{\tau}$"), 
            ax022.set(ylabel=r"$h_{\tau}$", ylim=ylims_htau )
            ax022.tick_params(axis='y', labelcolor="red")
            
            tshifts_total = get_time_shifts_shots(tshifts[idxes_iter,:], df_shots["delta_t"], use_mean=True)
            if tshift_paras_true is not None:
                tshift_total_true = get_time_shifts_shots(tshift_paras_true, df_shots["delta_t"], use_mean=True)
                ax02.axhline(tshift_total_true, c='k', linestyle='--', zorder=1, linewidth=1.5)
        
            ax02.plot(idxes_iter, tshifts_total, '-o', c='k', markersize=2.5, linewidth=1, label=r"$\tau$")
            l1, labs1 = ax02.get_legend_handles_labels()
            l2, labs2 = ax022.get_legend_handles_labels()
            
            ax02.legend(l1+l2, labs1+labs2, ncols=2, loc="lower right" )
            ax02.tick_params(axis='y', labelcolor="darkblue")
        
        else:
            ax02.legend()
        ax02.set(xlabel="Iteration", ylabel=r"Time Shift $\tau$ [s]", ylim=ylims_tshift)
            
        
   
    
    ## plot traveltimes
    
    if shots_plot is not None: 
       shotnums_plot = np.array(shots_plot)
    elif shot_interval_plot is not None:  
        shotnums_plot =  np.unique(df_data["shotnumber"])
    shotlims_plot = (shotnums_plot.min(), shotnums_plot.max() )
        
    #df_data_plot = df_data[df_data["shotnumber"].isin(shots_plot)]
    norm_cmap = mpl.colors.Normalize(vmin=shotlims_plot[0], vmax=shotlims_plot[1])
    cmap = mpl.cm.ScalarMappable(norm=norm_cmap, cmap=mpl.cm.jet)
    cmap.set_array([])
    
    niters_plot = len(iters_plot)
    if niters_plot==1:
        alphas_iter=(1,); zorders=(3,)
    else:
        alphas_iter=(0.3, 1); zorders=(2,3)
    #shotnum_max_rel = shotnums.max() - shotnums.min()
    
    if tshifts is not None: 
        ax1.axhline(tshifts_total[-1], c="gray", linewidth=3, alpha=0.6, zorder=0)
    
    for i,iterx in enumerate(iters_plot):
        
    
        for s, shotnum in enumerate(shotnums_plot):
            color = cmap.to_rgba(shotnum)  #cmap((shotnum-shotnums.min()) / shotnum_max_rel)
            if i+1==len(iters_plot):
                edgecolor = 'k'; linewidth_edge=0.4
            else:   edgecolor=color; linewidth_edge=None
                
            df_data_tmp = df_data[ (df_data["shotnumber"]==shotnum) & (df_data["iter"]==iterx)] 
            if i==0:
                ax1.errorbar(df_data_tmp[xlabel_tt], df_data_tmp["traveltime"], yerr=df_data_tmp["uncertainty"], fmt='o', 
                             c=color, ecolor=color, markersize=0.6, alpha=0.7, zorder=1)
                if label_shots: 
                    if rec_true:
                        data_smooth = gaussian_filter1d(df_data_tmp["traveltime"].values, 5)
                        chidx_min = np.argmin(data_smooth)
                        ch_apex, time_apex = [ df_data_tmp.iloc[chidx_min][label] for label in (xlabel_tt,"traveltime")]
                    else:    
                        info_shot = df_shots.loc[df_shots["shotnumber"]==shotnum].iloc[0]
                        ch_apex, time_apex = int(info_shot["channel_idx_apex"]), info_shot["time_apex"]
                        #ch_apex = int(ch_apex/DEC_CHS) if xlabel_tt=="channel_idx" else ch_apex
                        #print(f"DEBUG: chapex={ch_apex}, time_apex={time_apex}")
                    
                    ax1.scatter(ch_apex, time_apex, marker='*', c='k', zorder=5, s=40)
                    ax1.annotate(shotnum,(ch_apex-3, time_apex-0.005), fontsize=8, va='bottom', zorder=4, 
                                 bbox=dict(boxstyle="round4", fc="w", ec="k", pad=0.2))
                    
            ax1.scatter(df_data_tmp[xlabel_tt], df_data_tmp["traveltime_pred"], color=color, s=7.0, \
                     alpha=alphas_iter[i], label=shotnum, zorder=zorders[i], edgecolors=edgecolor, \
                     linewidths=linewidth_edge)
            # residuals
    if plot_channels_ref:
        #chs_label = (channels_ref/DEC_CHS).astype('int32') if xlabel_tt=="channel_idx" else channels_ref
        ax1.scatter(channels_ref, np.ones(len(channels_ref))*ylims_tt[1]-0.0185*np.diff(ylims_tt), marker='X', c='k', 
                    s=50, zorder=5 )
        for c,ch in enumerate(channels_ref):
            ax1.annotate(labels_chs[c], (ch, ylims_tt[1]-0.021*np.diff(ylims_tt) ), fontsize=12, 
                         fontweight='bold', zorder=6, va='bottom', ha='left')
    if cbar:
        #sm = plt.cm.ScalarMappable(cmap=cmap) #norm=plt.Normalize(vmin=vmin_loss, vmax=vmax_loss) #
        plt.colorbar(cmap, ax=ax1,  location='right', fraction=0.05, pad=0.025, label="shot number") 
    xlabel_plot = "Channel Index" if xlabel_tt=="channel_idx" else "Channel"
    ax1.set(xlabel=xlabel_plot, ylabel=ylabel_tt, ylim=ylims_tt)
    if xlims_chs is not None: 
        _ = [ ax.set(xlim=xlims_chs) for ax in axes_tts]
        
    ax1.invert_yaxis()
    _ = [ ax.invert_xaxis() for ax in axes_tts ]
    if label_subs: 
        utils_plot.label_subfigs(axes_all, 0.025, 0.975, **kargs_subfig_labels)
    
    if figname: 
        fig.savefig(figname + f'.{file_format}', format=file_format, dpi=dpi, bbox_inches='tight' )
    if show_fig:
        plt.show()
    return 




def plot_residuals_data(df_data, df_shots, tshifts, iters_plot=(2,12), tshift_paras_true=None, 
                   shots_plot=None, label_shots=True, xlims_chs=None,
                  rec_true=True, title=None, shot_interval_plot=5, 
                  cbar=False, ylims=(-0.05,0.05),
                  figname=None, figsize=(10,5), dpi=150, 
                  plot_channels_ref=True, channels_ref=None, interval_chs_ref=500, xlabel_tt="channel_idx", 
                  show_fig=True, 
                  ylabel_tt=r"Residual Time [s]", 
                  file_format='png'):
    
    """ plot results data residuals """
    
    # prep paras
    # plotparas = {"cable_true":{"linestyle":'-', "linewidth":5.5, "c":"gray",
    #                            "label":r'$cable_{true}$', "zorder":2, "alpha":0.75}, 
    #              "cable_init":{"linestyle":'-', "linewidth":2, "c":"k",
    #                                         "label":r'$cable_{init}$', "zorder":1, "alpha":1.0}, 
    #              "cable_iter":{"linestyle":'-', "zorder":3, "alpha":1.0}, 
    #              "cable_orig":{"linestyle":'-', "linewidth":3.5, "c":"gray", 
    #                                          "label":r'$cable_{orig}$', "zorder":1, "alpha":0.75}, 
    #              "shots":{"marker":'*', "c":"r", "s":12,
    #                                          "label":r'source', "zorder":4, "alpha":0.75},
    #              "channels_ref_iter":{"marker":'X', "c":"k", "s":50, "edgecolor":None,
    #                                         "zorder":7, "alpha":1.0}, 
    #              "channels_ref_true":{"marker":'s', "c":"w", "s":110, "edgecolor":'k', "linewidths":1.5,
    #                                         "zorder":6, "alpha":0.75}
    #              }
    
    
    iters = np.unique(df_data["iter"])
    idxes_iter = [np.where(iters==it)[0][0] for it in iters_plot]
     
    fig, ax1 = plt.subplots(1,1, dpi=dpi, figsize=figsize)
  
    if tshifts is not None:
        if tshifts.ndim==2: 
            #tshifts_plot = tshifts[:,0]
            
            tshifts_total = get_time_shifts_shots(tshifts[idxes_iter,:], df_shots["delta_t"], use_mean=True)
            if tshift_paras_true is not None:
                tshift_total_true = get_time_shifts_shots(tshift_paras_true, df_shots["delta_t"], use_mean=True)
    
    # plot reference channels
    if plot_channels_ref:
        labels_chs=("A","B","C") #markersizes=(20,35,20);
        if channels_ref is None:
            channels_ref = np.array([ CHANNEL_REF + fac*(int(interval_chs_ref/DEC_CHS) ) for fac in (-1,0,1)], dtype='int32' ) 
       
    if shots_plot is not None: 
       shotnums_plot = np.array(shots_plot)
    elif shot_interval_plot is not None:  
        shotnums_plot =  np.unique(df_data["shotnumber"])
    shotlims_plot = (shotnums_plot.min(), shotnums_plot.max() )
        
    #df_data_plot = df_data[df_data["shotnumber"].isin(shots_plot)]
    norm_cmap = mpl.colors.Normalize(vmin=shotlims_plot[0], vmax=shotlims_plot[1])
    cmap = mpl.cm.ScalarMappable(norm=norm_cmap, cmap=mpl.cm.jet)
    cmap.set_array([])
    
    niters_plot = len(iters_plot)
    if niters_plot==1:
        alphas_iter=(1,); zorders=(3,)
    else:
        alphas_iter=(0.3, 1); zorders=(2,3)
    #shotnum_max_rel = shotnums.max() - shotnums.min()
    
    # if tshifts is not None: 
    #     ax1.axhline(tshifts_total[-1], c="gray", linewidth=3, alpha=0.6, zorder=0)
    
    for i,iterx in enumerate(iters_plot):
        
    
        for s, shotnum in enumerate(shotnums_plot):
            color = cmap.to_rgba(shotnum)  #cmap((shotnum-shotnums.min()) / shotnum_max_rel)
            if i+1==len(iters_plot):
                edgecolor = 'k'; linewidth_edge=0.4
            else:   edgecolor=color; linewidth_edge=None
                
            df_data_tmp = df_data[ (df_data["shotnumber"]==shotnum) & (df_data["iter"]==iterx)] 
            if i==0:
                _ = [ ax1.plot(df_data_tmp[xlabel_tt], fac*df_data_tmp["uncertainty"], linewidth=1.5,
                             c=color, alpha=0.7, zorder=1) for fac in (-1,1)]
                if label_shots: 
                    if rec_true:
                        data_smooth = gaussian_filter1d(df_data_tmp["traveltime"].values, 5)
                        chidx_min = np.argmin(data_smooth)
                        ch_apex, time_apex = [ df_data_tmp.iloc[chidx_min][label] for label in (xlabel_tt,"traveltime")]
                    else:    
                        info_shot = df_shots.loc[df_shots["shotnumber"]==shotnum].iloc[0]
                        ch_apex, time_apex = int(info_shot["channel_idx_apex"]), info_shot["time_apex"]
                        #ch_apex = int(ch_apex/DEC_CHS) if xlabel_tt=="channel_idx" else ch_apex
                        #print(f"DEBUG: chapex={ch_apex}, time_apex={time_apex}")
                    
                    ax1.scatter(ch_apex, time_apex, marker='*', c='k', zorder=5, s=40)
                    ax1.annotate(shotnum,(ch_apex-3, time_apex-0.005), fontsize=8, va='bottom', zorder=4, 
                                 bbox=dict(boxstyle="round4", fc="w", ec="k", pad=0.2))
                    
            ax1.scatter(df_data_tmp[xlabel_tt], df_data_tmp["res_time"], color=color, s=7.0, \
                     alpha=alphas_iter[i], label=shotnum, zorder=zorders[i], edgecolors=edgecolor, \
                     linewidths=linewidth_edge)
            # residuals
    if plot_channels_ref:
        #chs_label = (channels_ref/DEC_CHS).astype('int32') if xlabel_tt=="channel_idx" else channels_ref
        ax1.scatter(channels_ref, np.ones(len(channels_ref))*ylims[1]-0.0185*np.diff(ylims), marker='X', c='k', 
                    s=50, zorder=5 )
        for c,ch in enumerate(channels_ref):
            ax1.annotate(labels_chs[c], (ch, ylims[1]-0.021*np.diff(ylims) ), fontsize=12, 
                         fontweight='bold', zorder=6, va='bottom', ha='left')
    if cbar:
        #sm = plt.cm.ScalarMappable(cmap=cmap) #norm=plt.Normalize(vmin=vmin_loss, vmax=vmax_loss) #
        plt.colorbar(cmap, ax=ax1,  location='right', fraction=0.05, pad=0.025, label="Shot Number") 
    xlabel_plot = "Channel Index" if xlabel_tt=="channel_idx" else "Channel"
    ax1.set(xlabel=xlabel_plot, ylabel=ylabel_tt, xlim=xlims_chs ,ylim=ylims)
    ax1.invert_yaxis(); ax1.invert_xaxis()
    # if label_subs: 
    #     utils_plot.label_subfigs(axes_all, 0.025, 0.975, **kargs_subfig_labels)
    if figname: 
        fig.savefig(figname + f'.{file_format}', format=file_format, dpi=dpi, bbox_inches='tight' )
    if show_fig:
        plt.show()
    return 


## OBSOLET

       
def hessian(x_rec, y_rec, coords_src, z_src, v=1500):
    """create full Hessian """
    
    xy_src = np.array(coords_src)
    
    tex_start=time.perf_counter()
    
    nparams = 2*x_rec.shape[0]
    H = np.zeros([nparams,nparams])
    
    #populate d2/dx2
    ddx2 = _ddm2_coord(x_rec, y_rec, xy_src, z_src, v=v )
    ddy2 = _ddm2_coord(y_rec, x_rec, np.flip(xy_src), z_src, v=v )
    
    for i in range(nparams):
        H[i,i] = ddx2[i]
        H[i+nparams,i+nparams] = ddy2[i]
    
    
    # populate cross derivatives dxdy
    dxdy = _dxdy_coord(x_rec, y_rec, xy_src, z_src, v=v)
    dydx = _dxdy_coord(y_rec, x_rec , np.flip(xy_src), z_src, v=v)
    for i in range(nparams):
        H[i,i+nparams] = dxdy[i]
        H[i+nparams,i] = dydx[i]
        
    print("Hessian constructed in {:.3f}s".format(time.perf_counter()-tex_start) )
    
    return H

