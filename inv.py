#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  4 20:57:38 2025

@author: keving
"""

import time, sys
import numpy as np, pandas as pd

from Functions import utils, utils_pd
import utils_cable_inv
from utils_cable_inv import model2paras, paras2model, data2df, df_from_data, forward_multi, \
        lsqr_misfit_model, TIMESTAMP_FIRST_SHOT_LINE2 #datavec2data


def jacobian_timeshift(delta_ts, bools_rec ):
    """ jacobian of timeshift model with two parameters (linear function)"""
    nshots = bools_rec.shape[0]
    assert len(delta_ts)==nshots
    ndata = sum(bools_rec.flatten().astype('int32'))
    
    J = np.ones([ndata,2]);  counter=0
    for s in range(nshots):
        bools_tmp = bools_rec[s,:]
        ndata_shot = sum(bools_tmp.astype('int32'))
        J[counter:counter+ndata_shot,1] = delta_ts[s]
        counter += ndata_shot
        
    return J

def jacobian_timeshift_ttsq(parameters, paras_tshift, bools_rec, x_src, y_src, z_src, delta_ts,
                            vel_water=1500 ):
    """ jacobian of timeshift model for squared forward equation"""
    
    nparams = len(parameters); assert nparams%2==0
    n_rec = int(nparams/2); 
    x_rec, y_rec = paras2model(parameters, n_rec)
    
    nshots = bools_rec.shape[0]
    assert len(delta_ts)==nshots
    ndata = sum(bools_rec.flatten().astype('int32'))
    
    J = np.zeros([ndata,2]);  counter=0
    for s in range(nshots):
        bools_tmp = bools_rec[s,:]
        ndata_shot = sum(bools_tmp.astype('int32'))
        
        # fill in derivatives
        J[counter:counter+ndata_shot,0] = utils_cable_inv._ddm_ttsq_tau0(x_rec[bools_tmp], y_rec[bools_tmp], (x_src[s], y_src[s]),
                                                z_src, paras_tshift[0], paras_tshift[1], delta_ts[s], v=vel_water)
        J[counter:counter+ndata_shot,1] = utils_cable_inv._ddm_ttsq_ht(x_rec[bools_tmp], y_rec[bools_tmp], (x_src[s], y_src[s]), z_src, 
                                                paras_tshift[0], paras_tshift[1], delta_ts[s], v=vel_water)
        
        counter += ndata_shot
        
    return J

def ddm_jacobian_timeshift(parameters, paras_tshift, bools_rec, x_src, y_src, z_src, delta_ts, tt_squared=True,
                            vel_water=1500 ):
    """ derivatives of the timeshift jacobian"""
    
    nparams = len(parameters); assert nparams%2==0
    n_rec = int(nparams/2); 
    x_rec, y_rec = paras2model(parameters, n_rec)
    
    nshots = bools_rec.shape[0]
    assert len(delta_ts)==nshots
    ndata = sum(bools_rec.flatten().astype('int32'))
    
    J = np.zeros([ndata,2]);  counter=0
    if tt_squared==False: 
        return J
    for s in range(nshots):
        bools_tmp = bools_rec[s,:]
        ndata_shot = sum(bools_tmp.astype('int32'))
        delta_t = delta_ts[s]
        
        ddxddy  =  utils_cable_inv._dtau_dh_ttsq(ndata_shot, delta_t) # d/dx d/dy = d/dy d/dx
        for p in range(2):
            if p==0:
                ddm2 = utils_cable_inv._ddm2_ttsq_tau0(ndata_shot)
            else:
                ddm2 = utils_cable_inv._ddm2_ttsq_ht(ndata_shot, delta_t)
        
            # fill in derivatives
            J[counter:counter+ndata_shot,p] = ddm2 + ddxddy
           
        counter += ndata_shot
        
    return J

def jacobian_positions(parameters, bools_rec, x_src, y_src, z_src, tt_squared=False,
                        tshift_paras=None, delta_ts=None, 
                        record_time=True, verbose=False, vel_water=1500): 
    """ jacobian of the receiver coordinates"""
    
    nparams = len(parameters); assert nparams%2==0
    n_rec = int(nparams/2); 
    nshots = len(x_src); assert nshots==len(y_src)
    ndata = sum(bools_rec.flatten().astype('int32')) #n_rec * nshots
    
    x_rec, y_rec = paras2model(parameters, n_rec)
    
    if tt_squared:
        tau0, htau = tshift_paras[0], tshift_paras[1]
        assert delta_ts is not None, "! relative shot times (delta_t) need be provided" 
    
    tex_start = time.perf_counter() if record_time else None
    J = np.zeros([ndata,nparams]); ndata_filled=0
    for s in range(nshots):
        if verbose: print(f"s={s}")
        
        bools_rec_tmp = bools_rec[s,:]
        idxes_rec_used= np.where(bools_rec_tmp==True)[0]
        
        for k in range(2):   
            nparas_before = k*n_rec
            if k==0:
                 coords_src = ( x_src[s], y_src[s] )
                 coords_sec = parameters[n_rec::][bools_rec_tmp]
            else: 
                coords_src = (y_src[s], x_src[s] )
                coords_sec = parameters[0:n_rec][bools_rec_tmp]
            coords = parameters[k*n_rec:k*n_rec+n_rec][bools_rec_tmp]
            
            if tt_squared: 
                derivatives = utils_cable_inv._ddm_ttsq_coord(coords, coords_sec, coords_src,
                                                              z_src, tau0, htau, delta_ts[s], v=vel_water)
            else: 
               derivatives = utils_cable_inv._ddm_coord(coords, coords_sec, coords_src, z_src, v=vel_water)
               
            assert len(derivatives)==len(idxes_rec_used)
            for i, (idx_rec, ddm) in enumerate(zip( idxes_rec_used, derivatives) ):
                J[ndata_filled+i, idx_rec + nparas_before] = ddm
        
        ndata_filled += sum(bools_rec_tmp.astype(int))
    if record_time: 
        print("Jacobian built in {:.3f}s".format(time.perf_counter()-tex_start))
        
    return J


def ddm_jacobian_positions(parameters, tshift_paras, bools_rec, x_src, y_src, z_src, tt_squared=False, delta_ts=None,
                 record_time=True, verbose=False, vel_water=1500): 
    """ derivatives of the jacobian of the receiver coordinates"""
    
    nparams = len(parameters); assert nparams%2==0
    n_rec = int(nparams/2); 
    nshots = len(x_src); assert nshots==len(y_src)
    ndata = sum(bools_rec.flatten().astype('int32')) #n_rec * nshots
    
    if tt_squared: 
        assert delta_ts is not None and len(delta_ts)==nshots, "delta_ts for shots must be provided" 
    
    x_rec, y_rec = paras2model(parameters, n_rec)
    
    tex_start = time.perf_counter() if record_time else None
    J = np.zeros([ndata,nparams]); ndata_filled=0
    for s in range(nshots):
        if verbose: print(f"s={s}")
        
        bools_rec_tmp = bools_rec[s,:]
        idxes_rec_used= np.where(bools_rec_tmp==True)[0]
        
        
        for k in range(2):   
            nparas_before = k*n_rec
            if k==0:
                 coords_src = ( x_src[s], y_src[s] )
                 coords_sec = parameters[n_rec::][bools_rec_tmp]
            else: 
                coords_src = (y_src[s], x_src[s] )
                coords_sec = parameters[0:n_rec][bools_rec_tmp]
            coords = parameters[k*n_rec:k*n_rec+n_rec][bools_rec_tmp]
            
            if tt_squared:
                derivatives_main = utils_cable_inv._ddm2_ttsq_coord(coords, coords_sec, coords_src, z_src, 
                                            tshift_paras[0], tshift_paras[1], delta_ts[s], v=vel_water )
                derivatives_cross = utils_cable_inv._dxdy_ttsq_coord(coords, coords_sec, coords_src, z_src, 
                                            tshift_paras[0], tshift_paras[1], delta_ts[s], v=vel_water )
            else:
                derivatives_main = utils_cable_inv._ddm2_coord(coords, coords_sec, coords_src, z_src, v=vel_water)
                derivatives_cross = utils_cable_inv._dxdy_coord(coords, coords_sec, coords_src, z_src, v=vel_water)
            
            assert len(derivatives_main) == len(derivatives_cross) == len(idxes_rec_used)
            for i, (idx_rec, ddm2, dxdy) in enumerate(zip( idxes_rec_used, derivatives_main, derivatives_cross) ):
                J[ndata_filled+i, idx_rec + nparas_before] = ddm2 + dxdy
        
        ndata_filled += sum(bools_rec_tmp.astype(int))
    if record_time: 
        print("d/dm Jacobian built in {:.3f}s".format(time.perf_counter()-tex_start))
        
    return J 


def inv_iter_timeshift(parameters, tshift_paras, d_res, d_obs, obj_val,  Wd, 
                       alpha=1.0, steplen_init=1.0, gamma=1e-3, beta_steplen=0.5, 
                       tt_squared=False, jacobian=None, d_pred=None, args_jac=None, 
                       kargs_jac=None, args_fwd=None, kargs_fwd=None, 
                       kargs_obj=None, thresh_delta=None, full_newton=False, 
                       it=0, full_output=False, verbose=True, vel_water=1500,
                       ncalls_max=200,
                       ):
    """ iteration of time shift inversion """
    
    if tt_squared==False and full_newton==True: 
        print("""!!! WARNING: the second derivatives w.r.t. the time shift parameters are zero.
              Method will be reduced to Gauss-Newton.""")
        #time.sleep(1)
    
    delta_used = np.zeros(len(tshift_paras), dtype='float32')
    
    w = np.diag(Wd) # extract diagonal weights
    
    if tt_squared:
        assert args_jac is not None, "provide list of key arguments to generate Jacobian"
        jacobian = jacobian_timeshift_ttsq(parameters, tshift_paras, *args_jac, vel_water=vel_water)
        
        assert d_pred is not None, "! d_pred ust be provided"
        d_res_scaled = 2* d_pred* w**2 * d_res
    else:
        d_res_scaled= w**2 * d_res
        
        if jacobian is None:
            jacobian = jacobian_timeshift(kargs_fwd["delta_ts"], kargs_fwd["bools_rec"]  )
        
    hessian = jacobian.T @ Wd.T @ Wd @ jacobian + alpha*np.eye(len(tshift_paras))
    
    if full_newton:
        ddm_jac = ddm_jacobian_timeshift(parameters, tshift_paras, *args_jac, tt_squared=tt_squared, vel_water=vel_water )
        hessian +=  ddm_jac.T @ jacobian
            
    hess_pos_def = utils.is_pos_def(hessian)
    hessian_inv = np.linalg.inv(hessian )
 
    grad = jacobian.T @ d_res_scaled + alpha *tshift_paras
    search_dir =  -hessian_inv @ grad
    steplen = steplen_init
    obj_val_new = obj_val
    rhs = gamma* grad @ search_dir
    ncalls=0
    while (obj_val_new > steplen*rhs + obj_val) and (ncalls<=ncalls_max):
        steplen = steplen_init * beta_steplen**ncalls
        delta = steplen * search_dir
        tshift_paras_new = tshift_paras + delta
        d_pred_tmp = forward_multi(parameters, *args_fwd, paras_tshift_ext=tshift_paras_new, 
                              **kargs_fwd)
        d_res_tmp = d_pred_tmp - d_obs
        obj_val_new = lsqr_misfit_model(d_res_tmp, parameters,  **kargs_obj)
        #print(f"DEBUG: ncalls:{ncalls_tshift}, obj={obj_val_new}, steplen:{steplen_tshift[i]}, delta_t:{delta_tshift[i]}")
        ncalls+=1
    if ncalls==0:
        delta = steplen_init*search_dir
    
    # clip delta timeshift 
    if thresh_delta is not None: 
        for q in range(len(delta) ):
            if np.abs(delta[q]) > thresh_delta[q]:
                sign = delta[q]/np.abs(delta[q])
                delta_used[q] = sign *thresh_delta[q]
            else: 
                delta_used[q] = delta[q]
    else: 
        delta_used = delta
    tshift_paras_new = tshift_paras + delta_used
    if verbose:
        print("\n iter={}, delta_tshift_used={}, tshift_new={}, misfit={}, ncalls={}, alpha={:.1f}".format( \
                    it,delta_used,tshift_paras_new,obj_val_new,ncalls,alpha) )
    
    if full_output: 
        return (tshift_paras_new, delta_used, delta, grad, search_dir, steplen)
    else:
        return tshift_paras_new


def inv_iter_position(parameters, tshift_paras, d_res, d_obs, obj_val, Wd, alpha, Wm, args_jac, args_fwd, kargs_jac=None, 
                    kargs_fwd=None, full_newton=False, tt_squared=False, d_pred=None, thresh_delta=None, full_output=False, verbose=True,
                    steplen_init=1.0, gamma=1e-3, ncalls_max=200, beta_steplen=0.5):
    """ iteration of coordinate inversion"""
    
    jacobian = jacobian_positions(parameters,  *args_jac, tshift_paras=tshift_paras, **kargs_jac)
    
    hessian = jacobian.T @ Wd.T @ Wd @ jacobian + alpha* Wm.T@Wm
    if full_newton:
        ddm_jac = ddm_jacobian_positions(parameters, tshift_paras, *args_jac, **kargs_jac)    #tt_squared=False, delta_ts=None, record_time=True, verbose=False, vel_water=1500
        hessian += ddm_jac.T @ jacobian
    
    hess_pos_def = utils.is_pos_def(hessian)
    hessian_inv = np.linalg.inv(hessian )
    
    w = np.diag(Wd) # extract diagonal weights

    if tt_squared: 
        assert d_pred is not None, "! d_pred ust be provided"

        d_res_scaled = 2* d_pred* w**2 * d_res
    else:
        d_res_scaled= w**2 * d_res
    
    gradient = jacobian.T @d_res_scaled + alpha* Wm.T @Wm @ parameters # + #@smooth_mat
    search_dir =  -hessian_inv @ gradient
    
    steplen =  steplen_init
    obj_val_new = obj_val
    rhs = gamma* gradient.dot(search_dir) #np.linalg.norm(
    ncalls_step=0
    #print(f"\nDEBUG: i={iterx}, obj_val_init={obj_val_new}, uneq: {steplen*rhs + obj_val}")
    while (obj_val_new > obj_val + steplen*rhs) and (ncalls_step<=ncalls_max):
        steplen =  steplen_init* beta_steplen**ncalls_step # 0.5
        delta_paras = steplen* search_dir
        parameters_new = parameters + delta_paras
        d_pred = forward_multi(parameters_new, *args_fwd, paras_tshift_ext=tshift_paras, 
                                   **kargs_fwd)
        d_res = d_pred - d_obs
        obj_val_new = lsqr_misfit_model(d_res, parameters_new,  alpha=alpha, Wd=Wd, Wm=Wm)
        ncalls_step+=1
        #print(f"DEBUG: ncalls:{ncalls_step}, obj={obj_val_new}, steplen:{steplen}, uneq:{steplen*rhs + obj_val:.3f}")
    if ncalls_step==0: 
        delta_paras=search_dir
        
    if thresh_delta:
        delta_paras = np.clip(delta_paras, -thresh_delta, thresh_delta)
    parameters_new = parameters + delta_paras; 
    
    if verbose:
        print(f"""alpha={alpha:.3f}, misfit: {obj_val_new:.1f}, ncalls_step: {ncalls_step}""")
    
    if full_output: 
        return (parameters_new, d_res, obj_val_new, delta_paras, gradient, search_dir, steplen)
    else: 
        return (parameters_new, d_res, obj_val_new)