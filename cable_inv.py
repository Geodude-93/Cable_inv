#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 18:55:07 2024

@author: keving
"""
import time
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pickle

from Functions import utils, utils_pd
import utils_cable_inv
from utils_cable_inv import model2paras, paras2model, data2df, df_from_data, forward_multi, \
        lsqr_misfit_model, _path_lengths, _traveltimes_rays, TIMESTAMP_FIRST_SHOT_LINE2 #datavec2data


def jacobian_timeshift(timestamps_shots, bools_rec, timestamp_orig=TIMESTAMP_FIRST_SHOT_LINE2 ):
    """ jacobian of timeshift model with two parameters (linear function)"""
    nshots = bools_rec.shape[0]
    assert len(timestamps_shots)==nshots
    ndata = sum(bools_rec.flatten().astype('int32'))
    
    J = np.ones([ndata,2]);  counter=0
    for s in range(nshots):
        bools_tmp = bools_rec[s,:]
        ndata_shot = sum(bools_tmp.astype('int32'))
        J[counter:counter+ndata_shot,1] = timestamps_shots[s] - timestamp_orig
        counter += ndata_shot
        
    return J

def jacobian_analytical(parameters, bools_rec, x_src, y_src, z_src, record_time=True, verbose=False, vel_water=1500): 
    """ create the analytical jacobian"""
    
    nparams = len(parameters); assert nparams%2==0
    n_rec = int(nparams/2); 
    nshots = len(x_src); assert nshots==len(y_src)
    ndata = sum(bools_rec.flatten().astype('int32')) #n_rec * nshots
    
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
        
            derivatives = utils_cable_inv._ddm_coord(coords, coords_sec, coords_src, z_src, vel_water=vel_water)
            assert len(derivatives)==len(idxes_rec_used)
            for i, (idx_rec, ddm) in enumerate(zip( idxes_rec_used, derivatives) ):
                J[ndata_filled+i, idx_rec + nparas_before] = ddm
        
        ndata_filled += sum(bools_rec_tmp.astype(int))
    if record_time: 
        print("Jacobian built in {:.3f}s".format(time.perf_counter()-tex_start))
        
    return J 


def inv_iter_timeshift(parameters, tshift_paras, d_res, obj_val, jacobian, hessian_inv,  Wd, alpha_tshift=1.0, 
                       steplen_init=1.0, gamma=1e-3, beta_steplen=0.5, args_fwd=None, kargs_fwd=None, 
                       kargs_obj=None, thresh_delta=None, it=0, full_output=False, verbose=True
                       ):
    """ iteration of time shift inversion """
    
    delta_used = np.zeros(2)
    
    grad = jacobian.T @Wd.T @Wd @d_res + alpha_tshift *tshift_paras
    search_dir =  -hessian_inv @ grad
    steplen = steplen_init
    obj_val_new = obj_val
    rhs = gamma* grad @ search_dir
    ncalls_tshift=0
    while (obj_val_new > steplen*rhs + obj_val) and (ncalls_tshift<=ncalls_max):
        steplen = steplen_init * beta_steplen**ncalls_tshift
        delta = steplen * search_dir
        tshift_paras_new = tshift_paras + delta
        d_pred_tmp = forward_multi(parameters, *args_fwd, paras_tshift_ext=tshift_paras_new, 
                              **kargs_fwd)
        d_res_tmp = d_pred_tmp - d_obs
        obj_val_new = lsqr_misfit_model(d_res_tmp, parameters,  **kargs_obj)
        #print(f"DEBUG: ncalls:{ncalls_tshift}, obj={obj_val_new}, steplen:{steplen_tshift[i]}, delta_t:{delta_tshift[i]}")
        ncalls_tshift+=1
    if ncalls_tshift==0:
        delta = steplen_init*search_dir
    
    # clip delta timeshift 
    if thresh_delta_tshift is not None: 
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
        print("\n iter={}, delta_tshift_used={}, tshift_new={}, misfit={}, ncalls={}, grad={}".format( \
                    it,delta_used,tshift_paras_new,obj_val_new,ncalls_tshift,grad) )
    
    if full_output: 
        return (tshift_paras_new, delta_used, delta, grad, search_dir, steplen)
    else:
        return tshift_paras_new


def inv_iter_position(parameters, tshift_paras, d_res, obj_val, Wd, alpha, Wm, args_jac, args_fwd, kargs_jac=None, 
                    kargs_fwd=None, thresh_delta=None, full_output=False, verbose=True,
                    steplen_init=1.0, gamma=1e-3, ncalls_max=200):
    """ iteration of coordinate inversion"""
    
    jacobian = jacobian_analytical(parameters,  *args_jac, **kargs_jac)
    hessian = jacobian.T @ Wd.T @ Wd @ jacobian + alpha* Wm.T@Wm
    hess_pos_def = utils.is_pos_def(hessian)
    hessian_inv = np.linalg.inv(hessian )
    gradient = jacobian.T @Wd.T @Wd @d_res + alpha* Wm.T @Wm @ parameters # + #@smooth_mat
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
        obj_val_new = lsqr_misfit_model(d_res, parameters_new,  alpha=alpha, Wd=Wd, Wm=smooth_mat)
        ncalls_step+=1
        #print(f"DEBUG: ncalls:{ncalls_step}, obj={obj_val_new}, steplen:{steplen}, uneq:{steplen*rhs + obj_val:.3f}")
    if ncalls_step==0: 
        delta_paras=search_dir
        
    if thresh_delta:
        delta_paras = np.clip(delta_paras, -thresh_delta_xy, thresh_delta_xy)
    parameters_new = parameters + delta_paras; 
    
    if verbose:
        print(f"""alpha={alpha:.3f}, misfit: {obj_val_new:.1f}, ncalls_step: {ncalls_step}""")
    
    if full_output: 
        return (parameters_new, d_res, obj_val_new, delta_paras, gradient, search_dir, steplen)
    else: 
        return (parameters_new, d_res, obj_val_new)


# constants
DEC_CHS=6   # channel spacing

#input
pickfile = "./Info/direct_wave_picks.csv"
shotfile = "./Info/shots.csv"
cablefile = "./Info/cable_orig.csv"

#output
path_figs = "./Figs/"


####################### SETTINGS ############################################

save_results=0                 # save results as csv and pickle files for reload

## settings forward 
vel_water=1500                  # acoustic water velocity
sigma_noise = 0.003 #0.006      # random noise level for synthetic case 
sigma_gaussfilt_tt=3            # smoothing of traveltimes to determine apex of shot

### settings inversion
niter=20 #80                    # number of iterations for inversion            
interval_iter_save=10           # interval for iterations to save results
iters_save = np.append(np.array([1,2]), np.arange(10,niter+interval_iter_save, interval_iter_save) )
iters_save_cable=np.arange(0,niter+1)

# data and weights
lims_uncertainty= (0.003, 0.009)    # uncertainty limits for observed data
lims_offset_meter = (0,300)         # offset limits in meter
uncert_without_apex=0.005           # additional uncertainty added to shots without determined apex
uncert_crit_angle = 0.004
offset_crit_angle = 130
#beta_pick_weight, beta_channel_weight, beta_shot_weight = 1.0, 0.0, 0.0 #0.8, 0.1, 0.1
only_even_shots=0                 # if 1 only even shot numbers are used
only_odd_shots=0                    # # if 1 only odd shot numbers are used
shot_interval_use=None #5 # None    # interval of used shots, default is None -> every shot is used

# true model 
true_model=1                       # if true generate synthetic model and data 
sign_coord_shift=1                 # coordinate shift in positive (1) or negative (-1) direction
offset_xy_true = (100, 75)         # offset in x and y of true model from initial model 
sign_tshift=1
tshift_paras_true = np.array(( 0.025, 3e-06))*sign_tshift   # true time shift for synthetic model
sinosoidal_cable=0;             # introduce sinosoidal cable gemoetry if 1
amp_sin_cable=150               # amplitude of sinosoidal coordinate anomaly in meter

# init model 
offset_xy_init=None #(-75,-75) #None               # shift the initial model in x-y from center acquisition line

## timeshift inversion
invert_for_timeshift=1              # set 1 if inversion for time shift and cable position 
tshift_paras_init = np.array((0.0, 0.0)) # initial time shift parameters
alpha_tshift=1.0                    # regularization weight for time shift
steplen_tshift_init, beta_steplen_tshift = 1.0, 0.5 # steplen parameters for time shift

#clip_tshift_iter=True
thresh_delta_tshift=(0.025, 1e-5)    # thresholds for time shift parameter updates
gamma_tshift = 0.001                # weight used in steplen estimation (see Eqs.)

## coordinate inversion 
steplen_init, beta_steplen = 1.0, 0.5   # steplen parameters for cable position inversion (see Eqs)
gamma=0.001 #0.0001 #0.0001 # 0.01      # weight used in steplen estimation (see Eqs.)

ncalls_max=200                          # max number of call to determine the step length
alpha_start, alpha_min = 100, 0.1       # min and max regularization weights for cable pos inversion
beta_alpha = 0.8 #0.8                   # regularization decay (see Eqs)
thresh_delta_xy = 25 #50 #25 #50        # clip model updates, set to None or 0 if undesired
apply_gauss_filt=0; sigma_smoothing=3   # apply gaussian smoothing during iterations


## plot settings
plot_model=1                       # plot true and initial model
plot_dobs=1                             # plot exemplary data
plot_smooth_mat=0                       # plot regularization matrix
 
 ######################### END Settings #########################################

## check settings
assert (only_even_shots != only_odd_shots) or (only_even_shots==only_odd_shots==0), \
    "do not select both only_even or only_odd shots simultaneously"


# read input
df_shots_all = pd.read_csv(shotfile, sep='\t')
df_picks = pd.read_csv(pickfile, sep='\t')
df_cable_orig = pd.read_csv(cablefile, sep='\t')

#utils.done()

# remove shots 
if only_even_shots: 
    df_picks = df_picks[df_picks["shotnumber"]%2==0]
elif only_odd_shots:
    df_picks = df_picks[df_picks["shotnumber"]%2==1]
elif shot_interval_use: 
    df_picks = df_picks[df_picks["shotnumber"]%shot_interval_use==0]

#remove offsets
lims_offset_channels = tuple([ round(val/DEC_CHS) for val in lims_offset_meter])
df_picks = utils_pd.sub_from_df(df_picks, lims_offset_channels, "offset_channel_abs" ) 

shots_picks = np.unique(df_picks.shotnumber)
n_shots = len(shots_picks)
df_shots = df_shots_all[df_shots_all["shotnumber"].isin(shots_picks)].reset_index(drop=True)

wdepth_mean=df_shots.src_wdepth.mean()
x_src, y_src = [df_shots[label].values for label in ("UTM_X","UTM_Y")]


#utils.done()


# determine channels to invert for 
channels_picks = np.unique(df_picks["channel_idx"])
chlims_picks = (channels_picks.min(), channels_picks.max())
if np.any( np.diff(channels_picks)-1):
    print("WARNING: index of picked channels not constant")
channels_used = np.arange(chlims_picks[0],chlims_picks[1]+1)
assert np.array_equal(channels_picks, channels_used), "channels missing"
df_picks = df_picks.merge(pd.DataFrame({"channel_idx":channels_used, "chidx_local":np.arange(0,len(channels_used))} ), \
                          on="channel_idx").sort_values(\
                              by=["shotnumber","channel_idx"], ignore_index=True)
    
# create uncertainties linear increasing with offset
df_uncertainties =  utils_cable_inv.gen_uncertainty_offset(lims_linear=(0,300), 
                               dec_chs=DEC_CHS, offset_max=500, lims_unc=lims_uncertainty, 
                               offset_crit_angle=offset_crit_angle, unc_crit=uncert_crit_angle)

fig=plt.figure()
plt.plot(df_uncertainties["offset_channel_abs"], df_uncertainties["uncertainty"])

#utils.done()

# generate init cable
df_cable_init = utils_cable_inv.gen_cable_init_apexes(df_shots_all)
df_cable_init = utils_pd.sub_from_df(df_cable_init, chlims_picks, "channel_idx")
n_rec = len(df_cable_init)
parameters_init = model2paras(df_cable_init.UTM_X.values, df_cable_init.UTM_Y.values)



#utils.done()
    

# create true model and data
if true_model:
    if invert_for_timeshift==False: 
        tshift_paras_true=np.array((0.0, 0.0))
    
    df_cable_true = df_cable_init.copy()
    if sinosoidal_cable: 
        xtmp = np.linspace(0,2*np.pi,n_rec)
        df_cable_true["UTM_X"] += amp_sin_cable* sign_coord_shift*np.sin(xtmp)
    else:
        df_cable_true["UTM_X"] += sign_coord_shift* offset_xy_true[0]
        df_cable_true["UTM_Y"] += sign_coord_shift* offset_xy_true[1]
    
    xrec_true, yrec_true = [df_cable_true[label].values for label in ("UTM_X","UTM_Y")]
    parameters_true = model2paras(xrec_true, yrec_true)
    df_cable_init = pd.merge(df_cable_init, df_cable_true, on="channel_idx", suffixes=('_init','_true') )
    
    # shift the initial cable 
    if offset_xy_init is not None: 
        df_cable_init["UTM_X_init"] += offset_xy_init[0]
        df_cable_init["UTM_Y_init"] += offset_xy_init[1]
        
    bools_chs_all = np.ones([n_shots,n_rec], dtype=bool)
    
    #df_obs = df_picks[["shotnumber","channel_idx","chidx_local"]]
    d_obs = forward_multi(parameters_true, n_rec, x_src, y_src, wdepth_mean, 
                          vel_water=vel_water, bools_rec=bools_chs_all, paras_tshift_ext=tshift_paras_true, 
                          timestamps_shots=df_shots["timestamp_shot"].values,
                          sigma_noise=sigma_noise)
    
    df_obs = df_from_data(d_obs, shots_picks, channels_used)
    
  
    
    # determine apexes and offset of predicted shots
    df_obs_pro =  utils_pd.gaussfilt_df(df_obs, cname_sep="shotnumber", cname_data="traveltime", cname_filt="traveltime_gaussfilt", 
                     sigma=3)
    df_apex_pred = utils_cable_inv.get_apexes(df_obs_pro, label_time="traveltime_gaussfilt", labels_out=("channel_idx_apex","time_apex") )
    df_obs =  utils_cable_inv.add_offset2dfpicks(df_obs_pro, df_apex_pred, label_apex_channel="channel_idx_apex")
   
    
   
    #create bools
    df_obs["flag_offset"] = np.zeros(len(df_obs))
    df_obs.loc[(df_obs["offset_channel_abs"]>=lims_offset_channels[0]) & 
                       (df_obs["offset_channel_abs"]<=lims_offset_channels[1]), "flag_offset"]=1
    bools_chs = np.zeros([n_shots,n_rec])
    for s, shot in enumerate(shots_picks):
        df_tmp = df_obs[df_obs["shotnumber"]==shot]
        bools_chs[s,:] = df_tmp["flag_offset"].values
    bools_chs = bools_chs.astype(bool)
    
    df_obs = utils_pd.sub_from_df(df_obs, lims_offset_channels, "offset_channel_abs" ).sort_values(\
                                       by=["shotnumber","channel_idx"], ignore_index=True)
    #utils.done()
    
else: 
    df_obs = df_picks.copy()
    df_obs.rename(columns={"time_after_shot":"traveltime"}, inplace=True)
    df_cable_init.rename(columns={"UTM_X":"UTM_X_init","UTM_Y":"UTM_Y_init"}, inplace=True)
    
    # create bools for receivers for all shots
    bools_chs = np.zeros([n_shots,n_rec], dtype=bool)
    for s,shot in enumerate(shots_picks):
        df_tmp = df_picks[df_picks["shotnumber"]==shot]
        bools_chs[s, df_tmp["chidx_local"].values] = True
    #bools_chs_flat = bools_chs.flatten()
    
d_obs = df_obs["traveltime"].values
ndata=len(d_obs)


#utils.done()

# merge data with uncertainties
df_obs = df_obs.merge(df_uncertainties, on="offset_channel_abs")
# add additional uncertainty to shots where apex is undetermined / at edge of the model
if true_model:
    shots_no_apex = df_apex_pred[df_apex_pred["flag_apex"]==0]["shotnumber"].values
    df_obs.loc[df_obs["shotnumber"].isin(shots_no_apex), "uncertainty" ] += uncert_without_apex
df_obs["weight"] = 1 / df_obs["uncertainty"]
df_obs.sort_values(by=["shotnumber","channel_idx"], inplace=True, ignore_index=True)
Wd = utils_cable_inv.create_weight_mat(df_obs.weight.values, flatten=False)




smooth_mat = utils_cable_inv.fdmat_xy_sec(n_rec)
#plot_smooth_mat=1
if plot_smooth_mat: 
    #fig=plt.figure(figsize=(10,8), dpi=150); 
    mesh = plt.matshow(smooth_mat) #smooth_mat_square
    plt.colorbar(mesh)
    plt.title("smooth mat")
    plt.show()
    
if plot_model: 
    fig, ax = plt.subplots(1,1)
    ax.plot(df_cable_init["UTM_X_init"],df_cable_init["UTM_Y_init"], label='init', c='k' )
    if true_model:
        ax.plot(df_cable_init["UTM_X_true"],df_cable_init["UTM_Y_true"], label='true', c='tab:blue', linewidth=2 )
    ax.scatter(df_shots["UTM_X"], df_shots["UTM_Y"], marker='*', c='r', s=10, label="shots" )
    ax.set(xlabel="X", ylabel="Y")
    ax.legend()
    plt.show()
    
#utils.done()
    
    
d_pred_init = forward_multi(parameters_init, n_rec, x_src, y_src, wdepth_mean, vel_water=vel_water, paras_tshift_ext=tshift_paras_init,
                             bools_rec=bools_chs, timestamps_shots=df_shots["timestamp_shot"].values)
d_res = d_pred_init -d_obs
df_pred = df_obs.copy()
df_pred["traveltime_pred"]=d_pred_init
df_pred["res_time"] = df_pred["traveltime_pred"] - df_pred["traveltime"] 
misfit_init = lsqr_misfit_model(d_res, parameters_init,  alpha=alpha_start, Wd=Wd, Wm=smooth_mat)
rms_misfit_init = utils_cable_inv.rms_misfit(d_pred_init-d_obs)
misfit_norm_init = utils_cable_inv.rms_misfit_weighted(d_res, df_obs.uncertainty.values) 
    

if plot_dobs:
    shotnum_test=280
    df_obs_tmp, df_pred_tmp = [ df[df["shotnumber"]==shotnum_test] for df in (df_obs, df_pred)]
    fig, ax = plt.subplots(1,1)
    ax.errorbar(df_obs_tmp["offset_channel"], df_obs_tmp["traveltime"], yerr=df_obs_tmp["uncertainty"], linewidth=1.5, 
                marker='s', markersize=1.5, zorder=1, label="d_obs")
    if true_model:
        ax.plot(df_obs_tmp["offset_channel"], df_obs_tmp["traveltime_gaussfilt"], '-o', c='tab:orange', markersize=2.5, 
                zorder=2,label="d_obs_filt")
    ax.plot(df_pred_tmp["offset_channel"], df_pred_tmp["traveltime_pred"], '-o', c='tab:red', markersize=2.5, 
             zorder=2,label="d_pred")
    ax.set(xlabel="Offset Channels", ylabel="Traveltime [s]")
    ax.legend()
    ax.invert_yaxis()
    plt.show()
    

    
# prepare inversion 
#nparams = len(parameters_init)
tshifts_iter = np.zeros((niter+1,2)); tshifts_iter[0,:]=tshift_paras_init
misfits, rms_misfits, misfits_norm = [ np.zeros(niter+1) for q in range(3)]
misfits[0]=misfit_init; misfits_norm[0]=misfit_norm_init;  rms_misfits[0]=rms_misfit_init
alphas = np.zeros(niter)

jacobian_tshift = jacobian_timeshift(df_shots["timestamp_shot"].values, bools_chs )
#gradient_tshift, search_dir_tshift, delta_tshift, delta_tshift_used = [np.zeros((niter,2)) for q in range(4)]
#rhs_tshift, steplen_tshift = [np.zeros(niter) for q in range(2)]

hessian_inv_tshift = 1/(jacobian_tshift.T @ Wd.T @ Wd @ jacobian_tshift + alpha_tshift) # * smooth_mat.T@smooth_mat

#utils.done()

### Inversion
niter_saved, niter_saved_cable = 0, 0
tex_start_all = time.perf_counter(); list_figs=[]
for i in range(niter):
    tex_start_iter = time.perf_counter()
    iterx=i+1
    if i==0: 
        parameters = parameters_init.copy()
        tshift_paras = tshift_paras_init
        
    alphas[i] = np.max([ alpha_min, alpha_start* beta_alpha**i])
    
    # forward step
    d_pred = forward_multi(parameters, n_rec, x_src, y_src, wdepth_mean, 
                           bools_rec=bools_chs, paras_tshift_ext=tshift_paras, 
                           timestamps_shots=df_shots["timestamp_shot"].values )
    d_res = d_pred - d_obs 
    obj_val = lsqr_misfit_model(d_res, parameters, alpha=alphas[i], Wd=Wd, Wm=smooth_mat)
    
    #utils.done()
    
    args_fwd=[n_rec, x_src, y_src, wdepth_mean]
    kargs_fwd = dict(bools_rec=bools_chs, timestamps_shots=df_shots["timestamp_shot"].values)
    kargs_obj = dict(alpha=alphas[i], Wd=Wd, Wm=smooth_mat)

    ### inversion for timeshift
    if invert_for_timeshift:
        
        tshift_paras_new = inv_iter_timeshift(parameters, tshift_paras, d_res, obj_val, jacobian_tshift, hessian_inv_tshift,  Wd, 
                           alpha_tshift=alpha_tshift, steplen_init=1.0, gamma=gamma_tshift, beta_steplen=beta_steplen_tshift,
                               args_fwd=args_fwd, kargs_fwd=kargs_fwd, kargs_obj=kargs_obj, 
                               thresh_delta=thresh_delta_tshift, 
                               it=iterx, full_output=False
                               )

    ### inversion for cable position
    args_jac = [bools_chs, x_src, y_src, wdepth_mean]
    kargs_jac = dict(record_time=True, verbose=False, vel_water=vel_water)
    #args_fwd = [n_rec, x_src, y_src, wdepth_mean]
    # kargs_fwd = dict(timeshift_in_model=False, bools_rec=bools_chs, 
    #                       timestamps_shots=df_shots["timestamp_shot"].values)
    
    parameters_new, d_res, obj_val_new = inv_iter_position(parameters, tshift_paras, d_res, obj_val, Wd, alphas[i], smooth_mat, 
                                                         args_jac, args_fwd, kargs_jac=kargs_jac, 
                                                         kargs_fwd=kargs_fwd, thresh_delta=thresh_delta_xy, 
                                                         steplen_init=steplen_init, gamma=gamma, ncalls_max=ncalls_max)

    if apply_gauss_filt: 
       parameters_new = utils_cable_inv.smooth_params_gauss(parameters_new, n_rec, sigma_smoothing)
    
    # assign new parameters and time shift
    parameters = parameters_new
    if invert_for_timeshift:
        tshift_paras = tshift_paras_new;  tshifts_iter[iterx,:]=tshift_paras
    else:   tshifts_iter=None
    xrec_new, yrec_new = paras2model(parameters, n_rec)
    
    # save different misfit metrics
    misfits[iterx] =  obj_val_new 
    rms_misfits[iterx] = utils_cable_inv.rms_misfit(d_res)
    misfits_norm[iterx] = utils_cable_inv.rms_misfit_weighted(d_res, 1/df_obs.weight.values)
    
    print(f"""misfit: {misfits[iterx]:.1f}, misfit_norm: {misfits_norm[iterx]:.2f}, 
           rms: {rms_misfits[iterx]:.3f}, time_iter: {time.perf_counter()-tex_start_iter:.1f}s""")
          
    ## save intermediate results
    if iters_save is not None: 
        
        if iterx in iters_save_cable:
            # save cable pos
            df_cable_tmp= pd.DataFrame(dict(UTM_X=xrec_new, UTM_Y=yrec_new, channel_idx=df_cable_init.channel_idx.values \
                                             ) ).sort_values(by="channel_idx", ignore_index=True)
            df_cable_tmp["iter"] = (np.ones(len(df_cable_tmp))*iterx).astype('int32')
            
            if niter_saved_cable==0: 
                df_cable_all = df_cable_tmp.copy()
            else: 
                 df_cable_all = pd.concat([df_cable_all, df_cable_tmp])
            niter_saved_cable+=1
        
        if iterx in iters_save:
            # save data
            df_data = data2df(df_obs, d_pred, colname_data="traveltime_pred", residual=True, colname_obs="traveltime", colname_res="res_time", 
                              iteration=iterx)
            if niter_saved==0: 
                df_data_all = df_data.copy()
            else: 
                df_data_all = pd.concat([df_data_all, df_data])
            niter_saved+=1

print(f"ex time inversion: {time.perf_counter()-tex_start_all}")

#%% plot results 

if invert_for_timeshift:
    tshifts_total = utils_cable_inv.get_time_shifts_shots(tshifts_iter[:,:], df_shots["timestamp_shot"],
                                          timestamp_orig=TIMESTAMP_FIRST_SHOT_LINE2, mean=True)

## plot settings

iters_plot=iters_save[ [0,-2] ] #-1
xlims_map=(-600,800) #(-300, 800)
y_center=0 #-200
shot_interval_plot=20
xlims_chs=(7500, 7810) #None #(45_000,46_850)
shots_plot=(260,275,290,305,320,335,350)

plot_inset_zoom=True
paras_inset = {"orig":(75,75), "dxdy":(150,150), "width":0.45, "xy_anker":(0.02,0.02) }
figsize=(8.5,8.5); dpi=150
kargs_subfig_labels = dict(labels=None, kw_text=None, zorder=5, halfbracket=True, 
                  fontsize=13, xy_shifts={"1":(0.08,0),"3":(-0.02,0)} )

# automatic plot settings
if true_model:
    ylims_tt=(0.0, 0.4) if sinosoidal_cable else  (0.0, 0.3) #(-0.1, 0.3)
    ylims_tshift=(None,1.05*tshifts_total.max()) if invert_for_timeshift else None
    
    if invert_for_timeshift:
        tshift_paras_true_plot = tshift_paras_true
        if sinosoidal_cable: 
            suffix_result="sin"
            paras_inset.update({"orig":(50,0)})
        else:
            suffix_result="neg" if sign_coord_shift==-1 else "pos"
        if sign_tshift==-1:
            suffix_result += "_neg_tshift"
        
    else: 
        suffix_result="simple"
        ylims_tt=(0, 0.375)
        tshift_paras_true_plot = None
else: 
    ylims_tt=(0.01, 0.30)#(-0.1, 0.3)
    ylims_tshift=(0,0.03)
    suffix_result="real"
    tshift_paras_true_plot = None
    paras_inset.update({"orig":(25,0), "dxdy":(100,100)})

if offset_xy_init is not None: 
    suffix_result += "_m0shift" 

if only_even_shots or only_odd_shots:
    remainder=0 if only_even_shots else 1
    shots_plot = [ shot  if shot%2==remainder else shot+1 for shot in shots_plot ]
    label = "_even_srcs" if only_even_shots else "_odd_srcs"
    suffix_result += label
elif shot_interval_use:
    suffix_result += f"_shotinterv{shot_interval_use}"

if sigma_noise >= 0.006:
    suffix_result +="_noisy"
    
    

# plotting
figname_tmp = path_figs + f'cable_inv_{suffix_result}'
utils_cable_inv.plot_inv_iter(df_cable_all, df_shots, df_data_all, misfits_norm, tshifts_iter, 
                              iters_plot=iters_plot, 
                              xlims_chs=xlims_chs, 
                              tshift_paras_true=tshift_paras_true_plot,
                              df_cable_init=df_cable_init, 
                              df_cable_orig=df_cable_orig, 
                              shots_plot=shots_plot,
                              rec_init=True, 
                              rec_true=true_model, 
                              rec_iter=True, 
                              title=None, 
                              shot_interval_plot=shot_interval_plot, 
                              xlims_map=xlims_map, 
                              y_center=y_center, 
                              cbar=False, 
                              ylims_tt=ylims_tt, 
                              ylims_tshift=ylims_tshift, 
                              figname=figname_tmp,
                              label_subs=True, 
                              figsize=figsize, 
                              dpi=dpi, 
                              xlabel_tt="channel_idx", 
                              plot_inset_zoom=plot_inset_zoom,
                              paras_inset=paras_inset, 
                              plot_channels_ref=True,
                              kargs_subfig_labels=kargs_subfig_labels)



#print("done")

#%% save results
save_results=1

if save_results: 
    path_results="./Output/"
    df_cable_all.to_csv(path_results + f"Cable/cable_{suffix_result}.csv", sep='\t', index=False)
    df_cable_init.to_csv(path_results + f"Cable/cable_init_{suffix_result}.csv", sep='\t', index=False)
    
    df_data_save = df_data_all.round({"traveltime":3, "traveltime_gaussfilt":4,
                                      "uncertainty":3, "weight":1,
                                      "traveltime_pred":4, "res_time":4})
    #df_data_save.drop(colums="traveltime_gaussfilt")
    df_data_save.to_csv(path_results + f"Data/data_{suffix_result}.csv", sep='\t', index=False)
    
    dict_results = { "misfit":misfits_norm, 
                    "timeshift":{"tshift_true":tshift_paras_true, "tshift_init":tshift_paras_init,
                                 "tshift_iter":tshifts_iter, "flag_timeshift":invert_for_timeshift, 
                                 "sign_tshift":sign_tshift},
                    "paras":{"alpha":alphas,"alpha_tshift":alpha_tshift,"beta_alpha":beta_alpha,
                             "alpha_min":alpha_min,"alpha_start":alpha_start,
                             "thresh_tshift":thresh_delta_tshift, "thres_xy":thresh_delta_xy,
                             "gamma":gamma, "gamma_tshift":gamma_tshift, "beta_steplen":beta_steplen,
                             "beta_steplen_tshift":beta_steplen_tshift, "sigma_noise":sigma_noise},
                    "model":{"true_model":true_model, "offset_xy_true":offset_xy_true,
                             "offset_xy_init":offset_xy_init, "sign_coord_shift":sign_coord_shift,
                             "sin_cable":sinosoidal_cable, "amp_sin":amp_sin_cable, 
                             "only_even_shots":only_even_shots, "only_odd_shots":only_odd_shots, 
                             "shot_interval":shot_interval_use, "uncertainty":lims_uncertainty,
                             "offsets_meter":lims_offset_meter, "offsets_channels":lims_offset_channels}
        }
    
    with open(path_results + f'/Info/info_{suffix_result}.pkl', 'wb') as fp:
        pickle.dump(dict_results, fp)
        print('results saved successfully pickl file')
    
    
print("done")
    
    