#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:46:13 2024

@author: keving
"""

import numpy as np, pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

from Functions import utils
import utils_cable_inv



#input
suffix = "pos_noisy"
shotfile = "./Info/shots.csv"
cablefile_orig = "./Info/cable_orig.csv"
path_results="./Output/"
infofile = path_results + f'./Info/info_{suffix}.pkl'

#output
path_figs = "./Figs/"

## plot settings, manual
save_fig=1
idxes_iter = [1,-1]

xlims_map = (-600,800) #(-300, 800)
y_center = 0 #-200
shot_interval_plot = 20
xlims_chs = (7450, 7825) #None #(45_000,46_850)
shots_plot=(260,275,290,305,320,335,350)
shots_plot= np.arange(255,370,15)
#shots_plot = [shot-7 for shot in shots_plot]

plot_inset_zoom=True
paras_inset = {"orig":(75,75), "dxdy":(150,150), "width":0.45, "xy_anker":(0.02,0.02) }
figsize=(8.5,8.5); dpi=150
kargs_subfig_labels = dict(labels=None, kw_text=None, zorder=5, halfbracket=True, 
                  fontsize=13, xy_shifts={"1":(0.08,0),"3":(-0.02,0)} )



df_shots = pd.read_csv(shotfile, sep='\t')
df_cable_orig = pd.read_csv(cablefile_orig, sep='\t')

df_cable = pd.read_csv(path_results + f"Cable/cable_{suffix}.csv", sep='\t')
df_cable_init = pd.read_csv(path_results + f"Cable/cable_init_{suffix}.csv", sep='\t')
df_data = pd.read_csv(path_results + f"Data/data_{suffix}.csv", sep='\t')



with open(infofile, 'rb') as fp:
    info = pickle.load(fp)
    print("info loaded from pickle")

shotnums_data= np.unique(df_data["shotnumber"])
df_shots = df_shots[df_shots["shotnumber"].isin(shotnums_data)].reset_index(drop=True)

#utils.done()

tshifts_iter = info["timeshift"]["tshift_iter"]
if info["timeshift"]["flag_timeshift"]:
    tshifts_total = utils_cable_inv.get_time_shifts_shots(tshifts_iter, df_shots["timestamp_shot"],
                                          timestamp_orig=utils_cable_inv.TIMESTAMP_FIRST_SHOT_LINE2, 
                                          mean=True)
else: 
    tshifts_total=None
    
iters = np.unique(df_data["iter"])
assert np.array_equal(iters,  np.unique(df_cable["iter"]) ) , "different iterations found for model and data"
 #-1



# automatic plot settings
if info["model"]["true_model"]:
    ylims_tt=(0.155, 0.4) if info["model"]["sin_cable"] else  (0.175, 0.37) #(-0.1, 0.3)
    ylims_tshift=(None,1.1*tshifts_total.max()) if  info["timeshift"]["flag_timeshift"] else None
    
    if info["timeshift"]["flag_timeshift"]:
        tshift_paras_true_plot = info["timeshift"]["tshift_true"]
        if info["model"]["sin_cable"]: 
            suffix_result="sin"
            paras_inset.update({"orig":(50,0)})
        else:
            suffix_result="pos" if info["model"]["sign_coord_shift"]==1 else "neg"
        
        if info["timeshift"].get("sign_tshift") == -1: 
            suffix_result += "_neg_tshift"
            ylims_tt=(-0.1, 0.15)
            
        if info["model"]["sign_coord_shift"]==-1:
            paras_inset.update({"orig":(-10,-20)})
            
    else: 
        suffix_result="simple"
        ylims_tt=(0, 0.275)
        tshift_paras_true_plot = None
        idxes_iter=[1,3]
else: 
    ylims_tt=(0.05, 0.25)#(-0.1, 0.3)
    ylims_tshift=(0,0.03)
    suffix_result="real"
    tshift_paras_true_plot = None
    paras_inset.update({"orig":(25,0), "dxdy":(100,100)})

if info["model"]["offset_xy_init"] is not None: 
    suffix_result += "_m0shift"    

if info["model"]["only_even_shots"] or info["model"]["only_odd_shots"]:
    remainder=0 if info["model"]["only_even_shots"] else 1
    shots_plot = [ shot  if shot%2==remainder else shot+1 for shot in shots_plot ]
    label = "_even_srcs" if info["model"]["only_even_shots"] else "_odd_srcs"
    suffix_result += label
elif info["model"]["shot_interval"]:
    suffix_result += f'_shotinterv{info["model"]["shot_interval"]}'
    
if info["paras"]["sigma_noise"]>=0.006:
    suffix_result += "_noisy"
    
assert suffix==suffix_result, "suffix deviate from loaded one "

iters_plot = iters[ idxes_iter ]
    

# plotting
figname_tmp = path_figs + f'cable_inv_{suffix_result}' if save_fig else None
utils_cable_inv.plot_inv_iter(df_cable, df_shots, df_data, info["misfit"], tshifts_iter, 
                              iters_plot=iters_plot, 
                              xlims_chs=xlims_chs, 
                              tshift_paras_true=tshift_paras_true_plot,
                              df_cable_init=df_cable_init, 
                              df_cable_orig=df_cable_orig, 
                              shots_plot=shots_plot,
                              rec_init=True, 
                              rec_true=info["model"]["true_model"], 
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
                              yscale_misfit="log", 
                              kargs_subfig_labels=kargs_subfig_labels)

print("done")
utils.done()


#%% 


linewidth_start=6
trans_start=0.6

iters_all = np.unique(df_cable["iter"])   

norm_cmap = mpl.colors.Normalize(vmin=iters_all[0]-50, vmax=100) #iters_all[-1]
cmap = mpl.cm.ScalarMappable(norm=norm_cmap, cmap="Reds") #mpl.cm.jey
cmap.set_array([])

fig, axes = plt.subplots(2,1, sharex=True, figsize=(10,10), dpi=150)


for i,it in enumerate(iters_all):
    color = cmap.to_rgba(it)
    linewidth = max( (1.0,linewidth_start*0.85**i) )
    trans = min( (1.0,trans_start*1.05**i) )
    
    df_cable_tmp=df_cable[df_cable["iter"]==it]
    df_cable_tmp = df_cable_tmp.merge(df_cable_init, on="channel_idx")
    
    for k,suffix in enumerate( ("init","true")):
        # df_cable_tmp[f"dist_from_{suffix}"] = ( (df_cable_tmp[f"UTM_X_{suffix}"]-df_cable_tmp["UTM_X"])**2 +\
        #     (df_cable_tmp[f"UTM_Y_{suffix}"]-df_cable_tmp["UTM_Y"])**2) **0.5
        df_cable_tmp[f"diff_x_{suffix}"] = df_cable_tmp["UTM_X"] - df_cable_tmp[f"UTM_X_{suffix}"]
        
        #axes[k].plot(df_cable_tmp["channel_abs"], df_cable_tmp[f"dist_from_{suffix}"], c=color, linewidth=linewidth, alpha=trans )
        axes[k].plot(df_cable_tmp["channel_idx"], df_cable_tmp[f"diff_x_{suffix}"], c=color, linewidth=linewidth, alpha=trans )
        
#axes[0].set(ylabel="Distance from init model")
#axes[1].set(ylabel="Distance from true model")
axes[0].set(ylabel=r"$\Delta$x from $m_{init}$")
axes[1].set(ylabel=r"$\Delta$x from $m_{true}$")

axes[-1].set(xlabel="channel_idx")
axes[-1].invert_xaxis()

cax = fig.add_axes([0.9,0.3, 0.025, 0.35] )
plt.colorbar(cmap, cax=cax,  location='right', fraction=0.05, pad=0.025, label="iter") 
cax.set(ylim=(0,None))

plt.show()



#%% % create movie of data fit 

from Functions import utils_video 

#output
path4gif = "./Figs/Datafit/"
gif = path4gif +f"datafit_{suffix}"

if suffix == "real":
    ylims = (0.0, 0.2)

idx_iter  = -1
iter_test = iters[idx_iter]

df_data_plot = df_data[df_data["iter"]==iter_test]

shots = np.unique(df_data_plot["shotnumber"]) #[5:8]

#shots=shots[10:13]
list_figs=[]
for s,shot in enumerate(shots):
    
    df_tmp = df_data_plot[df_data_plot["shotnumber"]==shot]
    
    fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=150)
    
    ax.errorbar(df_tmp["offset_channel"], df_tmp["traveltime"], yerr=1/df_tmp["weight"] )
    ax.plot(df_tmp["offset_channel"], df_tmp["traveltime_pred"], '-o', c='tab:orange')
    
    ax.set(xlabel="Offset Channels", ylabel="Traveltime + Time Shift [s]", title=f"shot {shot}", 
           xlim=(-30,30), ylim=ylims
           )
    ax.invert_yaxis(); ax.invert_xaxis()
    
    figname = path4gif + f"datafit_{suffix_result}_shot_{shot}"
    fig.savefig(figname + ".png", format="png", bbox_inches='tight')
    list_figs.append(figname)
    #plt.show()
    plt.close(fig)
    
    

    
utils_video.gen_gif(list_figs, gif)
utils_video.gif2mp4(gif)

print("done")
    
    
    








