#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:26:50 2018

@author: xfz
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import pdb 
import pickle
import numpy as np
def plot_MI_bound_from_file(data_save_fN):
    matplotlib.rcParams['axes.labelsize']=25
    matplotlib.rcParams['font.serif']='Times New Roman'
    matplotlib.rcParams['font.family']='sans-serif'
    rc('text',usetex=True)
    
    dataDic=pickle.load(open(data_save_fN,'rb'))
    fig=plt.figure(figsize=(6, 4.5))
    ax=fig.add_subplot(111)
    colors=dict([(key,color) for key,color in zip(dataDic.keys(),['b','c','k'])])
    for key,val in dataDic.items():
        ax.plot(val['encodingCV'],val['mi_est_scaled'],'o',color=colors[key],label=key)
        sim_v_bound=val['mi_est_scaled']-np.log(1+val['encodingCV'])
        if sum(sim_v_bound>0)>0:
            pdb.set_trace()
#        ax.plot(val['encodingVar']/val['encodingMean'],val['mi_est'],'o',color=colors[key],label=key)
#    xlim=[1e-4,6]
    xlim=[1e-4,100]
    xAxis=np.logspace(np.log(xlim[0]),np.log(xlim[1]),1000)
    ax.plot(xAxis,np.log(1+xAxis),'-r',lw=4,label='Bound')
    
#    ax.plot(xAxis,xAxis,'-r',label='Bound')
    ax.set_xlim([xlim[0],15])
    ax.set_ylim([1e-6,0.5e1])
    ax.set_xlabel(r'$CV(\nu)^2$')
    ax.set_ylabel(r'$MI(X,\nu)$')
    ax.tick_params(axis='both', labelsize=20)
#    ax.set_xlabel('Fano(v)')
#    ax.set_ylabel('MI')
    ax.set_yscale('log')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=4, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0,fontsize=25,labelspacing=0.0)
    plt.savefig(data_save_fN+'_MI_bound.png', bbox_extra_artists=(lgd,), bbox_inches='tight') 

    ax.set_xlim(xlim)
    ax.set_xscale('log')
    plt.savefig(data_save_fN+'_MI_bound_log.png', bbox_extra_artists=(lgd,), bbox_inches='tight') 

data_save_fN='plot_MI_bound_degradation.p'
plot_MI_bound_from_file(data_save_fN)