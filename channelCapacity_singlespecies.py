#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20170918

@author: xfz
"""


from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pdb
import pandas as pd
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.integrate import odeint
import pickle
from sklearn import linear_model
from pathlib import Path
import warnings

class gillespie:
    def __init__(self,init_state,rate,al_create):
        self.t=0.
        self.state,self.rate=init_state,rate
        self.al_create=al_create
        self.al=al_create(self.state,self.rate)
    
    def step(self):
        r,t=self.draw(self.al)
        self.act(r)
        self.state=dict(list(self.state.items()))
        self.t+=t
        self.al=self.al_create(self.state,self.rate)
        return self.state,self.t,r,self.al

    def draw(self,al):
        r1,r2=np.random.rand(2)
        al_cumsum=np.cumsum(list(al.values()))
        t=-1/al_cumsum[-1]*np.log(r1)
        al_normalized=al_cumsum/al_cumsum[-1]
        idx=np.where(np.ceil(al_normalized-r2))[0]
        return list(al.keys())[idx[0]],t

    def act(self,r):
        for temp in r:
            self.state[temp[0]]=self.state[temp[0]]+temp[1]

def sim(g_sim,fN,T,dataFolder):
    stream=open(dataFolder+fN,'w')
    st=time.time()
    counter=0
    keys=['t']+list(g_sim.state.keys())
    stream.write(','+','.join(keys)+'\n')
    
    event_dict=dict(zip(g_sim.al.keys(),[0]*(len(g_sim.al.keys()))))
    t=0.
    while t<T:
        state,t,r,al=g_sim.step()
        stream.write(str(counter)+','+','.join([str(t)]+[str(state[x]) for x in keys[1:]])+'\n')
        event_dict[r]+=1
        counter+=1
    stream.close()
    pickle.dump(event_dict,open(dataFolder+fN+'_event.p','wb'))
    ed=time.time()
    print(ed-st)


def al_simple(state,rate):
    al={(('X',+1),):rate['alpha_X_func'](state['X'],rate),
        (('X',-1),):rate['mu_X_func'](state['X'],rate)*state['X'],
        }
    return al

def init_simple_default(param={'state':{},'rate':{}}):
    state_init={'X':10}
    state_init.update(param['state'])
    print(state_init)
    # rate_poly follows the format of ((coeffs),(corresponding degrees))
    rate={'alpha_X':10,
          'mu_X':1,
          'alpha_X_func':"lambda x,rate:rate['alpha_X']",
          'mu_X_func':"lambda x,rate:rate['mu_X']"
          }
    rate.update(param['rate'])
    print(rate)
    return state_init,rate
    
def plot_time_traj(fN):
    aa=pd.DataFrame.from_csv(dataFolder+fN)
#    t_cutoff=aa['t'].values[-1]/2
    t_cutoff=10
    t_mask=aa['t']>t_cutoff
    tt=(aa['t'].loc[t_mask]).values
    
    plot_keys=['X']
#    for var in aa.keys()[1:]:
    for var in plot_keys:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(tt,(aa[var].loc[t_mask]).values,'-k',label=var)
        ax.set_xlabel('time ')
        ax.set_ylabel(var)
#        ax.set_ylim([0,200])
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc=1, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
        plt.savefig(fN+'_'+var+'_traj.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  

def plot_var(fN):
    aa=pd.DataFrame.from_csv(dataFolder+fN)
    rate=pickle.load(open(dataFolder+fN+'_rate.p','rb'))['rate']
    t_cutoff=aa['t'].values[-1]/5.
    t_mask=aa['t']>t_cutoff
    tt=(aa['t'].loc[t_mask]).values
    t_weight=np.diff(tt)
    dur=tt[-1]-t_cutoff
    
    xx=(aa['X'].loc[t_mask]).values
    xx=xx[:-1]
    xx_mean_traj=np.cumsum(xx*t_weight)/np.cumsum(t_weight)
    xx_mean=xx_mean_traj[-1]
    xx_var_traj=np.cumsum((xx-xx_mean)**2*t_weight)/np.cumsum(t_weight)
    xx_var=xx_var_traj[-1]
#    pdb.set_trace()
    xx_degProp=np.array([eval(rate['mu_X_func'])(x,rate) for x in xx])
    xx_degProp_mean=np.sum(xx_degProp*t_weight)/dur
    xx_degProp_var=np.sum((xx_degProp-xx_degProp_mean)**2*t_weight)/dur
#    xx_prodProp=np.array([eval(rate['alpha_X_func'])(x,rate) for x in xx])
#    pdb.set_trace()
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(tt[:-1],xx_var_traj,'-k')
    bound_func=lambda rate,x_mean,deg_mean,deg_var:rate['alpha_X']/(deg_mean*(x_mean*np.log(1+deg_var/deg_mean**2)+1))
    bound=bound_func(rate,xx_mean,xx_degProp_mean,xx_degProp_var)
    ax.plot(tt,np.ones(len(tt))*bound,'-r')
    ax.set_xlabel('time ')
    ax.set_ylabel('Var X')
    ax.set_ylim([0,0.005])
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=1, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
    plt.savefig(fN+'_var.png', bbox_extra_artists=(lgd,), bbox_inches='tight') 
    
    pdb.set_trace()

def sim_par_func(arg):
    fN=arg['fN']
    T=arg['T']
    init_state,rate=init_simple_default(arg['param'])
    rate_dict={'init_state':init_state,'rate':rate,'T':T}
    pickle.dump(rate_dict,open(dataFolder+fN+'_rate.p','wb'))
    for key in ['alpha_C_func','alpha_X_func','mu_X_func','mu_C_func']:
        rate[key]=eval(rate[key])
    g_sim=gillespie(init_state,rate,al_simple,particle_gen)
    st=time.time()
    sim(g_sim,fN,T,dataFolder)
    ed=time.time()
    print(ed-st)

def sim_par(fNs,param_list,nCore,T,dataFolder):
    # param_list each element is a dictionary ((key,value),(key,value),...) corresponding to entries to be changed in rate dict.
    args=[{'dataFolder':dataFolder,'T':T,'param':x,'fN':fN} for x,fN in zip(param_list,fNs)]
    if nCore==1:
        for arg in args:
            sim_par_func(arg)
    else:
        pool=Pool(nCore)
        pool.map(sim_par_func,args)


if __name__=='__main__':
    dataFolder='./data/'

############### ONE TRAJ ##########################
    T=100000
    t_cutoff=T*2/5.
    nCore=1
    rate={'alpha_X':1,
          'mu_X':10,
          'x_d':1000,
          'mu_X_func':"lambda x,rate:rate['mu_X'] if x>rate['x_d'] else 1e-10"
          }
    state={'X':1000}
    param_list=[{'rate':rate,'state':state}]
    state_init,rate=init_simple_default(param=param_list[0])
    
    
    
    fN='channel_capacity_SINGLE_test_short'
    rate_dict={'init_state':state_init,'rate':rate,'T':T}
    pickle.dump(rate_dict,open(dataFolder+fN+'_rate.p','wb'))
    
    for key in ['alpha_X_func','mu_X_func']:
        rate[key]=eval(rate[key])
    g_sim=gillespie(state_init,rate,al_simple)
#    sim(g_sim,fN,T,dataFolder)
#    plot_time_traj(fN)
    plot_var(fN)
    pdb.set_trace()

