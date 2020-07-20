#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:23:39 2017

@author: xfz
"""

# simulate khammash system, compute the bounds, plot the scaling of CV vs the flux.

# Then, maybe simulate the heat shock system and do the same.

# Last. Do analytical computation on the Khammash system and see whether this happens.

# Add m_D to steadyStateApprox.

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
    def __init__(self,init_state,rate,input_var,al_create,state_approx=lambda obj,state,rate,input_var:{}):
        self.t=0.
        self.state,self.rate,self.input_var=init_state,rate,input_var
        #input_var takes t returns a dictionary
        self.al_create=al_create
        self.al=al_create(self.state,self.rate,self.input_var(self.t))
        self.state_approx=state_approx
    
    def step(self):
        r,t=self.draw(self.al)
        self.act(r)
        self.state=dict(list(self.state.items())+list(self.state_approx(self.state,self.rate,self.input_var(self.t),t).items()))
        self.t+=t
        self.al=self.al_create(self.state,self.rate,self.input_var(self.t))
        
        return self.state,self.t,r
    
    
    def draw(self,al):
        r1,r2=np.random.rand(2)
        al_cumsum=np.cumsum(list(al.values()))
        t=-1/al_cumsum[-1]*np.log(r1)
        al_normalized=al_cumsum/al_cumsum[-1]
        idx=np.where(np.ceil(al_normalized-r2))[0]
        return list(al.keys())[idx[0]],t

    def act(self,r):
        for temp in r:
            # only variable not updated here is delta_C
            self.state[temp[0]]=self.state[temp[0]]+temp[1]

def sim(g_sim,fN,T,dataFolder):
    stream=open(dataFolder+fN,'w')
    st=time.time()
    counter=0
    keys=['t']+list(g_sim.state.keys())
    [stream.write(','+x) for x in keys]
    stream.write('\n')

    event_dict=dict(zip(g_sim.al.keys(),[0]*(len(g_sim.al.keys()))))
    t=0.
    write_subsampling=10
    while t<T:
        [g_sim.step() for i in range(write_subsampling-1)]
        state,t,r=g_sim.step()
#        if state['Y1']>0:pdb.set_trace()
        rslt=[t]+[state[x] for x in keys[1:]]
        stream.write(str(counter)+','+','.join([str(x) for x in rslt])+'\n')
        event_dict[r]+=1
        counter+=1
#        if t>1: print(g_sim.al);pdb.set_trace()
#        print(str(t)+', '+str(t/T),end='\r')
    stream.close()
    pickle.dump(event_dict,open(dataFolder+fN+'_event.p','wb'))
    ed=time.time()
    print(ed-st)
   
def al_mean(state,rate,input_var):
    al={(('X',+1),):rate['k1']*state['U1'],
        (('X',-1),):rate['mu_X']*state['X'],
        (('Y1',+1),):rate['theta1']*state['X'],
        (('U1',+1),):rate['alpha1'],
        (('Y1',-1),('U1',-1)):rate['eta1']*state['Y1']*state['U1']
        }
    return al

def al_mean_var(state,rate,input_var):
    al={(('X',+1),):rate['k1']*state['U1']+rate['k2']*state['U2'],
        (('X',-1),):rate['mu_X']*state['X'],
        (('Y1',+1),):rate['theta1']*state['X'],
        (('Y2',+1),):rate['theta2']*state['X']*(state['X']-1),
        (('U1',+1),):rate['alpha1'],
        (('U2',+1),):rate['alpha2'],
        (('Y1',-1),('U1',-1)):rate['eta1']*state['Y1']*state['U1'],
        (('Y2',-1),('U2',-1)):rate['eta2']*state['Y2']*state['U2']
        }
    return al

def al_var(state,rate,input_var):
    al={(('X',+1),):rate['k2']*state['U2'],
        (('X',-1),):rate['mu_X']*state['X'],
        (('Y2',+1),):rate['theta2']*state['X']*(state['X']-1),
        (('U2',+1),):rate['alpha2'],
        (('Y2',-1),('U2',-1)):rate['eta2']*state['Y2']*state['U2']
        }
    if state['Y1']!=0 or state['U1']!=0: pdb.set_trace()
    return al


def init_simple_default(param={'state':{},'rate':{}}):
    state_init={'X':1,'Y1':0,'Y2':0,'U1':0,'U2':0}
    state_init.update(param['state'])
    print(state_init)
    # mean: alpha1/theta1, variance: alpha2/theta2 + alpha1/theta1 - (alpha1/theta1)^2
    rate={'k1':1,'k2':1,'mu_X':1,'theta1':1,'theta2':1,'alpha1':1,'alpha2':1,
          'eta1':100,'eta2':100
          }
    rate.update(param['rate'])
    print(rate)
    input_var="lambda t:0."
    return state_init,rate,input_var
    
def plot_time_traj(fN):
    aa=pd.DataFrame.from_csv(dataFolder+fN)
    t_cutoff=aa['t'].values[-1]/5
    t_mask=aa['t']>t_cutoff
    tt=(aa['t'].loc[t_mask]).values
    
    plot_keys=['X']
#    for var in aa.keys()[1:]:
    for var in plot_keys:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(tt,(aa[var].loc[t_mask]).values,'-k')
        ax.set_xlabel('time (min)')
        ax.set_ylabel(var)
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc=1, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
        plt.savefig(fN+'_'+var+'_traj.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  


def plot_mean_var(fNs,params):
    var='X'
    var_name=params['var']
    nterms=len(fNs)
    means=np.zeros(nterms)
    variances=np.zeros(nterms)
    ratio2s=np.zeros(nterms)
    for fN,i in zip(fNs,range(nterms)):
        aa=pd.DataFrame.from_csv(dataFolder+fN)
        t_cutoff=aa['t'].values[-1]/5
        t_mask=aa['t']>t_cutoff
        tt=(aa['t'].loc[t_mask]).values
        t_weights=np.diff(tt)
        val=(aa[var].loc[t_mask]).values
        mean=np.average(val[:-1],weights=t_weights)
        variance=np.average((val[:-1]-mean)**2,weights=t_weights)
        ratio2s[i]=variance-mean+mean**2
        means[i]=mean
        variances[i]=variance
    for quantity,name in zip([means,variances,ratio2s],['mean','variance','ratio2']):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(params['val'],quantity,'-o',lw=2)
        ax.set_xlabel(var_name)
        ax.set_ylabel(name)
        ax.set_xscale('log')
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc=1, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
        plt.savefig(fNs[0]+'_'+name+'.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  
    

def plot_distr(fN):
    aa=pd.DataFrame.from_csv(dataFolder+fN)
    t_cutoff=aa['t'].values[-1]/5
    t_mask=aa['t']>t_cutoff
    tt=(aa['t'].loc[t_mask]).values
    t_weights=np.diff(tt)
    plot_keys=['X']
#    for var in aa.keys()[1:]:
    for var in plot_keys:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        val=(aa[var].loc[t_mask]).values
        freq,bin_edges=np.histogram(val[:-1],bins=20,density=True,weights=t_weights)
        freq_padded=np.pad(freq,(1,0),'constant',constant_values=(0,))
        ax.step(bin_edges,freq_padded)
        ax.set_xlabel(var)
        ax.set_ylabel('density')
        mean=np.average(val[:-1],weights=t_weights)
        variance=np.average((val[:-1]-mean)**2,weights=t_weights)
        ratio2=variance-mean+mean**2
        ax.set_title('mean= %.2f' %mean  + ', var= %.2f' %variance+', ratio2= %.2f' %ratio2)
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc=1, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
        plt.savefig(fN+'_'+var+'_distr.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  


def cv_sim_par(arg):
    fN=arg['fN']
    T=arg['T']
    init_state,rate,input_var=init_simple_default(arg['param'])
    rate_dict={'init_state':init_state,'rate':rate,'input_var':input_var,'T':T}
    pickle.dump(rate_dict,open(dataFolder+fN+'_rate.p','wb'))
    input_var=eval(input_var)
    g_sim=gillespie(init_state,rate,input_var,al_mean_var)
    st=time.time()
    sim(g_sim,fN,T,dataFolder)
    ed=time.time()
    print(ed-st)
    print((ed-st)/T/60)

def cv_sim(fNs,param_list,nCore,T,dataFolder):
    # param_list each element is a dictionary ((key,value),(key,value),...) corresponding to entries to be changed in rate dict.
    args=[{'dataFolder':dataFolder,'T':T,'param':x,'fN':fN} for x,fN in zip(param_list,fNs)]
    if nCore==1:
        for arg in args:
            cv_sim_par(arg)
    else:
        pool=Pool(nCore)
        pool.map(cv_sim_par,args)

def mean_var_to_rate(m,v):
    # m is mean, v is variance
    ratio1=m
    ratio2=v-m*(1-m)
    return ratio1,ratio2

if __name__=='__main__':
    dataFolder='./data/'

    T=10000
    t_cutoff=T*1/5.
    nCore=4
#    
#    rate={'theta1':0,
#          'alpha1':0,
#          'k1':0,
#          'theta2':1,
#          'alpha2':20
#          }
    
    m=10
    v=15
    ratio1,ratio2=mean_var_to_rate(m,v)
    # for control of both mean and variance
    rate={'theta1':100,
          'alpha1':100*ratio1,
          'eta1':100*100,
          'eta2':100,
          'theta2':1,
          'alpha2':1*ratio2,
          }
    param_list=[{'rate':rate,'state':{}}]

#    params={'var':'eta2/eta1','val':np.logspace(-3,3,12)}
#    rates=[{'theta1':1,
#          'alpha1':1*ratio1,
#          'eta1':100*1,
#          'eta2':100*x,
#          'theta2':1,
#          'alpha2':1*ratio2,
#          } for x in params['val']]
#    param_list=[{'rate':rate,'state':{}} for rate in rates]
#    pickle.dump(params,open(dataFolder+fNs[0]+'_param.p','wb'))

    prefix='integral_controller_mean_var'
    fNs=[prefix+str(i) for x,i in zip(param_list,range(len(param_list)))]

    cv_sim(fNs,param_list,nCore,T,dataFolder)
    plot_distr(fNs[0])
#    plot_time_traj(fNs[0])
#    params=pickle.load(open(dataFolder+fNs[0]+'_param.p','rb'))
#    plot_mean_var(fNs,params)
    