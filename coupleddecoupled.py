#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 23:38:32 2018

@author: xfz
"""

from __future__ import print_function
import numpy as np
import pdb
import pandas as pd
import time
import matplotlib.pyplot as plt
import pickle

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
    state_keys=list(g_sim.state.keys())
    al_keys=list(g_sim.al.keys())
    al_keys_str=[str(x).replace(',','').replace('(','').replace(')','') for x in al_keys]
    keys=['t']+state_keys+al_keys_str
    stream.write(','+','.join(keys)+'\n')
    
    event_dict=dict(zip(al_keys,[0]*(len(al_keys))))
    t=0.
    while t<T:
        state,t,r,al=g_sim.step()
        stream.write(str(counter)+','+','.join([str(t)]+[str(state[x]) for x in state_keys]+[str(al[y]) for y in al_keys])+'\n')
        event_dict[r]+=1
        counter+=1
    stream.close()
    pickle.dump(event_dict,open(dataFolder+fN+'_event.p','wb'))
    ed=time.time()
    print(ed-st)

def al_decoupled(state,rate):
    al={(('X1',+1),):rate['x1-birth-prop'](state,rate),
        (('X1',-1),):rate['x1-death-prop'](state,rate),
        (('X2',+1),):rate['x2-birth-prop'](state,rate),
        (('X2',-1),):rate['x2-death-prop'](state,rate)
        }
    return al

def al_coupled(state,rate):
    pass
    

def prod_rate(x,const,x_d,q):
    #const is the value to be taken if not flipping.
    #x_d is desired value for cutoff
    #q is flipping rate, between 0 and 1.
    r=np.random.rand()
    return (r>q)*(x<=x_d)*const

def deg_rate(x,const,x_d,q):
    r=np.random.rand()
    return (r>q)*(x>=x_d)*const

def init_simple(param={'state':{},'rate':{}}):
    state_init={'X':10}
    state_init.update(param['state'])
    print(state_init)
    # rate_poly follows the format of ((coeffs),(corresponding degrees))
    rate={'w':10,
          'gamma':1,
          'k':1,
          'x1-death-prop':"lambda state,rate:state['X1']*state['X2']*rate['k']",
          'x2-death-prop':"lambda state,rate:state['X2']*rate['gamma']",
          'x1-birth-prop':"lambda state,rate:rate['w']",
          'x2-birth-prop':"lambda state,rate:rate['w']",
          'x1x2-birth':"lambda state,rate:rate['w']"
          }
    rate.update(param['rate'])
    print(rate)
    return state_init,rate
    
def plot_time_traj(fN):
    aa=pd.DataFrame.from_csv(dataFolder+fN)
#    t_cutoff=aa['t'].values[-1]/2
    t_cutoff=0
    t_mask=aa['t']>t_cutoff
    tt_temp=(aa['t'].loc[t_mask]).values[:-1]
    tt=np.pad(tt_temp,(1,0),'constant',constant_values=0)
    
    plot_keys=['X1','X2']
#    for var in aa.keys()[1:]:
    for var in plot_keys:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.step(tt,(aa[var].loc[t_mask]).values,'-k')
        ax.set_xlabel('time ')
        ax.set_ylabel(var)
#        ax.set_ylim([0,200])
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc=1, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
        plt.savefig(fN+'_'+var+'_traj.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  


def plot_joint_distr(fN):
    aa=pd.DataFrame.from_csv(dataFolder+fN)
    rate=pickle.load(open(dataFolder+fN+'_rate.p','rb'))['rate']
    t_cutoff=aa['t'].values[-1]/5.
    t_mask=aa['t']>t_cutoff
    tt_temp=(aa['t'].loc[t_mask]).values
    tt=np.pad(tt_temp,(1,0),'constant',constant_values=0)
    t_weight=np.diff(tt)
    dur=tt[-1]-t_cutoff
    
    x1=(aa['X1'].loc[t_mask]).values
    x2=(aa['X2'].loc[t_mask]).values

    hist,xedges,yedges=np.histogram2d(x1,x2,bins=20,normed=True,weights=t_weight)
    
#    mi=calc_MI(hist)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, hist)
    ax.set_xlabel('X1 count')
    ax.set_ylabel('X2 count')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=1, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
    plt.savefig(fN+'_distr.png', bbox_extra_artists=(lgd,), bbox_inches='tight') 

#    return mi


if __name__=='__main__':
    dataFolder='./data/'

############### From parameter to MI ##############
#    alpha=10
#    beta=1
#    x_d=10
#    q=0.1
#    mi=param_to_MI(alpha,beta,x_d,q,prod_flipping=False,deg_flipping=True,T=10000)
#    print(mi)

############### ONE TRAJ ##########################
    T=10000
    rate={'w':30,
          'gamma':1,
          'k':1,
          'x1-death-prop':"lambda state,rate:state['X1']*state['X2']*rate['k']",
          'x2-death-prop':"lambda state,rate:state['X2']*rate['gamma']",
          'x1-birth-prop':"lambda state,rate:rate['w']",
          'x2-birth-prop':"lambda state,rate:rate['w']",
          'x1x2-birth':"lambda state,rate:rate['w']"
          }
    state={'X1':10,'X2':10}
    param_list=[{'rate':rate,'state':state}]
    state_init,rate=init_simple(param=param_list[0])
    
    fN='decoupled_test'
    rate_dict={'init_state':state_init,'rate':rate,'T':T}
    pickle.dump(rate_dict,open(dataFolder+fN+'_rate.p','wb'))
    
    func_keys=['x1-death-prop',
          'x2-death-prop',
          'x1-birth-prop',
          'x2-birth-prop',
          'x1x2-birth']
    for key in func_keys:
        rate[key]=eval(rate[key])
    g_sim=gillespie(state_init,rate,al_decoupled)
    sim(g_sim,fN,T,dataFolder)
    plot_time_traj(fN)
    plot_joint_distr(fN)
