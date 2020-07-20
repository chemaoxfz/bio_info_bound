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

def al_simple(state,rate):
    al={(('X',+1),):rate['prod_prop'](state['X'],rate),
        (('X',-1),):rate['deg_prop'](state['X'],rate),
        }
    return al

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
    rate={'alpha':10,
          'beta':1,
          'x_d':5,
          'q':0.1,
          'prod_prop':"lambda x,rate:prod_rate(x,rate['alpha'],rate['x_d'],rate['q'])",
          'deg_prop':"lambda x,rate:x*deg_rate(x,rate['beta'],rate['x_d'],rate['q'])"
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
    
    plot_keys=['X']
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
    
    xx=(aa['X'].loc[t_mask]).values
    deg=(aa["'X' -1"].loc[t_mask]).values
    deg_rate=deg/xx #deg rate is without X count.

    hist,xedges,yedges=np.histogram2d(xx,deg_rate,bins=20,normed=True,weights=t_weight)
    
    mi=calc_MI(hist)

#    fig=plt.figure()
#    ax=fig.add_subplot(111)
#    X, Y = np.meshgrid(xedges, yedges)
#    ax.pcolormesh(X, Y, hist)
#    ax.set_xlabel('X count')
#    ax.set_ylabel('deg prop')
#    handles, labels = ax.get_legend_handles_labels()
#    lgd = ax.legend(handles, labels, loc=1, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
#    plt.savefig(fN+'_distr.png', bbox_extra_artists=(lgd,), bbox_inches='tight') 
    
    return mi

    
def calc_MI(hgram):
    # copied from matthew-brett.github.io/teaching/mutual_information.html
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def param_to_MI(alpha,beta,x_d,q,prod_flipping=False,deg_flipping=True,T=10000):
    rate={'alpha':alpha,
          'beta':beta,
          'x_d':x_d,
          'q':q
          }
    state={'X':x_d}
    param_list=[{'rate':rate,'state':state}]
    state_init,rate=init_simple(param=param_list[0])
    
    fN='special_test'
    rate_dict={'init_state':state_init,'rate':rate,'T':T}
    pickle.dump(rate_dict,open(dataFolder+fN+'_rate.p','wb'))
    
    
    if prod_flipping:
        rate['prod_prop']=eval(rate['prod_prop'])
    else:
        rate['prod_prop']=eval("lambda x,rate:rate['alpha']")
    
    if deg_flipping:
        rate['deg_prop']=eval(rate['deg_prop'])
    else:
        rate['deg_prop']=eval("lambda x,rate:x*rate['beta']")
    
    g_sim=gillespie(state_init,rate,al_simple)
    sim(g_sim,fN,T,dataFolder)
#    plot_time_traj(fN)
    mi=plot_joint_distr(fN)
    return mi
    

if __name__=='__main__':
    dataFolder='./data/'

############### From parameter to MI ##############
    alpha=10
    beta=1
    x_d=10
    q=0.1
    mi=param_to_MI(alpha,beta,x_d,q,prod_flipping=False,deg_flipping=True,T=10000)
    print(mi)

############### ONE TRAJ ##########################
#    prod_flipping=False
#    deg_flipping=True
#    T=10000
#    rate={'alpha':10,
#          'beta':1,
#          'x_d':11,
#          'q':.0
#          }
#    state={'X':10}
#    param_list=[{'rate':rate,'state':state}]
#    state_init,rate=init_simple(param=param_list[0])
#    
#    fN='special_test'
#    rate_dict={'init_state':state_init,'rate':rate,'T':T}
#    pickle.dump(rate_dict,open(dataFolder+fN+'_rate.p','wb'))
#    
#    
#    if prod_flipping:
#        rate['prod_prop']=eval(rate['prod_prop'])
#    else:
#        rate['prod_prop']=eval("lambda x,rate:rate['alpha']")
#    
#    if deg_flipping:
#        rate['deg_prop']=eval(rate['deg_prop'])
#    else:
#        rate['deg_prop']=eval("lambda x,rate:x*rate['beta']")
#    
#    
#    g_sim=gillespie(state_init,rate,al_simple)
#    sim(g_sim,fN,T,dataFolder)
#    plot_time_traj(fN)
#    plot_joint_distr(fN)
