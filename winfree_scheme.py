#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 00:46:00 2017

@author: xfz
"""
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
            self.state[temp[0]]=self.state[temp[0]]+temp[1]
            
        
def al_winfree(state,rate,input_var):
    NN=rate['N'];kk=rate['k'];xx=state['X']
    right=[(((x,-1),(x+1,+1)),xx*state[x])  for x in np.arange(-NN,NN,1)]
    left =[(((x,-1),(x-1,+1)),kk*state[x])  for x in np.arange(-NN+1,NN+1,1)]
    al=dict(right+left)
    al[(('X',-1),(NN,-1),(0,+1))]=rate['mu']*state[NN]*state['X']
    al[(('X',+1),(-NN,-1),(0,+1))]=rate['alpha']*state[-NN]
    return al


def sim(g_sim,fN,T):
    stream=open(dataFolder+fN,'w')
    st=time.time()
    counter=0
    keys=['t']+list(g_sim.state.keys())
    [stream.write(','+str(x)) for x in keys]
    stream.write('\n')

    event_dict=dict(zip(g_sim.al.keys(),[0]*(len(g_sim.al.keys()))))
    t=0.
    while t<T:
        state,t,r=g_sim.step()
        rslt=[t]+[state[x] for x in keys[1:]]
        stream.write(str(counter))
        [stream.write(','+str(x)) for x in rslt]
        stream.write('\n')
        event_dict[r]+=1
        counter+=1
#        print(str(t)+', '+str(t/T),end='\r')
    stream.close()
    pickle.dump(event_dict,open(dataFolder+fN+'_event','wb'))
    ed=time.time()
    print(ed-st)

    
def init_winfree(n_molecule=1,state_in={},rate_in={}):
    rate={'N':10,'k':5.,'alpha':1,'mu':1}
    rate.update(rate_in)
    NN=rate['N']
    state=dict([(x,0) for x in np.arange(-NN,NN+1,1)])
    state[0]=n_molecule
    state['X']=rate['k']
    state.update(state_in)
    input_var=lambda t:0.
    return state,rate,input_var

def plot_time_traj(fN,var_list=[]):
    aa=pd.DataFrame.from_csv(dataFolder+fN)
    if not var_list:
        var_list=aa.keys()[1:]
        
    for var in var_list:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(aa['t'],aa[var],'-k')
        ax.set_xlabel('time (min)')
        ax.set_ylabel(var)
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc=1, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
        plt.savefig(fN+'_'+var+'_traj.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  

def plot_time_traj_diffusion(fN):
    aa=pd.DataFrame.from_csv(dataFolder+fN)
    data=np.transpose(np.transpose(aa.values)[1:])
    pos=(aa.keys()[1:][np.where(data)[1]]).astype(np.float)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.step(aa['t'],pos,'-k')
    ax.set_xlabel('time (min)')
    ax.set_ylabel('pos')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=1, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
    plt.savefig(fN+'_traj_diffusion.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  
    
def plot_distr_diffusion(fN,ts=np.logspace(-3,-0.1,5)):
    aa=pd.DataFrame.from_csv(dataFolder+fN)
    N_tot=np.sum(aa.values[0][1:]) # total number of molecules there is. to convert to prob.
    data=aa.ix[:,aa.columns != 't']
    tt=aa['t']
    ts=ts*tt.values[-1]
    pos_axis=(data.columns).astype(np.float)
    pos_idx=np.argsort(pos_axis)
    pos_axis_sorted=pos_axis[pos_idx]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for t in ts:
        idx=np.where(tt-t>0)[0][0]
        prob=data.loc[idx] / N_tot
        ax.step(pos_axis_sorted,prob.values[pos_idx],label='t='+str(tt[idx]))
    ax.set_xlabel('pos')
    ax.set_ylabel('prob')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=1, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
    plt.savefig(fN+'_distr_diffusion.png', bbox_extra_artists=(lgd,), bbox_inches='tight')     

def continuous_bound_plot(picFN,newDic_list,newRate_list,prevPicDicFN='simple',x1var='C',x2var='E'):
    # given newDic, which represents a list of new data points, 
    # make a new plot using all the old data points form prevPicDicFN, as well as the new ones.
    # Then save the resulting whole data file into prevPicDicFN.
    # this way this plot can continue to grow...
    key_ordering=('architecture','alpha_C','mu_C','alpha_E','alpha_C_func','alpha_E_func','mu_C_func','x1_est_func')
    fPath=Path(dataFolder+prevPicDicFN+'.p')
    if fPath.is_file():
        pic_dic=pickle.load(open(dataFolder+prevPicDicFN+'.p','rb'))
#        pd.DataFrame.from_csv(dataFolder+prevPicDicFN+'.csv')
    else:
        pic_dic={}
#        picDF=pd.DataFrame()
    if newRate_list and newDic_list:
        # if not empty, add entries. Else, just plot.
        for newRate,newDic in zip(newRate_list,newDic_list):
            temp=tuple(((key,newRate[key]) for key in key_ordering))
            pic_dic[temp]=newDic # if existent, cover. if non-existent, add.
#        pickle.dump(rate_dic,open(dataFolder+prevPicDicFN+'_rate.p','wb'))
#        picDF.to_csv(dataFolder+prevPicDicFN+'.csv')
        pickle.dump(pic_dic,open(dataFolder+prevPicDicFN+'.p','wb'))
########### x-axis is Nc, general bound ############################
    fig=plt.figure()
    ax=fig.add_subplot(111)
    clr_dic={'deg':'c','noDeg':'g','both':'k','new':'b'}
    for row in pic_dic.values():
        ax.plot(row['Nc'],row['fano'],'o',markersize=10,color=clr_dic[row['architecture']])
    for row in newDic_list:
        ax.plot(row['Nc'],row['fano'],'o',markersize=10,color=clr_dic['new'])
#    [ax.plot([],[],'o',markersize=10,color=val,label=key) for key,val in clr_dic.items()]

    xx=np.logspace(-3,3,num=100)
    paulsson_bound_func=lambda x:2/(1+np.sqrt(1+4*x))
    ax.plot(xx,paulsson_bound_func(xx),'-r',lw=2,label='Theoretical Bound')
    ax.set_xlabel('Nc')
    ax.set_ylabel('Var(x)/Mean(x)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([1e-3,1e3])
    ax.set_ylim([1e-2,1e1])
#    handles, labels = ax.get_legend_handles_labels()
#    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1.,1.02),borderaxespad=0.0)
#    plt.savefig(picFN+'_x1_'+x1var+'_x2_'+x2var+'_ncplot_both.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  
    plt.savefig(picFN+'_x1_'+x1var+'_x2_'+x2var+'_ncplot_both.png', bbox_inches='tight')  

def get_dic(fN,rate):
    aa=pd.DataFrame.from_csv(dataFolder+fN)
    events=pickle.load(open(dataFolder+fN+'_event','rb'))
#    n_events=np.sum(list(events.values()))-events[(('X',1),(-rate['N'],-1),(0,1))]-events[((-1,-1),(-2,1))]
    n_events=np.sum(list(events.values()))-events[((-rate['N']+1,-1),(-rate['N'],1))]
    
    dur=aa['t'].values[-1]
    signaling_rate=n_events/dur
    t_weight=np.diff(aa['t'].values)
    xx=aa['X'].values[:-1]
    temp=xx*aa[str(rate['N'])].values[:-1]
    tau=1/(rate['mu']*np.sum(temp*t_weight)/dur)
    x_mean=np.sum(xx*t_weight)/dur
    x_var=np.sum((xx-x_mean)**2*t_weight)/dur
    fano=x_var/x_mean
    nc=signaling_rate*tau/x_mean
    pdb.set_trace()
   
if __name__=='__main__':
    global dataFolder
    dataFolder='./data/'
    rate={'N':2}
    state,rate,input_var=init_winfree(n_molecule=1,rate_in=rate)
    g_sim=gillespie(state,rate,input_var,al_winfree)
    fN='test'
    T=10000
    sim(g_sim,fN,T)
    get_dic(fN,rate)
#    plot_time_traj(fN,var_list=['X'])
    pdb.set_trace()
#    plot_distr_diffusion(fN)
#    plot_time_traj_diffusion(fN)
    
    
#    T=300
#    t_cutoff=T*1/2.
#    nCore=4
#    alpha_C_list=np.logspace(2,4,20)
#    prefix='KhammashFull_v_1'
#    fNs=[prefix+'_'+str(i) for i in range(len(alpha_C_list))]
#    cv_sim(fNs,alpha_C_list,nCore,T)
#    plot_bound(prefix,'C','P',fNs,t_cutoff,rate,state_init,T,alpha_C_list)
#    plot_bound(prefix,'Ea','C',fNs,t_cutoff,rate,state_init,T,alpha_C_list)
#    plot_time_traj(fNs[0])
    

