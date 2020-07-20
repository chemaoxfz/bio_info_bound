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
        self.horizon=[(self.t,self.state[self.rate['horizon_var']])]
        self.al=al_create(self.state,self.rate,self.input_var(self.t),self.horizon)
        self.state_approx=state_approx
    
    def step(self):
        r,t=self.draw(self.al)
        self.act(r)
        self.state=dict(list(self.state.items())+list(self.state_approx(self.state,self.rate,self.input_var(self.t),t).items()))
        self.t+=t
        self.al=self.al_create(self.state,self.rate,self.input_var(self.t),self.horizon)
        self.horizon+=[(self.t,self.state[self.rate['horizon_var']])]
        t_cutoff=self.t-self.rate['horizon_dur']
        idx_cutoff=0
        horizon_gen=iter(self.horizon)
        tt=horizon_gen.__next__()[0]
        while tt<t_cutoff: 
            idx_cutoff+=1
            tt=horizon_gen.__next__()[0]
        self.horizon=self.horizon[idx_cutoff:]
#        if self.t>10:
#            tt,e_traj=np.array(self.horizon).T
#            temp=(e_traj[-1]-e_traj[0])/(tt[-1]-tt[0])/self.rate['alpha_E']
#            pdb.set_trace()
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

def sim(g_sim,fN,T,dataFolder):
    stream=open(dataFolder+fN,'w')
    st=time.time()
    counter=0
    keys=['t']+list(g_sim.state.keys())
    [stream.write(','+x) for x in keys]
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
    pickle.dump(event_dict,open(dataFolder+fN+'_event.p','wb'))
    ed=time.time()
    print(ed-st)
   
def al_simple(state,rate,input_var,horizon):
    tt,e_traj=np.array(horizon).T
    try:
        tt[1]
        c_est=rate['x1_est_func'](e_traj,tt,rate)
    except IndexError:
        # this is when there is only one entry
        c_est=0.
    state['C_est']=c_est
    al={(('C',+1),):rate['alpha_C_func'](c_est,rate),
        (('C',-1),):rate['mu_C_func'](c_est,rate)*state['C'],
        (('E',+1),):rate['alpha_E_func'](state['E'],state['C'],rate)
#        (('E',-1),):eval(rate['mu_E_func'])(state['E'],state['C'])*state['E']
        }
    return al

def init_simple_default(param={'state':{},'rate':{}},architecture='deg'):
    state_init={'E':1,'C':20,'C_est':20}
    state_init.update(param['state'])
    print(state_init)
    # rate_poly follows the format of ((coeffs),(corresponding degrees))
    rate={'alpha_C':30,
          'alpha_E':10,
          'mu_C':10,
          'alpha_C_func':"lambda C_est,rate: rate['alpha_C']",
          'alpha_E_func':"lambda E,C,rate: rate['alpha_E']*C",
          'mu_C_func':"lambda C_est,rate: rate['mu_C']",
#          'mu_E_func':"lambda E_traj,C: 1",
          'architecture':architecture,
          'horizon_dur':0.1,
          'horizon_var':'E',
          'x1_est':"lambda e_traj,tt,rate: (e_traj[-1]-e_traj[0])/(tt[-1]-tt[0])/rate['alpha_C']"
          }
    
    rate.update(param['rate'])
    print(rate)
    input_var="lambda t:0."
    return state_init,rate,input_var
    
def plot_time_traj(fN):
    aa=pd.DataFrame.from_csv(dataFolder+fN)
    t_cutoff=aa['t'].values[-1]/2
    t_mask=aa['t']>t_cutoff
    tt=(aa['t'].loc[t_mask]).values
    for var in aa.keys()[1:]:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(tt,(aa[var].loc[t_mask]).values,'-k')
        ax.set_xlabel('time (min)')
        ax.set_ylabel(var)
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc=1, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
        plt.savefig(fN+'_'+var+'_traj.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  

def continuous_bound_plot(picFN,newDic_list,newRate_list,prevPicDicFN='simple',x1var='C',x2var='E'):
    # given newDic, which represents a list of new data points, 
    # make a new plot using all the old data points form prevPicDicFN, as well as the new ones.
    # Then save the resulting whole data file into prevPicDicFN.
    # this way this plot can continue to grow...
    key_ordering=('architecture','alpha_C_func','alpha_E_func','mu_C_func','x1_est_func')
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
    
############ x-axis is n2/n1 ##########################
    fig=plt.figure()
    ax=fig.add_subplot(111)
    clr_dic={'deg':'k','noDeg':'g','new':'b'}
    bound_clr_dic={'fmax_bound':'r'}
    
    f_fold=3
    fmax_bound_func=lambda f_fold,n2:1/(n2*np.log(f_fold)+1)
        
    for row in pic_dic.values():
        ax.plot(row['n2/n1'],row['fano'],'o',markersize=10,color=clr_dic[row['architecture']])
        ax.plot(row['n2/n1'],fmax_bound_func(f_fold,row['n2']),'v',markersize=7,color=bound_clr_dic['fmax_bound'])
    for row in newDic_list:
        ax.plot(row['n2/n1'],row['fano'],'o',markersize=10,color=clr_dic['new'])
    [ax.plot([],[],'o',markersize=10,color=val,label=key) for key,val in clr_dic.items()]
    ax.plot([],[],'v',markersize=7,color='r',label='fmax_bound:'+str(f_fold))

    xx=np.logspace(-5,3,num=100)
    paulsson_bound_func=lambda x:2/(1+np.sqrt(1+4*x))
    ax.plot(xx,paulsson_bound_func(xx),'-r',lw=2,label='Paulsson Bound')
    ax.set_xlabel('N2/N1')
    ax.set_ylabel('Fano for '+x1var)
    ax.set_yscale('log')
    ax.set_xscale('log')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1.,1.02),borderaxespad=0.0)
    plt.savefig(picFN+'_x1_'+x1var+'_x2_'+x2var+'_n2n1plot.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  

############ x-axis is C ###############################
    fig=plt.figure()
    ax=fig.add_subplot(111)
    clr_dic={'deg':'k','noDeg':'g'}
    paulsson_bound_func = lambda C,tau:1/(C*tau+1)
#    yorie_DTCS_bound_func = lambda C,lam,x1_mean:lam/(2**(2*C)-1)*x1_mean
#    yorie_CTCS_bound_func = lambda C,lam,x1_mean:lam/(2*C)*x1_mean
    c_func= lambda f_mean,f_var:f_mean*np.log(1+f_var/f_mean**2)
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    clr_dic={'deg':'k','noDeg':'g','new':'b'}
    bound_clr_dic={'paulsson_bound':'r','yorie_DTCS_bound':'c','yorie_CTCS_bound':'m'}
    for row in pic_dic.values():
        cc=c_func(row['f_mean'],row['f_var'])
        ax.plot(cc,row['fano'],'o',markersize=10,color=clr_dic[row['architecture']])
        ax.plot(cc,paulsson_bound_func(cc,row['tau']),'v',markersize=7,color=bound_clr_dic['paulsson_bound'])
#        ax.plot(cc,yorie_DTCS_bound_func(cc,row['lambda'],row['x1_mean']),'v',markersize=7,color=bound_clr_dic['yorie_DTCS_bound'])
#        ax.plot(cc,yorie_CTCS_bound_func(cc,row['lambda'],row['x1_mean']),'v',markersize=7,color=bound_clr_dic['yorie_CTCS_bound'])
        
    for row in newDic_list:
        ax.plot(cc,row['fano'],'o',markersize=10,color=clr_dic['new'])
    [ax.plot([],[],'o',markersize=10,color=val,label=key) for key,val in clr_dic.items()]
    [ax.plot([],[],'v',markersize=7,color=val,label=key) for key,val in bound_clr_dic.items()]

    ax.set_xlabel('Channel Capacity')
    ax.set_ylabel('Fano for '+x1var)
    ax.set_yscale('log')
    ax.set_xscale('log')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1.,1.02),borderaxespad=0.0)
    plt.savefig(picFN+'_x1_'+x1var+'_x2_'+x2var+'_Cplot.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  
    ax.set_ylim([1e-5,1e2])
    plt.savefig(picFN+'_x1_'+x1var+'_x2_'+x2var+'_Cplot_zoom.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  
    
def get_dic(fN,x1var,x2var,x2_birth_event_keys,architecture='noDeg'):
    aa=pd.DataFrame.from_csv(dataFolder+fN)
    t_mask=aa['t']>t_cutoff
    t_weight=np.diff(aa['t'].loc[t_mask])
    dur=aa['t'].values[-1]-t_cutoff
    x1x1=aa[x1var].loc[t_mask]
    x1x1=x1x1.values[:-1]
    x2x2=aa[x2var].loc[t_mask]
    x2x2=x2x2.values[:-1]
    x2_mean=np.sum(x2x2*t_weight)/dur
    x1_mean=np.sum(x1x1*t_weight)/dur
    x1_sq=np.sum(x1x1**2*t_weight)/dur
    if x1_mean==0:pdb.set_trace()
    x1_var=x1_sq-x1_mean**2
    
    temp=pickle.load(open(dataFolder+fN+'_rate.p','rb'))
    rate=temp['rate']
    
    if rate['architecture']=='noDeg':
        tau=1/rate['mu_C']
        print(tau)
    elif rate['architecture']=='deg':
        tau=dur/np.sum(eval(rate['mu_C_func'])(x2x2)*t_weight)
    else:
        raise(ValueError('Architecture option not allowed'))

    alpha=rate['alpha_E'] # this estimation only works if alpha_E_func = alpha * x1.
    f_mean=x1_mean*alpha
    f_max=max(x1x1*alpha)
    x2_birth=np.sum(np.diff(x2x2)==1)
    
#    f_mean2=x2_birth/dur
#    print(f_mean2-f_mean)

#    ff= eval(rate['alpha_E_func'])(x2x2,x1x1)
#    f_mean = np.sum(ff*t_weight)/dur
#    f_max=max(ff)
    
    
    n2=f_mean*tau
    
    f_var=x1_var*alpha**2
    
#    n_batch=10
#    t_max=aa['t'].values[-1]
#    batch_t_size=(t_max-t_cutoff)/n_batch
#    x1_means=np.zeros(n_batch)
#    x1_sqs=np.zeros(n_batch)
#    for t,i in zip(np.arange(t_cutoff,t_max,batch_t_size),range(n_batch)):
#        t_mask=(aa['t'].values>t) & (aa['t'].values<t+batch_t_size)
#        x1x1=aa[x1var].loc[t_mask]
#        x1x1=x1x1.values[:-1]
#        t_weight=np.diff(aa['t'][t_mask])
#        x1_means[i]=np.sum(x1x1*t_weight)/batch_t_size
#        x1_sqs[i]=np.sum(x1x1**2*t_weight)/batch_t_size
#    pdb.set_trace()        
    
#    lam=rate['alpha_C'] # this only works if alpha_C_func = lambda, constant.
#    ('lambda',lam),
    dic=dict([('T',temp['T']),('tau',tau),('f_mean',f_mean),
               ('f_var',f_var),('f_max',f_max),('x1_var',x1_var),('x1_mean',x1_mean),
               ('x2_mean',x2_mean),('x2_birth',x2_birth),('n2',n2),
               ('architecture',rate['architecture']),('n2/n1',n2/x1_mean),
               ('fano',x1_var/x1_mean)])
    return dic,rate

    
def cv_sim_par(arg):
    fN=arg['fN']
    T=arg['T']
    init_state,rate,input_var=init_simple_default(arg['param'],architecture=arg['param']['architecture'])
    rate_dict={'init_state':init_state,'rate':rate,'input_var':input_var,'T':T}
    pickle.dump(rate_dict,open(dataFolder+fN+'_rate.p','wb'))
    input_var=eval(input_var)
    for key in ['alpha_C_func','alpha_E_func','mu_C_func','x1_est_func']:
        rate[key]=eval(rate[key])
    g_sim=gillespie(init_state,rate,input_var,al_simple)
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

def param_random_gen(N,rate_bound_dic):
    # rate_bound_dic is a dictionary of the foramt {'k_E':(min,max)}
    pdb.set_trace()

 
    

if __name__=='__main__':
    dataFolder='./data/'


#    #########################################
#    ###   continuous plot              ######
#    #########################################
#    T=500
#    t_cutoff=T*1/2.
#    nCore=1
#    
#    architecture='noDeg'
#    rate={'alpha_C':10,
#          'alpha_E':100,
#          'mu_C':1,
#          'alpha_C_func':"lambda C_est,rate: (C_est<10)*abs(C_est-10)**2*20",
##          'alpha_C_func':"lambda C_est,rate: (C_est<10)*20",
##          'alpha_C_func':"lambda C_est,rate: rate['alpha_C']",
#          'horizon_dur':0.1,
#          'horizon_var':'E',
#          'x1_est_func':"lambda e_traj,tt,rate: (e_traj[-1]-e_traj[0])/(tt[-1]-tt[0])/rate['alpha_E']"
#          }
#
#    param_list=[{'rate':rate,'architecture':architecture,'state':{}}]
#    prefix='simple_continuousPlot_noDeg_'
#
#    fNs=[prefix+str(i)+x['architecture'] for x,i in zip(param_list,range(len(param_list)))]
#    cv_sim(fNs,param_list,nCore,T,dataFolder)
#    
#    plot_time_traj(fNs[0])
#    
#    x1var='C';x2var='E';x2_birth_event_keys=[(('E',+1),)]
#    temp=[get_dic(fN,x1var,x2var,x2_birth_event_keys) for fN in fNs]
#    newDic_list=[x[0] for x in temp]
#    newRate_list=[x[1] for x in temp]
##
#    picFN=prefix+'bound_'
#    continuous_bound_plot(picFN,newDic_list,newRate_list,prevPicDicFN='simple_noDeg_')
#    
    
    
    
#    continuous_bound_plot(picFN,[],[],prevPicDicFN='simple_ctsPlot_bound_noDeg')




    #########################################
    ###   deg time traj                ######
    #########################################
#    state_init,rate,input_var=init_deg_default()
#    g_sim=gillespie(state_init,rate,input_var,al_deg)
#    fN='test_deg'
#    T=5000
#    sim(g_sim,fN,T,ddtaFolder)
#    plot_time_traj(fN)
#    t_cutoff=T/2.
#    plot_bound(fN,'C','E',[fN],t_cutoff,[5.],'k',state_init,rate,input_var)
    
    
    #########################################
    ###   noDeg time traj              ######
    #########################################
#    state_init,rate,input_var=init_noDeg_default()
#    g_sim=gillespie(state_init,rate,input_var,al_noDeg)
#    fN='test_noDeg'
#    T=5000
#    sim(g_sim,fN,T)
#    plot_time_traj(fN)
#    t_cutoff=T/2.
#    plot_bound(fN,'C','E',[fN],t_cutoff,[5.],'b',rate,input_var)
    
    #########################################
    ###   varying parameters           ######
    #########################################
#    T=5000
#    t_cutoff=T*1/2.
#    nCore=4
#    
#    param_random_gen(N)
#    
#    param_list=[{'rate':{'alpha_CE':x},'state':{}} for x in alpha_CE_list]
#    prefix='simple_alphaCE_'
#    fN_suffix_is_deg={True:'deg_',False:'noDeg_'}
#    
#    fN_dic={'deg':[prefix+fN_suffix_is_deg[True]+'_'+str(i) for i in range(len(param_list))]}
#    fN_dic['noDeg']=[prefix+fN_suffix_is_deg[False]+'_'+str(i) for i in range(len(param_list))]
#    
#    cv_sim(fN_dic,param_list,nCore,T,dataFolder)
#    
#    input_var_dic={'deg':{},'noDeg':{}}
#    clr_dic={'deg':'b','noDeg':'k'}
#    _,rate,_=init_deg_default(param_list[0])
#    plot_compare('simple_alphaCE_compare_','C','E',fN_dic,t_cutoff,param_list,clr_dic,rate,input_var_dic)
#    plot_time_traj(fNs[0])

