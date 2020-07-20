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
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pandas as pd
import time
from multiprocessing import Pool
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
    key_ordering=('architecture','alpha_C','mu_C','alpha_E','alpha_C_func','alpha_E_func','mu_C_func','x1_est_func','experiment_id')
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
        ax.plot(row['Nc'],row['fano'],'o',markersize=7,color=clr_dic[row['architecture']])
    for row in newDic_list:
        ax.plot(row['Nc'],row['fano'],'o',markersize=7,color=clr_dic['new'])
#    [ax.plot([],[],'o',markersize=10,color=val,label=key) for key,val in clr_dic.items()]

    xx=np.logspace(-1,3.2,num=100)
    paulsson_bound_func=lambda x:2/(1+np.sqrt(1+4*x))
    ax.plot(xx,paulsson_bound_func(xx),'-r',lw=2,label='Theoretical Bound')
    ax.set_xlabel('Nc')
    ax.set_ylabel('Var(x)/Mean(x)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([1e-1,1.5e3])
    ax.set_ylim([1e-2,2])
#    handles, labels = ax.get_legend_handles_labels()
#    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1.,1.02),borderaxespad=0.0)
#    plt.savefig(picFN+'_x1_'+x1var+'_x2_'+x2var+'_ncplot_both.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  
    plt.savefig(picFN+'_x1_'+x1var+'_x2_'+x2var+'_ncplot_both.png', bbox_inches='tight')  


############ x-axis is Nc, Paulsson bound ##########################
#    fig=plt.figure()
#    ax=fig.add_subplot(111)
#    clr_dic={'deg':'k','noDeg':'g','new':'b'}
##    bound_clr_dic={'fmax_bound':'r'}
##    f_fold=3
##    fmax_bound_func=lambda f_fold,n2:1/(n2*np.log(f_fold)+1)
#    for row in pic_dic.values():
#        ax.plot(row['Nc_paulsson'],row['fano'],'o',markersize=10,color=clr_dic[row['architecture']])
##        ax.plot(row['n2/n1'],fmax_bound_func(f_fold,row['n2']),'v',markersize=7,color=bound_clr_dic['fmax_bound'])
#    for row in newDic_list:
#        ax.plot(row['Nc_paulsson'],row['fano'],'o',markersize=10,color=clr_dic['new'])
#    [ax.plot([],[],'o',markersize=10,color=val,label=key) for key,val in clr_dic.items()]
##    ax.plot([],[],'v',markersize=7,color='r',label='fmax_bound:'+str(f_fold))
#
#    xx=np.logspace(-3,3,num=100)
#    paulsson_bound_func=lambda x:2/(1+np.sqrt(1+4*x))
#    ax.plot(xx,paulsson_bound_func(xx),'-r',lw=2,label='Paulsson Bound')
#    ax.set_xlabel('Nc')
#    ax.set_ylabel('Var('+x1var+')/Mean('+x1var+')')
#    ax.set_yscale('log')
#    ax.set_xscale('log')
#    handles, labels = ax.get_legend_handles_labels()
#    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1.,1.02),borderaxespad=0.0)
#    plt.savefig(picFN+'_x1_'+x1var+'_x2_'+x2var+'_ncplot_paulsson.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  

############# x-axis is Nc, Yorie bound ##########################
#    fig=plt.figure()
#    ax=fig.add_subplot(111)
#    clr_dic={'deg':'k','noDeg':'g','new':'b'}
#    for row in pic_dic.values():
#        ax.plot(row['Nc_yorie'],row['fano'],'o',markersize=10,color=clr_dic[row['architecture']])
#    for row in newDic_list:
#        ax.plot(row['Nc_yorie'],row['fano'],'o',markersize=10,color=clr_dic['new'])
#    [ax.plot([],[],'o',markersize=10,color=val,label=key) for key,val in clr_dic.items()]
#
#    xx=np.logspace(-3,3,num=100)
#    yorie_bound_func=lambda x:1/(1+np.sqrt(1+2*x))
#    ax.plot(xx,yorie_bound_func(xx),'-r',lw=2,label='Yorie Bound')
#    ax.set_xlabel('Nc')
#    ax.set_ylabel('Var('+x1var+')/Mean('+x1var+')')
#    ax.set_yscale('log')
#    ax.set_xscale('log')
#    handles, labels = ax.get_legend_handles_labels()
#    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1.,1.02),borderaxespad=0.0)
#    plt.savefig(picFN+'_x1_'+x1var+'_x2_'+x2var+'_ncplot_yorie.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  


############ x-axis is C ###############################
#    fig=plt.figure()
#    ax=fig.add_subplot(111)
#    clr_dic={'deg':'k','noDeg':'g'}
#    paulsson_bound_func = lambda C,tau:1/(C*tau+1)
#    yorie_bound_func = lambda C,lam,x1_mean:1/(2*(1+C*x1_mean/lam))
#    c_func= lambda f_mean,f_var:f_mean*np.log(1+f_var/f_mean**2)
#    
#    fig=plt.figure()
#    ax=fig.add_subplot(111)
#    clr_dic={'deg':'k','noDeg':'g','new':'b'}
#    bound_clr_dic={'paulsson_bound':'r','yorie_bound':'c'}
#    for row in pic_dic.values():
#        cc=c_func(row['f_mean'],row['f_var'])
#        ax.plot(cc,row['fano'],'o',markersize=10,color=clr_dic[row['architecture']])
#        ax.plot(cc,paulsson_bound_func(cc,row['tau']),'v',markersize=7,color=bound_clr_dic['paulsson_bound'])
#        ax.plot(cc,yorie_bound_func(cc,row['lambda'],row['x1_mean']),'v',markersize=7,color=bound_clr_dic['yorie_bound'])
#        
#    for row in newDic_list:
#        ax.plot(cc,row['fano'],'o',markersize=10,color=clr_dic['new'])
#    [ax.plot([],[],'o',markersize=10,color=val,label=key) for key,val in clr_dic.items()]
#    [ax.plot([],[],'v',markersize=7,color=val,label=key) for key,val in bound_clr_dic.items()]
#
#    ax.set_xlabel('Channel Capacity')
#    ax.set_ylabel('Var('+x1var+')/Mean('+x1var+')')
#    ax.set_yscale('log')
#    ax.set_xscale('log')
#    handles, labels = ax.get_legend_handles_labels()
#    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1.,1.02),borderaxespad=0.0)
#    plt.savefig(picFN+'_x1_'+x1var+'_x2_'+x2var+'_Cplot.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  
#    ax.set_ylim([1e-5,1e2])
#    plt.savefig(picFN+'_x1_'+x1var+'_x2_'+x2var+'_Cplot_zoom.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  
    
def get_dic(fN,x1var,x2var,x1estVar,t_cutoff_ratio,architecture='noDeg'):
    aa=pd.DataFrame.from_csv(dataFolder+fN)
    if np.isnan(aa['t'].values[-1]):
        ended_premature=True
        
    else:
        ended_premature=False
    t_cutoff=t_cutoff_ratio*aa['t'].values[-2]
    t_mask=aa['t']>t_cutoff
    tt=aa['t'].loc[t_mask]
    t_weight=np.diff(tt.values[:-1])
    dur=aa['t'].values[-2]-t_cutoff
    x1x1=aa[x1var].loc[t_mask]
    x1x1=x1x1.values[:-2]
    x2x2=aa[x2var].loc[t_mask]
    x2x2=x2x2.values[:-2]
    x1x1_est=aa[x1estVar].loc[t_mask]
    x1x1_est=x1x1_est.values[:-2]
    x2_mean=np.sum(x2x2*t_weight)/dur
    x1_mean=np.sum(x1x1*t_weight)/dur
    x1_sq=np.sum(x1x1**2*t_weight)/dur
    if x1_mean==0:pdb.set_trace()
    x1_var=x1_sq-x1_mean**2
    temp=pickle.load(open(dataFolder+fN+'_rate.p','rb'))
    rate=temp['rate']
    
    alpha=rate['alpha_E'] # this estimation only works if alpha_E_func = alpha * x1.
    f_mean=x1_mean*alpha
    f_max=max(x1x1*alpha)
    f_var=x1_var*alpha**2

    x2_birth=np.sum(np.diff(x2x2)==1)
    
    if rate['architecture']=='noDeg':
        tau=1/rate['mu_C']
        lam=dur/np.sum(eval(rate['alpha_C_func'])(x1x1_est,rate)*t_weight)
        Nc=alpha*tau
        esterr=Nc-f_mean*tau/x1_mean
        Nc_yorie=alpha*x1_mean/lam
        print("tau="+str(tau)+', effective lambda='+str(lam),' Nc_paulsson='+str(Nc)+', est error of two methods='+str(esterr)+', Nc_yorie='+str(Nc_yorie))
    elif rate['architecture']=='deg':
        tau=dur/np.sum(eval(rate['mu_C_func'])(x1x1_est,rate)*t_weight)
        lam=rate['alpha_C'] # this only works if alpha_C_func = lambda, constant.
        
        esterr=Nc_yorie-f_mean/lam
        Nc=alpha*tau
        print("effective tau="+str(tau)+', lambda='+str(lam),' Nc+yorie='+str(Nc_yorie)+', est error of two methods='+str(esterr)+', Nc_paulsson='+str(Nc))
    if rate['architecture']=='both':
        tau=dur/np.sum(eval(rate['mu_C_func'])(x1x1_est,rate)*t_weight)
        lam=dur/np.sum(eval(rate['alpha_C_func'])(x1x1_est,rate)*t_weight)
        Nc=alpha*tau
        Nc_yorie=alpha*x1_mean/lam
        print("tau="+str(tau)+', effective lambda='+str(lam),' Nc='+str(Nc)+', NC_yorie='+str(Nc_yorie))
    else:
        raise(ValueError('Architecture option not allowed'))

    
    dic=dict([('lambda',lam),('T',temp['T']),('tau',tau),('f_mean',f_mean),
               ('f_var',f_var),('f_max',f_max),('x1_var',x1_var),('x1_mean',x1_mean),
               ('x2_mean',x2_mean),('x2_birth',x2_birth),
               ('architecture',rate['architecture']),('Nc',Nc),
               ('Nc_yorie',Nc_yorie),('fano',x1_var/x1_mean)])
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

#####################################################
#### ULTIMATE BOUND PLOT!!!!!!!!!!!!!################
#####################################################

    T=500
    t_cutoff_ratio=1/5.
    nCore=1
    
#    ratio_list=np.logspace(1,2,10)
    ratio_list=np.logspace(1,2,4)
    alpha_C=10
    
    architecture='both'
    rate_func=lambda alpha_C,ratio:{'alpha_C':alpha_C,
          'alpha_E':alpha_C*ratio,
          'mu_C':10,
#          'alpha_C_func':"lambda C_est,rate: rate['alpha_C']",
          'alpha_C_func':"lambda C_est,rate: (C_est<10)*(10-C_est)**2*10",
#          'alpha_C_func':"lambda C_est,rate: (C_est<10)*20",
#          'alpha_C_func':"lambda C_est,rate: rate['alpha_C']",
          'mu_C_func':"lambda C_est,rate: (C_est>10)*(C_est-10)**2*0.6",
#          'mu_C_func':"lambda C_est,rate: (C_est>10)*2",
#          'mu_C_func':"lambda C_est,rate: rate['mu_C']",
          'horizon_dur':.1,
          'horizon_var':'E',
          'x1_est_func':"lambda e_traj,tt,rate: (e_traj[-1]-e_traj[0])/(tt[-1]-tt[0])/rate['alpha_E']",
          'experiment_id':'try_go_lower_horizon_0.1'
          }

    param_list=[{'rate':rate_func(alpha_C,ratio),'architecture':architecture,'state':{}} for ratio in ratio_list]

#    rate={'alpha_C':20,
#          'alpha_E':10,
#          'mu_C':1,
#          'alpha_C_func':"lambda C_est,rate: rate['alpha_C']",
#          'alpha_C_func':"lambda C_est,rate: (C_est<5)*(5-C_est)**2*10",
##          'alpha_C_func':"lambda C_est,rate: (C_est<10)*20",
##          'alpha_C_func':"lambda C_est,rate: rate['alpha_C']",
#          'mu_C_func':"lambda C_est,rate: (C_est>5)*(C_est-5)**2*1",
##          'mu_C_func':"lambda C_est,rate: rate['mu_C']",
#          'horizon_dur':1,
#          'horizon_var':'E',
#          'x1_est_func':"lambda e_traj,tt,rate: (e_traj[-1]-e_traj[0])/(tt[-1]-tt[0])/rate['alpha_E']"
#          }
#    param_list=[{'rate':rate,'architecture':architecture,'state':{}}]
    
    
    
    prefix='simple_continuousPlot_both_'

    fNs=[prefix+str(i)+x['architecture'] for x,i in zip(param_list,range(len(param_list)))]
    print(fNs)
#    cv_sim([fNs[0]],[param_list[0]],nCore,T,dataFolder)
#    cv_sim(fNs,param_list,nCore,T,dataFolder)
    
#    plot_time_traj(fNs[0])
    fNs=fNs[:1]
    x1var='C';x2var='E';x1estVar='C_est';x2_birth_event_keys=[(('E',+1),)]
    temp=[get_dic(fN,x1var,x2var,x1estVar,t_cutoff_ratio) for fN in fNs]
    newDic_list=[x[0] for x in temp]
    newRate_list=[x[1] for x in temp]
    picFN=prefix+'bound_'
#    continuous_bound_plot(picFN,newDic_list,newRate_list,prevPicDicFN='simple_both_')

    continuous_bound_plot(picFN,[],[],prevPicDicFN='simple_both_')



    #########################################
    ###   continuous plot              ######
    #########################################
#    T=500
#    T=500
#    t_cutoff=T*1/2.
#    nCore=1
#    
#    ratio_list=np.logspace(1e-3,3,20)
#    alpha_C=10
#    
#    architecture='both'
#    rate={'alpha_C':alpha_C,
#          'alpha_E':0.,
#          'mu_C':1,
#          'alpha_C_func':"lambda C_est,rate: (C_est<10)*20",
##          'alpha_C_func':"lambda C_est,rate: (C_est<10)*20",
##          'alpha_C_func':"lambda C_est,rate: rate['alpha_C']",
#          'mu_C_func':"lambda C_est,rate: (C_est>10)*2",
#          'horizon_dur':0.1,
#          'horizon_var':'E',
#          'x1_est_func':"lambda e_traj,tt,rate: (e_traj[-1]-e_traj[0])/(tt[-1]-tt[0])/rate['alpha_E']"
#          }
#
#    param_list=[{'rate':rate,'architecture':architecture,'state':{}}]
#    prefix='simple_continuousPlot_both_'
#
#    fNs=[prefix+str(i)+x['architecture'] for x,i in zip(param_list,range(len(param_list)))]
#    cv_sim(fNs,param_list,nCore,T,dataFolder)
#    
#    plot_time_traj(fNs[0])
#    
#    x1var='C';x2var='E';x1estVar='C_est';x2_birth_event_keys=[(('E',+1),)]
#    temp=[get_dic(fN,x1var,x2var,x1estVar,x2_birth_event_keys) for fN in fNs]
#    newDic_list=[x[0] for x in temp]
#    newRate_list=[x[1] for x in temp]
##
#    picFN=prefix+'bound_'
#    continuous_bound_plot(picFN,newDic_list,newRate_list,prevPicDicFN='simple_both_')
    
    
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

