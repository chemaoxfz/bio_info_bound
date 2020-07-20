#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:23:39 2017

@author: xfz
"""

# simulate khammash system, compute the bounds, plot the scaling of CV vs the flux.

# Then, maybe simulate the heat shock system and do the same.

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

class gillespie:
    def __init__(self,init_state,rate,input_var,al_create,state_approx=lambda obj,state,rate,input_var:{}):
        self.t=0.
        self.state,self.rate,self.input_var=init_state,rate,input_var
        #input_var takes t returns a dictionary
        self.horizon=[(self.t,self.state[self.rate['horizon_var']])]
        self.al_create=al_create
        self.al=al_create(self.state,self.rate,self.input_var(self.t),self.horizon)
        self.state_approx=state_approx
    
    def step(self):
        r,t=self.draw(self.al)
        self.act(r)
        self.state=dict(list(self.state.items())+list(self.state_approx(self.state,self.rate,self.input_var(self.t),t).items()))
        self.t+=t
        self.horizon+=[(self.t,self.state[self.rate['horizon_var']])]
        self.horizon=self.horizon[-2:]
        self.al=self.al_create(self.state,self.rate,self.input_var(self.t),self.horizon)
        return self.state,self.t,r,self.al

#    def step_old(self):
#        r,t=self.draw(self.al)
#        self.act(r)
#        self.state=dict(list(self.state.items())+list(self.state_approx(self.state,self.rate,self.input_var(self.t),t).items()))
#        self.t+=t
#        self.horizon+=[(self.t,self.state[self.rate['horizon_var']])]
#        self.horizon=self.horizon[-2:]
#        self.al=self.al_create(self.state,self.rate,self.input_var(self.t),self.horizon)
#        return self.state,self.t,r,self.al

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

def sim_old(g_sim,fN,T,dataFolder):
    stream=open(dataFolder+fN,'w')
    st=time.time()
    counter=0
    keys=['t']+list(g_sim.state.keys())
    stream.write(','+','.join(keys)+'\n')
    
    event_dict=dict(zip(g_sim.al.keys(),[0]*(len(g_sim.al.keys()))))
    t=0.
    
    stepsize=rate['stepsize']
    while t<T:
        tt=0
        t_start=g_sim.t
        while tt<stepsize:
            state,t,r,al=g_sim.step()
            stream.write(str(counter)+','+','.join([str(t)]+[str(state[x]) for x in keys[1:]])+'\n')
            event_dict[r]+=1
            counter+=1
            tt=t-t_start
        g_sim.horizon+=[(g_sim.t,g_sim.state[g_sim.rate['horizon_var']])]
        g_sim.step()
        g_sim.horizon=g_sim.horizon[-1:]

    stream.close()
#    stream_al.close()
    pickle.dump(event_dict,open(dataFolder+fN+'_event.p','wb'))
    ed=time.time()
    print(ed-st)
    
    
def sim(g_sim,fN,T,dataFolder):
    stream=open(dataFolder+fN,'w')
    st=time.time()
    counter=0
    keys=['t']+list(g_sim.state.keys())
    stream.write(','+','.join(keys)+'\n')
    
#    stream_al=open(dataFolder+fN+'_al','w')
#    keys_al=['t']+list(g_sim.al.keys())
#    stream_al.write(','+','.join([str(x) for x in keys_al])+'\n')
    
    event_dict=dict(zip(g_sim.al.keys(),[0]*(len(g_sim.al.keys()))))
    t=0.
    while t<T:
        state,t,r,al=g_sim.step()
        stream.write(str(counter)+','+','.join([str(t)]+[str(state[x]) for x in keys[1:]])+'\n')
#        stream_al.write(str(counter)+','+','.join([str(t)]+[str(al[x]) for x in keys_al[1:]])+'\n')
        event_dict[r]+=1
        counter+=1
#        if t>1: print(g_sim.al);pdb.set_trace()
#        print(str(t)+', '+str(t/T),end='\r')
#        pdb.set_trace()
    stream.close()
#    stream_al.close()
    pickle.dump(event_dict,open(dataFolder+fN+'_event.p','wb'))
    ed=time.time()
    print(ed-st)


def al_simple(state,rate,input_var,horizon):
    def derivative(x_vec,t,rate,state,dt,dC):
        uu=rate['alpha_X_func'](state['X_est'],rate)
        tau=1/rate['mu_X']
        dy=dC/dt
        x_mean,x_est,pp=x_vec
        dX_mean=uu-x_mean/tau
        dX_est=-x_est/tau+uu+pp/x_mean*(dy-rate['alpha_C']*x_est)
        ########################## ABOVE LINE IS WRONG. (dy - ... dt) is the right thing to do... But I guess this is the right numerical implementation... 
        #########################################################################
        dP= -2*pp/tau + uu + x_mean/tau - pp**2*rate['alpha_C']/x_mean
        return [dX_mean,dX_est,dP]
    
    tt,c_traj=np.array(horizon).T
    try:
        # this is with control action.
        tt[1]
        dt=tt[1]-tt[0]
        ## compute dK and new K
        
        
#        state['X_mean']=uu*tau*(1-np.exp(-dt/tau))+state['X_mean']*np.exp(-dt/tau)
#        dP= (- 2*state['P']/tau + uu+state['X_mean']/tau - state['P']**2*rate['alpha_C']/state['X_mean'])  * dt
#        state['P']=state['P']+dP
#        state['K']=state['P']/state['X_mean']
#        dC=c_traj[1]-c_traj[0]
#        state['dX_est']=dt*(-state['X_est']*rate['mu_X']+uu+state['K']*(dC-rate['alpha_C']*state['X_est']))
#        state['X_est']=state['X_est']+state['dX_est']
        
        ##### a better odeint calculation ###########
        init=[state['X_mean'],state['X_est'],state['P']]
        ts=[0,dt]
        dC=c_traj[1]-c_traj[0]
        _,aa=odeint(derivative,init,ts,args=(rate,state,dt,dC))
        state['X_mean'],state['X_est'],state['P']=aa

#        pdb.set_trace()
        al={(('X',+1),):rate['alpha_X_func'](state['X_est'],rate),
            (('X',-1),):rate['mu_X']*state['X'],
            (('C',+1),):rate['alpha_C']*state['X']
            }
#        if state['X']>150:pdb.set_trace()
#        if state['X']>120:pdb.set_trace()
#        state['delta_C']=delta_C
    except IndexError:
        # this is when there is only one entry, so it's first step, or no control action.
        al={(('X',+1),):rate['alpha_X_func'](state['X_est'],rate),
            (('X',-1),):rate['mu_X']*state['X'],
            (('C',+1),):rate['alpha_C']*state['X']
            }
    return al

def init_simple_default(param={'state':{},'rate':{}},architecture='noDeg'):
    x_d=20
    state_init={'X':x_d,'C':0,'K':1,'X_est':x_d,'dX_est':0,'P':x_d,'X_mean':x_d}
    state_init.update(param['state'])
    print(state_init)
    # rate_poly follows the format of ((coeffs),(corresponding degrees))
    rate={'alpha_X':30,
          'alpha_C':10,
          'mu_X':1,
#          'alpha_X_func':"lambda x_est,rate:rate['mu_X']*(2*rate['x_d']-x_est)",
#          'alpha_X_func':"lambda x_est,rate:rate['mu_X']*rate['x_d']*max(rate['x_d']-x_est,0)",
          'alpha_X_func':"lambda x_est,rate:rate['mu_X']*(2-x_est/rate['x_d'])**2*rate['x_d']**2",
          'horizon_var':'C',
          'architecture':architecture,
          'x_d':x_d
          }
    rate.update(param['rate'])
    print(rate)
    input_var="lambda t:0."
    return state_init,rate,input_var
    
def plot_time_traj(fN):
    aa=pd.DataFrame.from_csv(dataFolder+fN)
#    t_cutoff=aa['t'].values[-1]/2
    t_cutoff=0
    t_mask=aa['t']>t_cutoff
    tt=(aa['t'].loc[t_mask]).values
    
    plot_keys=['X','X_est','P','K']
#    for var in aa.keys()[1:]:
    for var in plot_keys:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(tt,(aa[var].loc[t_mask]).values,'-k')
        ax.set_xlabel('time ')
        ax.set_ylabel(var)
#        ax.set_ylim([0,200])
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc=1, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
        plt.savefig(fN+'_'+var+'_traj.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  

def continuous_bound_plot(picFN,newDic_list,newRate_list,prevPicDicFN='simple',x1var='X',x2var='C'):
    # given newDic, which represents a list of new data points, 
    # make a new plot using all the old data points form prevPicDicFN, as well as the new ones.
    # Then save the resulting whole data file into prevPicDicFN.
    # this way this plot can continue to grow...
    
    matplotlib.rcParams['axes.labelsize']=25
    
    key_ordering=('architecture','alpha_X','alpha_C','mu_X','experiment_id','x_d')
    fPath=Path(dataFolder+prevPicDicFN+'.p')
#    pdb.set_trace()
    if fPath.is_file():
        pic_dic=pickle.load(open(dataFolder+prevPicDicFN+'.p','rb'))
#        pdb.set_trace()
#        pd.DataFrame.from_csv(dataFolder+prevPicDicFN+'.csv')
    else:
        pic_dic={}
#        picDF=pd.DataFrame()
    if newRate_list and newDic_list:
        # if not empty, add entries. Else, just plot.
        for newRate,newDic in zip(newRate_list,newDic_list):
            try: 
                newRate['experiment_id']
            except KeyError:
                newRate['experiment_id']='old,gillespie'
            temp=tuple(((key,newRate[key]) for key in key_ordering))
            pic_dic[temp]=newDic # if existent, cover. if non-existent, add.
#        pickle.dump(rate_dic,open(dataFolder+prevPicDicFN+'_rate.p','wb'))
#        picDF.to_csv(dataFolder+prevPicDicFN+'.csv')
        pickle.dump(pic_dic,open(dataFolder+prevPicDicFN+'.p','wb'))
    
############ x-axis is n2/n1 ##########################
    fig=plt.figure(figsize=(6, 4.5))
    ax=fig.add_subplot(111)
    clr_dic={'deg':'k','noDeg':'g','new':'b'}
    bound_clr_dic={'fmax_bound':'r'}
    
    f_fold=3
#    fmax_bound_func=lambda f_fold,n2:1/(n2*np.log(f_fold)+1)
        
    for row in pic_dic.values():
        ax.plot(row['n2/n1'],row['fano'],'o',markersize=10,color=clr_dic[row['architecture']])
#        ax.plot(row['n2/n1'],fmax_bound_func(f_fold,row['n2']),'v',markersize=7,color=bound_clr_dic['fmax_bound'])
    for row in newDic_list:
        ax.plot(row['n2/n1'],row['fano'],'o',markersize=13,color=clr_dic['new'])
#        ax.plot(row['n2/n1'],row['fano'],'o',markersize=7,color=clr_dic[row['architecture']])
#    [ax.plot([],[],'o',markersize=10,color=val,label=key) for key,val in clr_dic.items()]
#    ax.plot([],[],'v',markersize=7,color='r',label='fmax_bound:'+str(f_fold))

    xx=np.logspace(-2,3,num=100)
    paulsson_bound_func=lambda x:2/(1+np.sqrt(1+4*x))
    yorie_bound_func=lambda x:2/(1+np.sqrt(1+2*x))
#    ax.set_title('no degradation')
    ax.plot(xx,paulsson_bound_func(xx),'-r',lw=4,label='Lestas 2010 Bound')
    ax.plot(xx,yorie_bound_func(xx),'-b',lw=4,label='Performance Bound')
    ax.set_xlabel(r'$N_C/N_X$')
    ax.set_ylabel(r'Fano$(X)$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=20)
    handles, labels = ax.get_legend_handles_labels()
    ax.set_ylim([1e-1,1.5])
    ax.set_xlim([1e-2,1.2e2])
#    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1.,1.02),borderaxespad=0.0)
#    plt.savefig(picFN+'_x1_'+x1var+'_x2_'+x2var+'_n2n1plot.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  
    plt.savefig(picFN+'_x1_'+x1var+'_x2_'+x2var+'_n2n1plot.png', bbox_inches='tight')  


############ channel capacity is achieved? #############
#    fig=plt.figure()
#    ax=fig.add_subplot(111)
#    cc_poisson_func=lambda row:np.log(1+row['f_var']/row['f_mean']**2)*row['f_mean']
#    cc_gaussian_func=lambda row:0.5*np.log(1+row['f_var']/row['f_mean'])
#    mi_func=lambda row:-0.5*np.log(1-row['x1x2_cov']**2/(row['x1_var']*row['x2_var']))
#    for row in pic_dic.values():
#        ax.plot(row['n2/n1'],cc_gaussian_func(row),'ok',markersize=7)
#        ax.plot(row['n2/n1'],mi_func(row),'ob',markersize=7)
#        ax.plot(row['n2/n1'],cc_poisson_func(row),'or',markersize=7)
#    ax.plot(row['n2/n1'],cc_poisson_func(row),'or',markersize=7,label='Poisson capacity')    
#    ax.plot(row['n2/n1'],cc_gaussian_func(row),'ok',markersize=7,label='Gaussian capacity')
#    ax.plot(row['n2/n1'],mi_func(row),'ob',markersize=7,label='actual capacity')
##    for row in newDic_list:
##        ax.plot(cc,row['fano'],'o',markersize=10,color=clr_dic['new'])
#
#    ax.set_ylabel('Channel Capacity')
#    ax.set_xlabel('N2/N1')
#    ax.set_yscale('log')
#    ax.set_xscale('log')
#    handles, labels = ax.get_legend_handles_labels()
#    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1.,1.02),borderaxespad=0.0)
##    ax.set_ylim([1e-5,1e2])
#    plt.savefig(picFN+'_x1_'+x1var+'_x2_'+x2var+'_capacity.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  
#    
############# rate distorsion is achieved? ###############
#    fig=plt.figure()
#    ax=fig.add_subplot(111)
#    for row in pic_dic.values():
#        ax.plot(mi_func(row),row['fano'],'ok',markersize=7)
#        ax.plot(cc_gaussian_func(row),row['fano'],'ob',markersize=7)
#        ax.plot(cc_poisson_func(row),row['fano'],'or',markersize=7)
#    ax.plot(cc_gaussian_func(row),row['fano'],'ok',markersize=7,label='distortion,Gaussian capacity')
#    ax.plot(mi_func(row),row['fano'],'ob',markersize=7,label='distortion,actual capacity')
#    ax.plot(cc_poisson_func(row),row['fano'],'or',markersize=7,label='distortion,Poisson capacity')
#    xx=np.logspace(0,3,100)
#    ax.plot(xx,1/(xx*0.1+1),'-r',label='distortion,bound')
##    for row in newDic_list:
##        ax.plot(cc,row['fano'],'o',markersize=10,color=clr_dic['new'])
#
#    ax.set_xlabel('Channel Capacity')
#    ax.set_ylabel('fano')
#    ax.set_yscale('log')
#    ax.set_xscale('log')
#    handles, labels = ax.get_legend_handles_labels()
#    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1.,1.02),borderaxespad=0.0)
##    ax.set_ylim([1e-5,1e2])
#    plt.savefig(picFN+'_x1_'+x1var+'_x2_'+x2var+'_distortion.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  
    
def get_dic(fN,x1var,x2var,x2_birth_event_keys,architecture='noDeg'):
    aa=pd.DataFrame.from_csv(dataFolder+fN)
    t_cutoff=aa['t'].values[-1]/5.
    t_mask=aa['t']>t_cutoff
    t_weight=np.diff(aa['t'].loc[t_mask])
    dur=aa['t'].values[-1]-t_cutoff
    x1x1=aa[x1var].loc[t_mask]
    x1x1=x1x1.values[:-1]
    x2x2=aa[x2var].loc[t_mask]
    x2x2=np.diff(x2x2.values)
    x2_mean=np.sum(x2x2*t_weight)/dur
    x1_mean=np.sum(x1x1*t_weight)/dur
    x1_sq=np.sum(x1x1**2*t_weight)/dur
    if x1_mean==0:pdb.set_trace()
    x2_sq=np.sum(x2x2**2*t_weight)/dur
    x2_var=x2_sq-x2_mean**2
    x1_var=x1_sq-x1_mean**2
    
    x1x2_cov=np.sum((x1x1-x1_mean)*(x2x2-x2_mean)*t_weight)/dur
    x1x2_corr=x1x2_cov/(x1_var*x2_var)**(0.5)
    mi_x1x2=-0.5*np.log(1-x1x2_corr**2)
    
    
    temp=pickle.load(open(dataFolder+fN+'_rate.p','rb'))
    rate=temp['rate']
    
    if rate['architecture']=='noDeg':
        tau=1/rate['mu_X']
        print(tau)
    elif rate['architecture']=='deg':
        tau=dur/np.sum(eval(rate['mu_C_func'])(x2x2)*t_weight)
    else:
        raise(ValueError('Architecture option not allowed'))

    alpha=rate['alpha_C'] # this estimation only works if alpha_E_func = alpha * x1.
    f_mean=x1_mean*alpha
    f_max=max(x1x1*alpha)
    n2=f_mean*tau
    
    f_var=x1_var*alpha**2
#    cc_yorie=0.5/rate['stepsize']*np.log(1+rate['stepsize']*f_var/f_mean)
    
#    fano_bound_cc=1/(cc_yorie*tau+1)
#    fano_bound_mi=1/(mi_x1x2*tau+1)
    
    dic=dict([('T',temp['T']),('tau',tau),('f_mean',f_mean),
               ('f_var',f_var),('f_max',f_max),('x1_var',x1_var),('x1_mean',x1_mean),
               ('x2_mean',x2_mean),('x2_var',x2_var),('n2',n2),('x1x2_cov',x1x2_cov),
               ('architecture',rate['architecture']),('n2/n1',n2/x1_mean),
               ('fano',x1_var/x1_mean)])
#               ('mi',mi_x1x2),('cc_theory',cc_yorie)])
    return dic,rate

    
def cv_sim_par(arg):
    fN=arg['fN']
    T=arg['T']
    init_state,rate,input_var=init_simple_default(arg['param'],architecture=arg['param']['architecture'])
    rate_dict={'init_state':init_state,'rate':rate,'input_var':input_var,'T':T}
    pickle.dump(rate_dict,open(dataFolder+fN+'_rate.p','wb'))
    input_var=eval(input_var)
    for key in ['alpha_X_func']:
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

def fix_dic(prevPicDicFN='kalman_controller_continuous_fix_',dataFolder='./data/'):
    pic_dic=pickle.load(open(dataFolder+prevPicDicFN+'.p','rb'))
    # remove key 1
    del pic_dic[list(pic_dic.keys())[1]]
    pickle.dump(pic_dic,open(dataFolder+prevPicDicFN+'.p','wb'))


if __name__=='__main__':
    dataFolder='./data/'


    #########################################
    ###   continuous plot              ######
    #########################################
#    T=2000
#    T=500
#    t_cutoff=T*2/5.
#    nCore=4
#    alpha_C=0.03
#    architecture='noDeg'
#    rate={'alpha_C':alpha_C,
#          'mu_X':1,
#          'horizon_var':'C',
##          'x_d':1,
##          'stepsize':0.001,
#          'experiment_id':'kalman_control_continuous_quad_u_win'
##          'experiment_id':'kalman_gillespie_interval_try_to_win'
#          }
#    param_list=[{'rate':rate,'architecture':architecture,'state':{}}]
##    
    architecture='noDeg'
    rates=[]
    for alpha_c in np.logspace(2.1,3,10):
        rates=rates+[{'alpha_C':alpha_c,
          'mu_X':1,
          'horizon_var':'C',
          'experiment_id':'kalman_control_continuous_quad_u_win',
          }]
    param_list=[{'rate':rate,'architecture':architecture,'state':{}} for rate in rates]
    

#    prefix='kalman_controller_continuous_small_molecule_'
#    prefix='kalman_controller_continuous_more_concrete_'
#    prefix='kalman_controller_continuous_abunch_'
    prefix='kalman_controller_continuous_fix_'

    fNs=[prefix+str(i)+x['architecture'] for x,i in zip(param_list,range(len(param_list)))]
#    cv_sim(fNs,param_list,nCore,T,dataFolder)
    
#    plot_time_traj(fNs[0])
#    fNs=[fNs[8]]
    x1var='X';x2var='C';x2_birth_event_keys=[(('C',+1),)]
    temp=[get_dic(fN,x1var,x2var,x2_birth_event_keys) for fN in fNs]
    newDic_list=[x[0] for x in temp]
    newRate_list=[x[1] for x in temp]
    picFN=prefix+'bound_test_'
#    continuous_bound_plot(picFN,newDic_list,newRate_list,prevPicDicFN='kalman_controller_continuous_fix_')
    
    
    
    
    continuous_bound_plot(picFN,[],[],prevPicDicFN='kalman_controller_continuous_fix_')

