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
import pickle

class gillespie:
    def __init__(self,init_state,rate,al_create,particle_gen,keykey):
        self.t=0.
        self.state,self.rate=init_state,rate
        self.al_create=al_create
        self.al=al_create(self.state,self.rate)
        self.particles=particle_gen(self.rate,self.state,NN=100)
        self.keykey=keykey
    
    def step(self):
        r,t=self.draw(self.al)
        self.particles,x_est,mi_est=self.particle_step(self.particles,r,self.rate,self.state,self.keykey)
        self.state['X_est']=x_est
        self.state['MI']=mi_est
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

    def particle_step(self,particles,r,rate,state,keykey='production'):
        def propensityMapping(key,particles,rate,state):
            #key is production or degradation
            if key=='production':
                return rate['alpha_C_func'](particles,rate)
            elif key=='degradation':
                return rate['mu_C_func'](particles,rate,state['C'])*state['C']+1e-23
            else:
                raise ValueError('INVALID PROPENSITY KEY!!!')

        def al_prev_calc(al,keys=[(('X',+1),),(('X',-1),)]):
            return [al[key]/sum(list(al.values())) for key in keys]
        NN=len(particles)
        #compute weights
        signaling=0
        if r==(('C',+1),):signaling=1
        signaling_probs=np.zeros([NN,3])
        order=[+1,-1]
        prev_probs=np.zeros([NN,3])
        for i in range(NN):
            particle=particles[i]
            particle_prev_state={'X':particle,'C':state['C']}
            al_prev=self.al_create(particle_prev_state,self.rate)
            probs=al_prev_calc(al_prev,[(('X',ii),) for ii in order])
            prev_probs[i]=[1-sum(probs)]+probs
#            if np.min(prev_probs[i])<0:pdb.set_trace()
            particle_now_states=[{'X':max(particle+aa,0),'C':state['C']} for aa in order]
            als_now=[al_prev]+[self.al_create(aa,self.rate) for aa in particle_now_states]
            signaling_probs[i]=[aa[(('C',+1),)]/sum(list(aa.values())) for aa in als_now]
        signaling_probs=np.abs(1-signaling-signaling_probs)
        weights_unsummed=prev_probs*signaling_probs
        weights=np.sum(weights_unsummed,1)
        weights_normalized=weights/np.sum(weights)
        #before resampling, compute X_est and MI
        x_est=np.sum(particles*weights_normalized)
        lam_particles=propensityMapping(keykey,particles,rate,state)
        lam_hat=np.sum(lam_particles*weights_normalized)
        
        numerical_error=1e-10
        temp=lam_hat*np.log(lam_hat)
        mask=np.abs(temp)>numerical_error
        second_term=temp*mask
        
        
        temp=lam_particles*np.log(lam_particles)
        mask=np.abs(temp)>numerical_error
        first_term=np.sum(temp*mask*weights_normalized)
        mi_est=first_term - second_term
        
        
        # resampling
        idx_particles_new=np.random.choice(np.arange(NN),size=NN,replace=True,p=weights_normalized)
        transition_probs=weights_unsummed/np.repeat(np.reshape(weights,(NN,1)),3,axis=1)
        # compute new particle step
        changeMap={0:0,1:order[0],2:order[1]}
        particles_new=np.zeros(NN)
        for i in range(NN):
            rr=np.random.rand()
            rIdx=3-np.sum((rr-np.cumsum(transition_probs[idx_particles_new[i]]))<=0)
            if rIdx==3:pdb.set_trace()
            particles_new[i]=max(particles[idx_particles_new[i]]+changeMap[rIdx],0)
#        if self.t>10:
#            pdb.set_trace()
        return particles_new,x_est,mi_est

def particle_gen(rate,state,NN=100):
    # generate particles 
    # yy is the observation molecule current count. Not useful if there is no feedback.
    # use al to construct the right Poisson distribution
    lam=rate['alpha_X_func'](0,rate)/rate['mu_X_func'](0,rate,state['X'])
#    particles=np.random.poisson(lam=lam,size=NN)
    particles=np.linspace(1,100,100)
    
    return particles

def sim(g_sim,fN,T,dataFolder):
    stream=open(dataFolder+fN,'w')
    st=time.time()
    counter=0
    keys=['t']+list(g_sim.state.keys())
    stream.write(','+','.join(keys)+'\n')
    
    event_dict=dict(zip(g_sim.al.keys(),[0]*(len(g_sim.al.keys()))))
    t=0.
    st=time.time()
    dur=0.
    while t<T and dur<3600: #dur limits real time a longest run.
        state,t,r,al=g_sim.step()
        stream.write(str(counter)+','+','.join([str(t)]+[str(state[x]) for x in keys[1:]])+'\n')
        event_dict[r]+=1
        counter+=1
        ed=time.time()
        dur=ed-st
    stream.close()
    pickle.dump(event_dict,open(dataFolder+fN+'_event.p','wb'))
    ed=time.time()
    print(ed-st)


def al_simple(state,rate):
    al={(('X',+1),):rate['alpha_X_func'](0,rate),
        (('X',-1),):rate['mu_X_func'](0,rate,state['X'])*state['X'],
        (('C',+1),):rate['alpha_C_func'](state['X'],rate),
        (('C',-1),):rate['mu_C_func'](state['X'],rate,state['C'])*state['C']
        }
    return al

def init_simple_default(param={'state':{},'rate':{}}):
    state_init={'X':10,'C':100,'X_est':10,'MI':0}
    state_init.update(param['state'])
#    print(state_init)
    # rate_poly follows the format of ((coeffs),(corresponding degrees))
    rate={'alpha_X':10,
          'alpha_C':10,
          'mu_C':1,
          'mu_X':1,
          'alpha_C_func':"lambda x,rate:rate['alpha_C']*x", #add 1e-23 for numerical stability when taking log later when calculgin MI
          'mu_C_func':"lambda x,rate,c:rate['mu_C']*(c>=1)+1e-23",
          'alpha_X_func':"lambda x_est,rate:rate['alpha_X']",
          'mu_X_func':"lambda x_est,rate,x:rate['mu_X']*(x>=1)+1e-23"
          }
    rate.update(param['rate'])
#    print(rate)
    return state_init,rate
    
def plot_time_traj(fN):
    aa=pd.DataFrame.from_csv(dataFolder+fN)
    t_cutoff=aa['t'].values[-1]/5.
    t_mask=aa['t']>t_cutoff
    tt=(aa['t'].loc[t_mask]).values
    plot_keys=['X','C','MI']
#    for var in aa.keys()[1:]:
    for var in plot_keys:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.step(tt,(aa[var].loc[t_mask]).values,'-k',label=var)
        if var=='X':
            ax.step(tt,(aa['X_est'].loc[t_mask]).values,'-b',label='X_est')
        ax.set_xlabel('time ')
        ax.set_ylabel(var)
#        ax.set_ylim([0,200])
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc=1, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
        plt.savefig(fN+'_'+var+'_traj.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  

def plot_MI(fN):
    aa=pd.DataFrame.from_csv(dataFolder+fN)
    try:
        rate=pickle.load(open(dataFolder+fN+'_rate.p','rb'))['rate']
    except UnicodeDecodeError:
        rate=pickle.load(open(dataFolder+fN+'_rate.p','rb'),encoding='latin1')['rate']
    t_cutoff=aa['t'].values[-1]/5.
    t_mask=aa['t']>t_cutoff
    tt=(aa['t'].loc[t_mask]).values
    
    var='MI'
    mi=(aa[var].loc[t_mask]).values
    mi=mi[:-1]
    t_weight=np.diff(tt)
    dur=np.sum(t_weight)
    mi_cum=np.cumsum(mi*t_weight)/np.cumsum(t_weight)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.step(tt[:-1],mi_cum,'-k',label=var)
    ax.set_xlabel('time ')
    ax.set_ylabel(var)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=1, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
    plt.savefig(fN+'_MI_cum.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  
    
    mi_mean=mi_cum[-1]
    xx=(aa['X'].loc[t_mask]).values[:-1]
    xx_mean=np.sum(xx*t_weight)/dur
    xx_var=np.sum((xx-xx_mean)**2*t_weight)/dur
    cc=(aa['C'].loc[t_mask]).values[:-1]
    cc_mean=np.sum(cc*t_weight)/dur
    alpha_c=eval(rate['alpha_C_func'])(xx,rate)
    alpha_c_mean=np.sum(alpha_c*t_weight)/dur
    alpha_c_var=np.sum((alpha_c-alpha_c_mean)**2*t_weight)/dur
    bound=alpha_c_mean*np.log(1+alpha_c_var/alpha_c_mean**2)
    mu_c=eval(rate['mu_C_func'])(xx,rate,cc)
    mu_c_mean=np.sum(mu_c*t_weight)/dur
    mu_c_var=np.sum((mu_c-mu_c_mean)**2*t_weight)/dur
    bound_production=alpha_c_mean*np.log(1+alpha_c_var/alpha_c_mean**2)
    bound_degradation=mu_c_mean*cc_mean*np.log(1+mu_c_var/mu_c_mean**2)
    
#    nBin_xx=int(np.max(xx)-np.min(xx))
#    xx_hist=np.histogram(xx,bins=nBin_xx,normed=True,weights=t_weight)
#    cc_change_idx=np.where(np.diff(cc)!=0)[0]
#    temp=np.pad(cc_change_idx+1,(1,0),'constant',constant_values=0)
#    cc_change_t=np.diff(tt[temp])
#    
#    nBin_cc_change_t=15
#    cc_change_t=np.histogram(cc_change_t,bins=nBin_cc_change_t,normed=True)
#    
#
#    np.histogram2d(xx,cc_change_t,bins=[nBin_xx,nBin_cc_change_t],normed=True,weights=
#    pdb.set_trace()


def plot_MI_efficiency_from_file(data_save_fN):
    matplotlib.rcParams['axes.labelsize']=20
    matplotlib.rcParams['font.family']='sans-serif'
    matplotlib.rcParams['font.serif']='Times New Roman'
    
    dataDic=pickle.load(open(data_save_fN,'rb'))
    fig=plt.figure()
    ax=fig.add_subplot(111)
    colors=dict([(key,color) for key,color in zip(dataDic.keys(),['b','c','k'])])
    for key,val in dataDic.items():
#        ax.plot(val['encodingMean'],val['mi_est']/val['encodingMean'],'o',color=colors[key],label=key)
#        pdb.set_trace()
        ax.plot(val['encodingPropensity'],val['mi_est']/val['encodingPropensity'],'o',color=colors[key],label=key)
#        ax.plot(val['encodingVar']/val['encodingMean'],val['mi_est'],'o',color=colors[key],label=key)
#    xlim=[1e-4,6]
#    xlim=[1e-4,100]
#    xAxis=np.logspace(np.log(xlim[0]),np.log(xlim[1]),1000)
#    ax.plot(xAxis,np.log(1+xAxis),'-r',label='Bound')
    
#    ax.plot(xAxis,xAxis,'-r',label='Bound')
#    ax.set_xlim([xlim[0],15])
#    ax.set_ylim([1e-6,0.5e1])
    ax.set_xlabel('signaling rate')
    ax.set_ylabel('mutual information efficiency')
#    ax.set_xlabel('Fano(v)')
#    ax.set_ylabel('MI')
    ax.set_yscale('log')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=4, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
    plt.savefig(data_save_fN+'_MI_efficiency.png', bbox_extra_artists=(lgd,), bbox_inches='tight') 

#    ax.set_xlim(xlim)
    ax.set_xscale('log')
    plt.savefig(data_save_fN+'_MI_efficiency_log.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    pdb.set_trace()

def plot_MI_bound_from_file(data_save_fN):
    matplotlib.rcParams['axes.labelsize']=20
    matplotlib.rcParams['font.serif']='Times New Roman'
    matplotlib.rcParams['font.family']='sans-serif'
    
    dataDic=pickle.load(open(data_save_fN,'rb'))
    fig=plt.figure()
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
    ax.plot(xAxis,np.log(1+xAxis),'-r',label='Bound')
    
#    ax.plot(xAxis,xAxis,'-r',label='Bound')
    ax.set_xlim([xlim[0],15])
    ax.set_ylim([1e-6,0.5e1])
    ax.set_xlabel('CV(V) squared')
    ax.set_ylabel('I(X,V) scaled')
#    ax.set_xlabel('Fano(v)')
#    ax.set_ylabel('MI')
    ax.set_yscale('log')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=4, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
    plt.savefig(data_save_fN+'_MI_bound.png', bbox_extra_artists=(lgd,), bbox_inches='tight') 

    ax.set_xlim(xlim)
    ax.set_xscale('log')
    plt.savefig(data_save_fN+'_MI_bound_log.png', bbox_extra_artists=(lgd,), bbox_inches='tight') 

def plot_MI_bound(entries,keykey='production'):
    data_save_fN='plot_MI_bound_'+keykey+'.p'
    encoding_key_map={'production':'alpha_C_func','degradation':'mu_C_func'}
    mi_bound_func_dic={'production':lambda encoding_mean,encoding_var,cc_mean:encoding_mean*np.log(1+encoding_var/encoding_mean**2),
                       'degradation':lambda encoding_mean,encoding_var,cc_mean:encoding_mean*cc_mean*np.log(1+encoding_var/encoding_mean**2)}
    mi_scale_func_dic={'production':lambda mi,encoding_mean,cc_mean:mi/encoding_mean, 
                        'degradation':lambda mi,encoding_mean,cc_mean:mi/(encoding_mean*cc_mean)}
    nEn=len(entries)
    dataDic=dict([(entry[0],[]) for entry in entries])
    for j in range(nEn):
        label,fNs=entries[j]
        ll=len(fNs)
        mi_est=np.zeros(ll)
        mi_bound=np.zeros(ll)
        xVar=np.zeros(ll)
        xMean=np.zeros(ll)
        cMean=np.zeros(ll)
        mi_est_scaled=np.zeros(ll)
        mi_bound_scaled=np.zeros(ll)
        encodingMean=np.zeros(ll)
        encodingVar=np.zeros(ll)
        encodingPropensity=np.zeros(ll)
        for fN,i in zip(fNs,range(ll)):
            st=time.time()
            try:
                rate_dict=pickle.load(open(dataFolder+fN+'_rate.p','rb'))
            except UnicodeDecodeError:
                rate_dict=pickle.load(open(dataFolder+fN+'_rate.p','rb'),encoding='latin1') #encoding argument to load python 2 pickle object
            rate=rate_dict['rate']
            
            aa=pd.DataFrame.from_csv(dataFolder+fN)
            t_cutoff=aa['t'].values[-1]/5.
            t_mask=aa['t']>t_cutoff
            tt=(aa['t'].loc[t_mask]).values
            var='MI'
            mi=(aa[var].loc[t_mask]).values
            mi=mi[:-1]
            t_weight=np.diff(tt)
            dur=tt[-1]-t_cutoff
            mi_est[i]=np.sum(mi*t_weight)/dur
            
            xx=(aa['X'].loc[t_mask]).values
            xx=xx[:-1]
            xx_mean=np.sum(xx*t_weight)/dur
            xx_var=np.sum((xx-xx_mean)**2*t_weight)/dur
            
            cc=(aa['C'].loc[t_mask]).values
            cc=cc[:-1]
            cc_mean=np.sum(cc*t_weight)/dur
            
            encoding=eval(rate[encoding_key_map[keykey]])(xx,rate,cc)
            encoding_mean=np.sum(encoding*t_weight)/dur
            encoding_var=np.sum((encoding-encoding_mean)**2*t_weight)/dur
            
            
            mi_bound[i]=mi_bound_func_dic[keykey](encoding_mean,encoding_var,cc_mean)
            
            mi_est_scaled[i]=mi_scale_func_dic[keykey](mi_est[i],encoding_mean,cc_mean)
            mi_bound_scaled[i]=mi_scale_func_dic[keykey](mi_bound[i],encoding_mean,cc_mean)
        
            xVar[i]=xx_var
            xMean[i]=xx_mean
            cMean[i]=cc_mean
            encodingMean[i]=encoding_mean
            encodingVar[i]=encoding_var
            if keykey=='production':
                encodingPropensity[i]=encodingMean[i]
            elif keykey=='degradation':
                encodingPropensity[i]=np.sum(cc*encoding*t_weight)/dur
            ed=time.time()
            print(ed-st)
#            if label=='hill' and alpha_c_var/alpha_c_mean**2>1e-2 and mi_est_scaled[i]>1e-1:
#                print(fN)
#            if mi_bound_scaled[i]<mi_est_scaled[i]:
#                print(fN)
    
        xCV=xVar/xMean**2
        encodingCV=encodingVar/encodingMean**2
        dataDic[label]={'xCV':xCV,'xMean':xMean,'xVar':xVar,'cMean':cMean,
               'mi_est':mi_est,'mi_est_scaled':mi_est_scaled,
               'mi_bound':mi_bound,'mi_bound_scaled':mi_bound_scaled,
               'encodingMean':encodingMean,'encodingVar':encodingVar,'encodingCV':encodingCV,
               'encodingPropensity':encodingPropensity}
    pickle.dump(dataDic,open(data_save_fN,'wb'))
    plot_MI_bound_from_file(data_save_fN)


def plot_var_bound_from_file(data_save_fN):
    dataDic=pickle.load(open(data_save_fN,'rb'))
    
    
    # first plot. Ratio of sim Var(X) / bound Var(X).
    # The keys are (encoding_keykey,control_keykey)
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    colors=dict([(key,color) for key,color in zip(dataDic.keys(),['b','g','r','c','m','gold','k','cyan','gray'])])
    legend_map={'linear':'l','hill':'h','switching':'s'}
    for key,val in dataDic.items():
        ax.plot(val['mi_est'],val['xVar']/val['xVar_bound_exact'],'o',color=colors[key],label=''.join([legend_map[i] for i in key]))
    
    xlim=[1e-3,1e3]
    xAxis=np.logspace(np.log(xlim[0]),np.log(xlim[1]),1000)
    ax.plot(xAxis,np.ones(1000),'-r',label='Bound')
    
    
    ax.set_xlabel('MI(sim)')
    ax.set_ylabel('Sim Var(X)/ Bound')
    ax.set_yscale('log')
    ax.set_xscale('log')
    handles, labels = ax.get_legend_handles_labels()
#    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1., 0.102), borderaxespad=0.0, mode='expand', ncol=5)
    plt.savefig(data_save_fN+'_var_bound_unbounded.png', bbox_extra_artists=(lgd,), bbox_inches='tight') 
    
    ax.set_xlim(xlim)
    ax.set_ylim([1e-1,1e2])
    plt.savefig(data_save_fN+'_var_bound.png', bbox_extra_artists=(lgd,), bbox_inches='tight') 


    # second plot, the y-axis is fano factor, and x-axis is a relevant variable different for different cases.
    fig=plt.figure()
    ax=fig.add_subplot(111)
    colors=dict([(key,color) for key,color in zip(dataDic.keys(),['b','g','r','c','m','gold','k','cyan','gray'])])
    legend_map={'linear':'l','hill':'h','switching':'s'}
    for key,val in dataDic.items():
        ax.plot(val['xVar_bound_approx_factor'],val['xFano'],'o',color=colors[key],label=''.join([legend_map[i] for i in key]))
    
    xlim=[1e-3,1e3]
    xAxis=np.logspace(np.log(xlim[0]),np.log(xlim[1]),1000)
    ax.plot(xAxis,1/(1+xAxis),'-r',label='Bound')
    
    
    ax.set_xlabel('Signaling factor')
    ax.set_ylabel('Fano X')
    ax.set_yscale('log')
    ax.set_xscale('log')
    handles, labels = ax.get_legend_handles_labels()
#    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1., 0.102), borderaxespad=0.0, mode='expand', ncol=5)
    plt.savefig(data_save_fN+'_fano_bound_unbounded.png', bbox_extra_artists=(lgd,), bbox_inches='tight') 
    
    ax.set_xlim(xlim)
    ax.set_ylim([1e-3,1e2])
    plt.savefig(data_save_fN+'_fano_bound.png', bbox_extra_artists=(lgd,), bbox_inches='tight') 


    fig=plt.figure()
    ax=fig.add_subplot(111)
    colors=dict([(key,color) for key,color in zip(dataDic.keys(),['b','g','r','c','m','gold','k','cyan','gray'])])
    legend_map={'linear':'l','hill':'h','switching':'s'}
    for key,val in dataDic.items():
        ax.plot(val['xVar_bound_approx_factor'],val['xFano_alternative'],'o',color=colors[key],label=''.join([legend_map[i] for i in key]))
    
    xlim=[1e-3,1e3]
    xAxis=np.logspace(np.log(xlim[0]),np.log(xlim[1]),1000)
    ax.plot(xAxis,1/(1+xAxis),'-r',label='Bound')
    
    
    ax.set_xlabel('Signaling factor')
    ax.set_ylabel('Fano X')
    ax.set_yscale('log')
    ax.set_xscale('log')
    handles, labels = ax.get_legend_handles_labels()
#    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,0.,1.,1),borderaxespad=0.0)
    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1., 0.102), borderaxespad=0.0, mode='expand', ncol=5)
    plt.savefig(data_save_fN+'_fano_alternative_bound_unbounded.png', bbox_extra_artists=(lgd,), bbox_inches='tight') 
    
    ax.set_xlim(xlim)
    ax.set_ylim([1e-3,1e2])
    plt.savefig(data_save_fN+'_fano_alternative_bound.png', bbox_extra_artists=(lgd,), bbox_inches='tight') 


def plot_var_bound(entries,control_keykey='production',encoding_keykey='production'):
    data_save_fN='plot_var_bound_CONTROL_'+control_keykey+'_ENCODING_'+encoding_keykey+'.p'
    encoding_key_map={'production':'alpha_C_func','degradation':'mu_C_func'}
    control_key_map={'production':'alpha_X_func','degradation':'mu_X_func'}
    control_const_key_map={'production':'mu_X','degradation':'alpha_X'}
    mi_bound_func_dic={'production':lambda encoding_mean,encoding_var,cc_mean:encoding_mean*np.log(1+encoding_var/encoding_mean**2),
                       'degradation':lambda encoding_mean,encoding_var,cc_mean:encoding_mean*cc_mean*np.log(1+encoding_var/encoding_mean**2)}
    mi_scale_func_dic={'production':lambda mi,encoding_mean,cc_mean:mi/encoding_mean, 
                        'degradation':lambda mi,encoding_mean,cc_mean:mi/(encoding_mean*cc_mean)}
    var_bound_func_exact={
            ('production','production'):lambda data: (data['controlMean']) / (data['mi_bound']+data['controlConst']),
            ('production','degradation'):lambda data: (data['controlConst']) / (data['mi_bound']+data['controlMean']) ,
            ('degradation','production'):lambda data: (data['controlMean']) / (data['mi_bound']+data['controlConst']),
            ('degradation','degradation'):lambda data: (data['controlConst']) / (data['mi_bound']+data['controlMean'])
            }
    
    # This compares with Fano factor of X.
    var_bound_func_approx={
            ('production','production'):lambda data: data['encodingVar']/data['encodingMean']/data['controlConst'],
            ('production','degradation'):lambda data:data['encodingVar']/data['encodingMean']/data['controlMean'],
            ('degradation','production'):lambda data: data['cMean']*data['encodingVar']/data['encodingMean']/data['controlConst'],
            ('degradation','degradation'):lambda data: data['cMean']*data['encodingVar']/data['encodingMean']/data['controlMean'],
            }
    fano_func_alternative={
            ('production','production'):lambda data: data['xVar']*data['controlConst']/data['controlMean'],
            ('production','degradation'):lambda data:data['xVar']*data['controlMean']/data['controlConst'],
            ('degradation','production'):lambda data: data['xVar']*data['controlConst']/data['controlMean'],
            ('degradation','degradation'):lambda data: data['xVar']*data['controlMean']/data['controlConst'],
            }
    nEn=len(entries)
    label_to_key=lambda label:(label['encoding_mode'],label['control_mode'])
    dataDic=dict([(label_to_key(entry[0]),[]) for entry in entries])
    for j in range(nEn):
        label,fNs=entries[j]
        label_key=label_to_key(label)
        ll=len(fNs)
        mi_est=np.zeros(ll)
        mi_bound=np.zeros(ll)
        xVar=np.zeros(ll)
        xMean=np.zeros(ll)
        cMean=np.zeros(ll)
        mi_est_scaled=np.zeros(ll)
        mi_bound_scaled=np.zeros(ll)
        encodingMean=np.zeros(ll)
        encodingVar=np.zeros(ll)
        controlConst=np.zeros(ll)
        controlMean=np.zeros(ll)
        controlVar=np.zeros(ll)
        for fN,i in zip(fNs,range(ll)):
            st=time.time()
            try:
                rate_dict=pickle.load(open(dataFolder+fN+'_rate.p','rb'))
            except UnicodeDecodeError:
                rate_dict=pickle.load(open(dataFolder+fN+'_rate.p','rb'),encoding='latin1') #encoding argument to load python 2 pickle object
            rate=rate_dict['rate']
            
            aa=pd.DataFrame.from_csv(dataFolder+fN)
            t_cutoff=aa['t'].values[-1]/5.
            t_mask=aa['t']>t_cutoff
            tt=(aa['t'].loc[t_mask]).values
            var='MI'
            mi=(aa[var].loc[t_mask]).values
            mi=mi[:-1]
            t_weight=np.diff(tt)
            dur=tt[-1]-t_cutoff
            mi_est[i]=np.sum(mi*t_weight)/dur
            
            xx=(aa['X'].loc[t_mask]).values
            xx=xx[:-1]
            xx_mean=np.sum(xx*t_weight)/dur
            xx_var=np.sum((xx-xx_mean)**2*t_weight)/dur
            
            cc=(aa['C'].loc[t_mask]).values
            cc=cc[:-1]
            cc_mean=np.sum(cc*t_weight)/dur
            
            xx_est=(aa['X_est'].loc[t_mask]).values
            xx_est=xx_est[:-1]
            
            encoding=eval(rate[encoding_key_map[encoding_keykey]])(xx,rate,cc)
            encoding_mean=np.sum(encoding*t_weight)/dur
            encoding_var=np.sum((encoding-encoding_mean)**2*t_weight)/dur
            
            control=eval(rate[control_key_map[control_keykey]])(xx_est,rate,xx)
            control_mean=np.sum(control*t_weight)/dur
            control_var=np.sum((control-control_mean)**2*t_weight)/dur
            
            mi_bound[i]=mi_bound_func_dic[encoding_keykey](encoding_mean,encoding_var,cc_mean)
            
            mi_est_scaled[i]=mi_scale_func_dic[encoding_keykey](mi_est[i],encoding_mean,cc_mean)
            mi_bound_scaled[i]=mi_scale_func_dic[encoding_keykey](mi_bound[i],encoding_mean,cc_mean)
        
            xVar[i]=xx_var
            xMean[i]=xx_mean
            cMean[i]=cc_mean
            encodingMean[i]=encoding_mean
            encodingVar[i]=encoding_var
            controlConst[i]=rate[control_const_key_map[control_keykey]]
            controlMean[i]=control_mean
            controlVar[i]=control_var
            ed=time.time()
            print(ed-st)
#            if label=='hill' and alpha_c_var/alpha_c_mean**2>1e-2 and mi_est_scaled[i]>1e-1:
#                print(fN)
#            if mi_bound_scaled[i]<mi_est_scaled[i]:
#                print(fN)
    
        xCV=xVar/xMean**2
        encodingCV=encodingVar/encodingMean**2
        dataDic[label_key]={'xCV':xCV,'xMean':xMean,'xVar':xVar,'cMean':cMean,'xFano':xVar/xMean,
               'mi_est':mi_est,'mi_est_scaled':mi_est_scaled,
               'mi_bound':mi_bound,'mi_bound_scaled':mi_bound_scaled,
               'encodingMean':encodingMean,'encodingVar':encodingVar,'encodingCV':encodingCV,
               'controlMean':controlMean,'controlVar':controlVar,'controlConst':controlConst,
               'encodingKey':encoding_keykey,'controlKey':control_keykey}
        dataDic[label_key]['xVar_bound_exact']=var_bound_func_exact[(encoding_keykey,control_keykey)](dataDic[label_key])
        dataDic[label_key]['xVar_bound_approx_factor']=var_bound_func_approx[(encoding_keykey,control_keykey)](dataDic[label_key])
        dataDic[label_key]['xFano_alternative']=fano_func_alternative[(encoding_keykey,control_keykey)](dataDic[label_key])
        

    pickle.dump(dataDic,open(data_save_fN,'wb'))
#    plot_var_bound_from_file(data_save_fN)



def sim_par_func(arg):
    fN=arg['fN']
    T=arg['T']
    init_state,rate=init_simple_default(arg['param'])
    keykey=arg['param']['keykey']
    rate_dict={'init_state':init_state,'rate':rate,'T':T,'keykey':keykey}
    pickle.dump(rate_dict,open(dataFolder+fN+'_rate.p','wb'))
    for key in ['alpha_C_func','alpha_X_func','mu_X_func','mu_C_func']:
        rate[key]=eval(rate[key])
    g_sim=gillespie(init_state,rate,al_simple,particle_gen,keykey)
    st=time.time()
    sim(g_sim,fN,T,dataFolder)
    ed=time.time()
    print(ed-st)

def sim_par(fNs,param_list,nCore,T,dataFolder):
    # param_list each element is a dictionary ((key,value),(key,value),...) corresponding to entries to be changed in rate dict.
    args=[{'dataFolder':dataFolder,'T':T,'param':x,'fN':fN} for x,fN in zip(param_list,fNs)]
    if nCore==1:
        for arg in args:
            try:
                sim_par_func(arg)
            except KeyError:
                pdb.set_trace()
    else:
        pool=Pool(nCore)
        pool.map(sim_par_func,args)

def alpha_C_func_switching(xx,rate,c=0):
    # DON'T CHANGE THESE FUNCTIONS AS THEY WILL BE USED IN DATA PROCESSING LATER!!!!!!
    try:
        return (xx<rate['x_threshold'])*(rate['alpha_C']-rate['alpha_C_lower'])+rate['alpha_C_lower']+1e-23
    except TypeError:
        return rate['alpha_C'] if xx<rate['x_threshold'] else rate['alpha_C_lower']

def alpha_C_func_hill(xx,rate,c=0):
    # DON'T CHANGE THESE FUNCTIONS AS THEY WILL BE USED IN DATA PROCESSING LATER!!!!!!
    return (rate['alpha_C']-rate['alpha_C_lower'])/((rate['x_threshold']/xx)**rate['hill_coeff']+1) + rate['alpha_C_lower']+1e-14

def alpha_C_func_linear(xx,rate,c=0):
    # DON'T CHANGE THESE FUNCTIONS AS THEY WILL BE USED IN DATA PROCESSING LATER!!!!!!
    return rate['alpha_C']*xx+1e-14

def mu_C_func_switching(xx,rate,c):
    # DON'T CHANGE THESE FUNCTIONS AS THEY WILL BE USED IN DATA PROCESSING LATER!!!!!!
    try:
        return ((xx<rate['x_threshold'])*(rate['mu_C']-rate['mu_C_lower'])+rate['mu_C_lower'])*(c>=1)+1e-14
    except TypeError:
        return rate['mu_C']*(c>=1)+1e-14 if xx<rate['x_threshold'] else rate['mu_C_lower']*(c>=1)+1e-14

def mu_C_func_hill(xx,rate,c):
    # DON'T CHANGE THESE FUNCTIONS AS THEY WILL BE USED IN DATA PROCESSING LATER!!!!!!
    return ((rate['mu_C']-rate['mu_C_lower'])/((rate['x_threshold']/xx)**rate['hill_coeff']+1) + rate['mu_C_lower'])*(c>=1)+1e-14

def mu_C_func_linear(xx,rate,c):
    return rate['mu_C']*xx*(c>=1)+1e-14

def alpha_X_func_switching(xx_est,rate,x=0):
    # DON'T CHANGE THESE FUNCTIONS AS THEY WILL BE USED IN DATA PROCESSING LATER!!!!!!
    try:
        return (xx_est<rate['x_est_threshold'])*(rate['alpha_X']-rate['alpha_X_lower'])+rate['alpha_X_lower']+1e-14
    except TypeError:
        return rate['alpha_X'] if xx_est<rate['x_est_threshold'] else rate['alpha_X_lower']

def alpha_X_func_hill(xx_est,rate,x=0):
    # DON'T CHANGE THESE FUNCTIONS AS THEY WILL BE USED IN DATA PROCESSING LATER!!!!!!
    return (rate['alpha_X']-rate['alpha_X_lower'])/((xx_est/rate['x_est_threshold'])**rate['hill_coeff_control']+1) + rate['alpha_X_lower']+1e-14

def alpha_X_func_linear(xx_est,rate,x=0):
    # DON'T CHANGE THESE FUNCTIONS AS THEY WILL BE USED IN DATA PROCESSING LATER!!!!!!
    return np.maximum(rate['alpha_X']*(rate['x_est_threshold']-xx_est)+rate['x_d']*rate['mu_X'],0)+1e-14


def mu_X_func_switching(xx_est,rate,x=0):
    # DON'T CHANGE THESE FUNCTIONS AS THEY WILL BE USED IN DATA PROCESSING LATER!!!!!!
    try:
        return ((xx_est<rate['x_est_threshold'])*(rate['mu_X']-rate['mu_X_lower'])+rate['mu_X_lower'])*(x>=1)+1e-14
    except TypeError:
        return rate['mu_X']*(x>=1)+1e-14 if xx_est<rate['x_est_threshold'] else rate['mu_X_lower']*(x>=1)+1e-14

def mu_X_func_hill(xx_est,rate,x=0):
    # DON'T CHANGE THESE FUNCTIONS AS THEY WILL BE USED IN DATA PROCESSING LATER!!!!!!
    return ((rate['mu_X']-rate['mu_X_lower'])/((rate['x_est_threshold']/xx_est)**rate['hill_coeff_control']+1) + rate['mu_X_lower'])*(x>=1)+1e-14

def mu_X_func_linear(xx_est,rate,x=0):
    # DON'T CHANGE THESE FUNCTIONS AS THEY WILL BE USED IN DATA PROCESSING LATER!!!!!!
    return np.maximum(rate['mu_X']*(xx_est-rate['x_est_threshold'])+rate['alpha_X']/rate['x_d'],0)+1e-14


def re_sim(T=1500,fN='channel_capacity_switching_22',keykey='production',replace=False,dataFolder='./data/'):
    rate=pickle.load(open(dataFolder+fN+'_rate.p','rb'),encoding='latin1')['rate']
    
    if not replace:
        fN='channel_capacity_resim'
    
#    rate['mu_X_func']="lambda x_est,rate,x:rate['mu_X']*(x>=1)+1e-23"
#    rate['alpha_C_lower']=1e-23
#    rate['alpha_X_func']="lambda x_est,rate:rate['alpha_X']+1e-23"
    
    state={'X':round(rate['alpha_X']),'C':0,'X_est':round(rate['alpha_X']),'MI':0}
    param_list=[{'rate':rate,'state':state,'keykey':keykey}]
    state_init,rate=init_simple_default(param=param_list[0])
    rate_dict={'init_state':state_init,'rate':rate,'T':T}
    pickle.dump(rate_dict,open(dataFolder+fN+'_rate.p','wb'))
    for key in ['alpha_C_func','alpha_X_func','mu_X_func','mu_C_func']:
        rate[key]=eval(rate[key])
    g_sim=gillespie(state_init,rate,al_simple,particle_gen,param_list[0]['keykey'])

    sim(g_sim,fN,T,dataFolder)
    plot_time_traj(fN)
    plot_MI(fN)

def rate_state_gen(NN,control_keykey,control_mode,encoding_keykey,encoding_mode,control_func_map,encoding_func_map,param_range_map):
    def rate_map(rate,encoding_key,encoding_mode,control_key,control_mode):
        if encoding_key=='degradation':
            if encoding_mode=='linear':
                rate['alpha_C']=min(rate['mu_C']*rate['c_count']/2,rate['mu_C']*np.sqrt(rate['c_count'])*3)*rate['x_d']
            else:
                rate['alpha_C']=rate['mu_C']*rate['c_count']
        #we also need to process mu_X_lower, alpha_X_lower, and those for C
        # for C, we just need them to be less than alpha_C, mu_C
        # for X, we need to make them less than alpha_X/x_d, or less than mu_X*x_d.
        # for X, for hill functions, we can determine lower directly.
        if encoding_key=='production' and (encoding_mode=='hill' or encoding_mode=='switching'):
            rate['alpha_C_lower']=max(rate['alpha_C_lower']-0.5,0)*rate['alpha_C']
        if encoding_key=='degradation' and (encoding_mode=='hill' or encoding_mode=='switching'):
            rate['mu_C_lower']=max(rate['mu_C_lower']-0.5,0)*rate['mu_C']
        if control_key=='production':
            if control_mode=='switching':
                rate['alpha_X_lower']=max(rate['alpha_X_lower']-0.5,0)*rate['mu_X']*rate['x_d']
            elif control_mode=='hill':
                rate['alpha_X_lower']=max(2*rate['mu_X']*rate['x_d']-rate['alpha_X'],0)
        if control_key=='degradation':
            if control_mode=='switching':
                rate['mu_X_lower']=max(rate['mu_X_lower']-0.5,0)*rate['alpha_X']/rate['x_d']
            elif control_mode=='hill':
                rate['mu_X_lower']=max(2*rate['alpha_X']/rate['x_d']-rate['mu_X'],0)
        return rate
    
    #state map only cares about encoding, so only encoding key is input.
    
    rate_mother={}
    keykey_to_func_name={'production':'alpha','degradation':'mu'}
    for branch in ['control','encoding']:
        for key,val in param_range_map[branch][(eval(branch+'_keykey'),eval(branch+'_mode'))].items():
            if val[1]=='log':
                temp=np.random.uniform(np.log10(val[0][0]),np.log10(val[0][1]),size=NN)
                rate_mother[key]=10**temp
            elif val[1]=='linear':
                rate_mother[key]=np.random.uniform((val[0][0]),(val[0][1]),size=NN)
    rates=[]
    states=[]
    keys=list(rate_mother.keys())
    vals=list(rate_mother.values())
    
    for i in np.arange(NN):
        rate={}
        rate[keykey_to_func_name[control_keykey]+'_X_func']=control_func_map[control_keykey][control_mode]
        rate[keykey_to_func_name[encoding_keykey]+'_C_func']=encoding_func_map[encoding_keykey][encoding_mode]
        for key,val in zip(keys,vals):
            rate[key]=val[i]
        try:
            state={'X':round(rate['x_d']),'C':round(rate['c_count']),'X_est':rate['x_d']}
        except KeyError:
            state={'X':round(rate['x_d']),'C':0,'X_est':rate['x_d']}
        state_init,rate_init=init_simple_default({'rate':rate,'state':state})
        rate_init=rate_map(rate_init,encoding_keykey,encoding_mode,control_keykey,control_mode)
        states=states+[state_init]
        rates=rates+[rate_init]
    
    return rates,states

if __name__=='__main__':
    dataFolder='./data/'
    controls={'production':['linear','hill','switching'],
              'degradation':['linear','hill','switching']}
    encodings={'production':['linear','hill','switching'],
              'degradation':['linear','hill','switching']}
    control_prod_func_map=dict([(mode,'alpha_X_func_'+mode) for mode in controls['production']])
    control_deg_func_map=dict([(mode,'mu_X_func_'+mode) for mode in controls['degradation']])
    encoding_prod_func_map=dict([(mode,'alpha_C_func_'+mode) for mode in encodings['production']])
    encoding_deg_func_map=dict([(mode,'mu_C_func_'+mode) for mode in encodings['degradation']])
    control_func_map={'production':control_prod_func_map,'degradation':control_deg_func_map}
    encoding_func_map={'production':encoding_prod_func_map,'degradation':encoding_deg_func_map}

#    plot_MI_efficiency_from_file('plot_MI_bound_production.p')
#    plot_MI_efficiency_from_file('plot_MI_bound_degradation.p')
#    data_save_fN='plot_var_bound_CONTROL_'+'degradation'+'_ENCODING_'+'degradation'+'.p'
#    data_save_fN='plot_var_bound_CONTROL_'+'production'+'_ENCODING_'+'degradation'+'.p'
#    plot_var_bound_from_file(data_save_fN)
#    plot_MI_bound_from_file('plot_MI_bound_degradation.p')
#    plot_MI_bound_from_file('plot_MI_bound_production.p')
#    fN='channel_capacity_CONTROL_production_hill_ENCODING_degradation_hill_2'
    fN='channel_capacity_switching_3_14' # switching_3_#: 14, 69, 81.
    T=1000
#    re_sim(T=T,fN=fN,keykey='production',replace=True)
#    plot_time_traj(fN)
#    plot_MI(fN)

####################################################################
#####################################################################
#####################     CONTROL    #################################
#####################################################################
####################################################################



############### PLOTTING ######################
#    mega_entries={}
#    NN=10
#    for control_keykey in controls.keys():
#        for encoding_keykey in encodings.keys():
#            labels=[]
#            fN_lists=[]
#            for control_mode in controls[control_keykey]:
#                for encoding_mode in encodings[encoding_keykey]:
#                    fN_stub='channel_capacity_CONTROL_'+control_keykey+'_'+control_mode+'_ENCODING_'+encoding_keykey+'_'+encoding_mode+'_'
#                    fN_lists=fN_lists+[[fN_stub+str(i) for i in np.arange(NN)]]
#                    labels=labels+[{'control_mode':control_mode,'encoding_mode':encoding_mode}]
#            entries=list(zip(labels,fN_lists))
#            mega_entries[(encoding_keykey,control_keykey)]=entries
##
#    for key,val in mega_entries.items():
##        plot_var_bound(val,encoding_keykey=key[0],control_keykey=key[1])
#        control_keykey=key[1]
#        encoding_keykey=key[0]
#        data_save_fN='plot_var_bound_CONTROL_'+control_keykey+'_ENCODING_'+encoding_keykey+'.p'
#    
#        plot_var_bound_from_file(data_save_fN)





############### ALL TRAJ ##########################


#    T=500
#    t_cutoff=T*1/5.
#    nCore=30
##    
#    x_d=20
#    param_range_map={'control':{
#                        ('production','linear'):{
#                            'x_d':([x_d,x_d],'linear'),
#                            'x_est_threshold':([x_d*0.7,x_d*1.2],'linear'),
#                            'alpha_X':([1,1e2],'log')},
#                        ('production','switching'):{
#                            'x_d':([x_d,x_d],'linear'),
#                            'x_est_threshold':([x_d*0.7,x_d*1.2],'linear'),
#                            'alpha_X':([1,1e2],'log'),
#                            'alpha_X_lower':([0,1],'linear')},
#                        ('production','hill'):{
#                            'x_d':([x_d,x_d],'linear'),
#                            'x_est_threshold':([x_d*0.7,x_d*1.2],'linear'),
#                            'alpha_X':([1,1e2],'log'),
#                            'alpha_X_lower':([0,1],'linear'),
#                            'hill_coeff_control':([1e-1,5],'log')},
#                        ('degradation','linear'):{
#                            'x_d':([x_d,x_d],'linear'),
#                            'x_est_threshold':([x_d*0.7,x_d*1.2],'linear'),
#                            'mu_X':([1,2e1],'log'),
#                            'alpha_X':([x_d,x_d],'linear')},
#                        ('degradation','switching'):{
#                            'x_d':([x_d,x_d],'linear'),
#                            'x_est_threshold':([x_d*0.7,x_d*1.2],'linear'),
#                            'mu_X':([1,1e1],'log'),
#                            'mu_X_lower':([0,1],'linear'),
#                            'alpha_X':([x_d,x_d],'linear')},
#                        ('degradation','hill'):{
#                            'x_d':([x_d,x_d],'linear'),
#                            'x_est_threshold':([x_d*0.7,x_d*1.2],'linear'),
#                            'mu_X':([1,1e1],'log'),
#                            'mu_X_lower':([0,1],'linear'),
#                            'hill_coeff_control':([1e-1,5],'log'),
#                            'alpha_X':([x_d,x_d],'linear')}
#                        },
#                    'encoding':{
#                        ('production','linear'):{
#                            'alpha_C':([1e-1,5e1],'log'),
#                            'mu_C':([0,0],'linear')},
#                        ('production','switching'):{
#                            'x_threshold':([x_d*0.8,x_d*1.2],'linear'),
#                            'alpha_C':([1e-1,5e1],'log'),
#                            'alpha_C_lower':([0,1],'linear'),
#                            'mu_C':([0,0],'linear')},
#                        ('production','hill'):{
#                            'x_threshold':([x_d*0.8,x_d*1.2],'linear'),
#                            'alpha_C':([1e-1,5e1],'log'),
#                            'alpha_C_lower':([0,1],'linear'),
#                            'hill_coeff':([1e-1,5],'log'),
#                            'mu_C':([0,0],'linear')},
#                        ('degradation','linear'):{
#                            'c_count':([10,100],'log'),
#                            'mu_C':([1e-1,1e1],'log')},
#                        ('degradation','switching'):{
#                            'c_count':([10,100],'log'),
#                            'x_threshold':([x_d*0.8,x_d*1.2],'linear'),
#                            'mu_C':([1e-1,2e2],'log'),
#                            'mu_C_lower':([0,1],'linear')},
#                        ('degradation','hill'):{
#                            'c_count':([10,100],'log'),
#                            'x_threshold':([x_d*0.8,x_d*1.2],'linear'),
#                            'mu_C':([1e-1,2e2],'log'),
#                            'mu_C_lower':([0,1],'linear'),
#                            'hill_coeff':([1e-1,5],'log')}
#                        }
#                 }
#    
#    param_list=[]
#    fNs=[]
#    NN=10
#    for control_keykey in controls.keys():
#        for encoding_keykey in encodings.keys():
#            for control_mode in controls[control_keykey]:
#                for encoding_mode in encodings[encoding_keykey]:
#                    fN_stub='channel_capacity_CONTROL_'+control_keykey+'_'+control_mode+'_ENCODING_'+encoding_keykey+'_'+encoding_mode+'_'
#                    rates,states=rate_state_gen(NN,control_keykey,control_mode,encoding_keykey,encoding_mode,control_func_map,encoding_func_map,param_range_map)
#                    param_list=param_list+[{'rate':rate,'state':state,'keykey':encoding_keykey} for rate,state in zip(rates,states)]
#                    fNs=fNs+[fN_stub+str(i) for i in np.arange(NN)]
#                    print([control_keykey,control_mode,encoding_keykey,encoding_mode])
#    pdb.set_trace()
#    sim_par(fNs,param_list,nCore,T,dataFolder)




####################### ONE TRAJ ###################################################
#    T=10
#    t_cutoff=T*1/5.
#    nCore=1
#    
#    control_keykey='production'
#    control_mode='linear'
#    encoding_keykey='production'
#    encoding_mode='linear'
#    fN='channel_capacity_CONTROL_'+control_keykey+'_'+control_mode+'_ENCODING_'+encoding_keykey+'_'+encoding_mode+'_'
#    
#    alpha_c=10
#    alpha_x=10
#    x_d=10
#    
#    
#    rate={'alpha_C':alpha_c,
#          'alpha_C_func':encoding_func_map[encoding_keykey][encoding_mode],
#          'alpha_X':alpha_x,
#          'alpha_X_func':control_func_map[control_keykey][control_mode],
#          'mu_X':1,
#          'mu_C':0,
#          'x_d':x_d,
#          'x_est_threshold':x_d
#          }
#    state={'X':round(alpha_x),'C':0,'X_est':alpha_x,'MI':0}
#    param_list=[{'rate':rate,'state':state}]
#    state_init,rate=init_simple_default(param=param_list[0])
#    keykey='production'
#    rate_dict={'init_state':state_init,'rate':rate,'T':T,'keykey':keykey}
#    pickle.dump(rate_dict,open(dataFolder+fN+'_rate.p','wb'))
#    for key in ['alpha_C_func','alpha_X_func','mu_X_func','mu_C_func']:
#        rate[key]=eval(rate[key])
#    
#    g_sim=gillespie(state_init,rate,al_simple,particle_gen,keykey=keykey)
#    
#    sim(g_sim,fN,T,dataFolder)
#    plot_time_traj(fN)
#    plot_MI(fN)
#    pdb.set_trace()
  
####################################################################
#####################################################################
#####################    DEGRADATION   #################################
#####################################################################
####################################################################

############## PLOT BOUND #########################
#    NN=50
#    fN_stubs_switching=['channel_capacity_DEGRADATION_switching_']
#    fNs_switching=[[fN_stub+str(i) for i in range(NN)] for fN_stub in fN_stubs_switching]
#    temp=[]
#    for xx in fNs_switching:
#        temp+=xx
#    fNs_switching=temp
##        
#    fN_stubs_linear=['channel_capacity_DEGRADATION_linear_']
#    fNs_linear=[[fN_stub+str(i) for i in range(NN)] for fN_stub in fN_stubs_linear]
##    fNs_linear=[[fN_stub+str(i) for i in [0,1,3]] for fN_stub in fN_stubs_linear]
#    temp2=[]
#    for xx in fNs_linear:
#        temp2+=xx
#    fNs_linear=temp2
##    
#    fN_stubs_hill=['channel_capacity_DEGRADATION_hill_']
#    fNs_hill=[[fN_stub+str(i) for i in range(NN)] for fN_stub in fN_stubs_hill]
#    temp3=[]
#    for xx in fNs_hill:
#        temp3+=xx
#    fNs_hill=temp3
##    
##    
#    labels=['switching','linear','hill']
#    fN_lists=[fNs_switching,fNs_linear,fNs_hill]
##    labels=['linear']
##    fN_lists=[fNs_linear]
##    
##    
#    entries=list(zip(labels,fN_lists))
#    plot_MI_bound(entries,keykey='degradation')

############### ONE TRAJ ##########################
#    T=20
#    t_cutoff=T*1/5.
#    nCore=1
#    mu_c=1
#    alpha_x=10
#    rate={'alpha_C':mu_c*10*alpha_x,
#          'alpha_C_func':"lambda x,rate:rate['alpha_C']",
#          'alpha_X':alpha_x,
#          'mu_X':1,
#          'mu_C':mu_c,
#          'mu_C_func':"lambda x,rate,c:rate['mu_C']*x*(c>=1)"
#          }
#    state={'X':round(alpha_x),'C':10,'X_est':alpha_x,'MI':0}
#    param_list=[{'rate':rate,'state':state}]
#    state_init,rate=init_simple_default(param=param_list[0])
#    for key in ['alpha_C_func','alpha_X_func','mu_X_func','mu_C_func']:
#        rate[key]=eval(rate[key])
#    g_sim=gillespie(state_init,rate,al_simple,particle_gen,'degradation')
#    
#    
#    fN='channel_capacity_DEGRADATION_linear_test'
#    sim(g_sim,fN,T,dataFolder)
#    plot_time_traj(fN)
#    plot_MI(fN)
#    pdb.set_trace()


############### MULTIPLE TRAJ Linear #####################
#    T=500
#    t_cutoff=T*1/5.
#    nCore=25
#    NN=50
#    rates=[]
#    states=[]
#    alpha_xs=10**(2.2*np.random.rand(NN)-0.2)
#    mu_cs=10**(2.2*np.random.rand(NN)-0.2)
#    c_counts=10**(2.2*np.random.rand(NN)-0.2)
#    for alpha_x,mu_c,c_count in zip(alpha_xs,mu_cs,c_counts):
#        rates=rates+[{'alpha_C':mu_c*c_count*alpha_x,
#                      'mu_X':1,
#                      'alpha_X':alpha_x,
#                      'mu_C':mu_c,
#                      'alpha_C_func':"lambda x,rate:rate['alpha_C']",
#                      'mu_C_func':"mu_C_func_linear"
#                      }]
#        states=states+[{'X':round(alpha_x),'C':round(c_count),'X_est':alpha_x,'MI':0}]
#    param_list=[{'rate':rate,'state':state,'keykey':'degradation'} for rate,state in zip(rates,states)]
#    fN_stub='channel_capacity_DEGRADATION_linear_'
#    fNs=[fN_stub+str(i) for i in range(NN)]
#    sim_par(fNs,param_list,nCore,T,dataFolder)
#
############### MULTIPLE TRAJ hill ############################
#    T=500
#    t_cutoff=T*1/5.
#    nCore=25
#    NN=50
#    rates=[]
#    states=[]
#    alpha_xs=10**(2.2*np.random.rand(NN)-0.2)
#    mu_cs=10**(2.2*np.random.rand(NN)-0.2)
#    c_counts=10**(2.2*np.random.rand(NN)-0.2)
#    x_ds=np.random.rand(NN)
#    mu_c_lowers=np.random.rand(NN)
#    hill_coeffs=np.random.rand(NN)*7
#    for alpha_x,mu_c,x_d,mu_c_lower,hill_coeff,c_count in zip(alpha_xs,mu_cs,x_ds,mu_c_lowers,hill_coeffs,c_counts):
#        rates=rates+[{'alpha_C':mu_c*c_count*alpha_x,
#                      'mu_C_lower':max((mu_c_lower-0.3)*mu_c,0)+1e-7, #30 percent prob it will be 0.
#                      'mu_X':1,
#                      'alpha_X':alpha_x,
#                      'x_threshold':(x_d-0.5)*0.2*alpha_x+alpha_x,
#                      'hill_coeff':hill_coeff,
#                      'mu_C':mu_c,
#                      'alpha_C_func':"lambda x,rate:rate['alpha_C']",
#                      'mu_C_func':'mu_C_func_hill'
#                      }]
#        states=states+[{'X':round(alpha_x),'C':round(c_count),'X_est':alpha_x,'MI':0}]
#    param_list=[{'rate':rate,'state':state,'keykey':'degradation'} for rate,state in zip(rates,states)]
#    fN_stub='channel_capacity_DEGRADATION_hill_'
#    fNs=[fN_stub+str(i) for i in range(NN)]
#    sim_par(fNs,param_list,nCore,T,dataFolder)
#    
################ MULTIPLE TRAJ switching #####################
#
#    T=500
#    t_cutoff=T*1/5.
#    nCore=25
#    NN=50
#    rates=[]
#    states=[]
#    alpha_xs=10**(2.2*np.random.rand(NN)-0.2)
##    alpha_xs=10**(3*np.random.rand(NN)-1)
#    mu_cs=10**(2.2*np.random.rand(NN)-0.2)
#    x_ds=np.random.rand(NN)
#    c_counts=10**(2.2*np.random.rand(NN)-0.2)
#    mu_c_lowers=np.random.rand(NN)
#    for alpha_x,mu_c,x_d,mu_c_lower,c_count in zip(alpha_xs,mu_cs,x_ds,mu_c_lowers,c_counts):
#        rates=rates+[{'alpha_C':mu_c*c_count*alpha_x,
#                      'mu_C_lower':max((mu_c_lower-0.3)*mu_c,0)+1e-7, #30 percent prob it will be 0.
#                      'mu_X':1,
#                      'alpha_X':alpha_x,
#                      'x_threshold':(x_d-0.5)*0.2*alpha_x+alpha_x,
#                      'mu_C':mu_c,
#                      'alpha_C_func':"lambda x,rate:rate['alpha_C']",
#                      'mu_C_func':'mu_C_func_switching'
#                      }]
#        states=states+[{'X':round(alpha_x),'C':round(c_count),'X_est':alpha_x,'MI':0}]
#    param_list=[{'rate':rate,'state':state,'keykey':'degradation'} for rate,state in zip(rates,states)]
#    fN_stub='channel_capacity_DEGRADATION_switching_'
#    fNs=[fN_stub+str(i) for i in range(NN)]
#    sim_par(fNs,param_list,nCore,T,dataFolder)

####################################################################
#####################################################################
#####################    PRODUCTION   #################################
#####################################################################
####################################################################
    NN=100
    fN_stubs_switching=['channel_capacity_switching_3_']
    fNs_switching=[[fN_stub+str(i) for i in range(NN) if i not in [14,69,81]] for fN_stub in fN_stubs_switching]
    temp=[]
    for xx in fNs_switching:
        temp+=xx
    fNs_switching=temp
        
    fN_stubs_linear=['channel_capacity_linear_']
    fNs_linear=[[fN_stub+str(i) for i in range(NN)] for fN_stub in fN_stubs_linear]
    temp2=[]
    for xx in fNs_linear:
        temp2+=xx
    fNs_linear=temp2
    
    fN_stubs_hill=['channel_capacity_hill_']
    fNs_hill=[[fN_stub+str(i) for i in range(NN)] for fN_stub in fN_stubs_hill]
    temp3=[]
    for xx in fNs_hill:
        temp3+=xx
    fNs_hill=temp3
    
    
    labels=['switching','linear','hill']
    fN_lists=[fNs_switching,fNs_linear,fNs_hill]
#    labels=['switching']
#    fN_lists=[fNs_switching]
#    
    
    entries=list(zip(labels,fN_lists))
    plot_MI_bound(entries)


############## MULTIPLE TRAJ hill ############################
#    T=500
#    t_cutoff=T*1/5.
#    nCore=32
#    NN=100
#    rates=[]
#    states=[]
#    alpha_xs=10**(3*np.random.rand(NN)-1)
#    alpha_cs=10**(3*np.random.rand(NN)-1)
#    x_ds=np.random.rand(NN)
#    alpha_c_lowers=np.random.rand(NN)
#    hill_coeffs=np.random.rand(NN)*10
#    for alpha_x,alpha_c,x_d,alpha_c_lower,hill_coeff in zip(alpha_xs,alpha_cs,x_ds,alpha_c_lowers,hill_coeffs):
#        rates=rates+[{'alpha_C':alpha_c,
#                      'alpha_C_lower':max((alpha_c_lower-0.3)*alpha_c,0)+1e-7,
#                      'mu_X':1,
#                      'alpha_X':alpha_x,
#                      'x_threshold':(x_d-0.5)*0.2*alpha_x+alpha_x,
#                      'hill_coeff':hill_coeff,
#                      'mu_C':1,
#                      'alpha_C_func':"alpha_C_func_hill",
#                      'mu_C_func':"lambda x,rate,c:0"
#                      }]
#        states=states+[{'X':round(alpha_x),'C':0,'X_est':alpha_x,'MI':0}]
#    param_list=[{'rate':rate,'state':state,'keykey':'production'} for rate,state in zip(rates,states)]
#    fN_stub='channel_capacity_hill_'
#    fNs=[fN_stub+str(i) for i in range(NN)]
#    sim_par(fNs,param_list,nCore,T,dataFolder)
    
    
    
    
#    sim_par([fNs[5]],[param_list[5]],nCore,T,dataFolder)
#    plot_time_traj(fNs[-1])
#    plot_MI(fNs[-1])
#    plot_time_traj(fNs[5])
#    plot_MI(fNs[5])

############### MULTIPLE TRAJ switching #####################

#    T=500
#    t_cutoff=T*1/5.
#    nCore=32
#    NN=100
#    rates=[]
#    states=[]
#    alpha_xs=10**(3*np.random.rand(NN)-1)
#    alpha_cs=10**(3*np.random.rand(NN)-1)
#    x_ds=np.random.rand(NN)
#    alpha_c_lowers=np.random.rand(NN)
#    for alpha_x,alpha_c,x_d,alpha_c_lower in zip(alpha_xs,alpha_cs,x_ds,alpha_c_lowers):
#        rates=rates+[{'alpha_C':alpha_c,
#                      'alpha_C_lower':max((alpha_c_lower-0.5)*alpha_c,0)+1e-7,
#                      'mu_X':1,
#                      'alpha_X':alpha_x,
#                      'x_threshold':(x_d-0.5)*0.2*alpha_x+alpha_x,
#                      'mu_C':1,
#                      'alpha_C_func':"alpha_C_func_switching",
#                      'mu_C_func':"lambda x,rate,c:0"
#                      }]
#        states=states+[{'X':round(alpha_x),'C':0,'X_est':alpha_x,'MI':0}]
#    param_list=[{'rate':rate,'state':state,'keykey':'production'} for rate,state in zip(rates,states)]
#    fN_stub='channel_capacity_switching_'
#    fNs=[fN_stub+str(i) for i in range(NN)]
#    sim_par(fNs,param_list,nCore,T,dataFolder)
#    
    
    
#    sim_par([fNs[5]],[param_list[5]],nCore,T,dataFolder)
#    plot_time_traj(fNs[-1])
#    plot_MI(fNs[-1])
#    plot_time_traj(fNs[1])
#    plot_MI(fNs[1])


############### MULTIPLE TRAJ Linear PLOTTING ###############
#    NN=10
#    fN_stub='channel_capacity_linear_'
#    fNs=[fN_stub+str(i) for i in range(NN)]
#    plot_MI_bound(fNs)



############### MULTIPLE TRAJ Linear #####################
#    T=500
#    t_cutoff=T*1/5.
#    nCore=32
#    NN=100
#    rates=[]
#    states=[]
#    alpha_xs=10**(3*np.random.rand(NN))
#    alpha_cs=10**(3*np.random.rand(NN)-1)
#    
#    for alpha_x,alpha_c in zip(alpha_xs,alpha_cs):
#        rates=rates+[{'alpha_C':alpha_c,
#                      'mu_X':1,
#                      'alpha_X':alpha_x,
#                      'mu_C':1,
#                      'alpha_C_func':"alpha_C_func_linear",
#                      'mu_C_func':"lambda x,rate,c:0"
#                      }]
#        states=states+[{'X':round(alpha_x),'C':0,'X_est':alpha_x,'MI':0}]
#    param_list=[{'rate':rate,'state':state,'keykey':'production'} for rate,state in zip(rates,states)]
#    fN_stub='channel_capacity_linear_'
#    fNs=[fN_stub+str(i) for i in range(NN)]
#    sim_par(fNs,param_list,nCore,T,dataFolder)
#    
############### ONE TRAJ ##########################
#    T=100
#    t_cutoff=T*2/5.
#    nCore=1
#    alpha_C=10
#    rate={'alpha_C':alpha_C,
#          'alpha_X':10,
#          'mu_X':1,
#          'mu_C':1,
#          'mu_C_func':"lambda x,rate:0"
#          }
#    state={'X':10,'C':0,'X_est':10,'MI':0}
#    param_list=[{'rate':rate,'state':state,'keykey':'production'}]
#    state_init,rate=init_simple_default(param=param_list[0])
#    for key in ['alpha_C_func','alpha_X_func','mu_X_func','mu_C_func']:
#        rate[key]=eval(rate[key])
#    g_sim=gillespie(state_init,rate,al_simple,particle_gen)
#    
#    
#    fN='channel_capacity_linear_test_100'
#    sim(g_sim,fN,T,dataFolder)
#    plot_time_traj(fN)
#    plot_MI(fN)
#    pdb.set_trace()

