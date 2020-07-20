
def plot_bound(picN,x1var,x2var,fNs,t_cutoff,param_list,clr,rate,input_var):
    
    x2_birth_events=[(('E',+1),)]
    tau1=lambda x2: 1/(rate['k'])
    tau1p=lambda x2: 1/(rate['k']+rate['k_E']*(x2))
    x1_cvs,x1_means,x2_means,x2_births,x2p_births=get_cv(fNs,x1var,x2var,x2_birth_events,tau1,tau1p)
    
    paulsson_bound_func=lambda x:2/(1+np.sqrt(1+4*x))
    n1=x1_means
    n2=x2_births
    n2p=x2p_births
    x_axis=n2/n1
    xp_axis=n2p/n1
    y_axis=x1_cvs*x1_means # fano factor
    xx=np.logspace(-5,3,num=100)
    
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(x_axis,y_axis,'ob',markersize=10,color=clr,label='tau1 without x2')
    ax.plot(xp_axis,y_axis,'sb',markersize=10,color=clr,label='tau1 with x2')
    ax.plot(xx,paulsson_bound_func(xx),'-r',lw=2,label='Paulsson Bound')
    ax.set_xlabel('N2/N1')
    ax.set_ylabel('Fano for '+x1var)
    ax.set_yscale('log')
    ax.set_xscale('log')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1.,1.02),borderaxespad=0.0)
    plt.savefig(picN+'_x1_'+x1var+'_x2_'+x2var+'_Bound.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  


def get_cv(fNs,x1var,x2var,x2_birth_events,tau1,tau1p,rate):
    x1_means=np.zeros(len(fNs))
    x2_means=np.zeros(len(fNs))
    x1_cvs=np.zeros(len(fNs))
    n2s=np.zeros(len(fNs))
    n2ps=np.zeros(len(fNs))
    for fN,i in zip(fNs,range(len(fNs))):
        aa=pd.DataFrame.from_csv(dataFolder+fN)
        t_mask=aa['t']>t_cutoff
        x1x1=aa[x1var].loc[t_mask]
        x2x2=aa[x2var].loc[t_mask]
        x2_means[i]=np.mean(x2x2)
        x1_means[i]=np.mean(x1x1)
        x1_cvs[i]=(np.std(x1x1)/x1_means[i])**2
        
        bb=pickle.load(open(dataFolder+fN+'_event','rb'))
        num_tau1=(aa['t'].values[-1]-t_cutoff)/tau1(x2_means[i])
        num_tau1p=(aa['t'].values[-1]-t_cutoff)/tau1p(x2_means[i])
        n2s[i]=np.sum([bb[x] for x in x2_birth_events])/num_tau1
        n2ps[i]=np.sum([bb[x] for x in x2_birth_events])/num_tau1p
    return x1_cvs,x1_means,x2_means,n2s,n2ps

def plot_compare(picN,x1var,x2var,fN_dic,t_cutoff,param_list,clr_dic,rate,input_var_dic):
    # here's the format for the dic inputs:
    # fN_dic={'deg':fNs,'noDeg':fNs}
    # clr_dic={'deg':clr,'noDeg':clr}
    # similar format for rate_dic,input_var_dic
#    data={'deg':{'x1_cvs':[],'x1_means':[],'x2_means':[],'x2_births':[],'x2p_births':[]},
#          'noDeg':{'x1_cvs':[],'x1_means':[],'x2_means':[],'x2_births':[],'x2p_births':[]}}
    data={'deg':{},'noDeg':{}}
    
    x2_birth_events=[(('E',+1),)]
    tau1=lambda x2,rate: 1/(rate['k'])
    tau1p=lambda x2,rate: 1/(rate['k']+rate['k_E']*(x2))
    for key in fN_dic.keys():
        data[key]['x1_cvs'],data[key]['x1_means'],data[key]['x2_means'],data[key]['x2_births'],data[key]['x2p_births']=get_cv(fN_dic[key],x1var,x2var,x2_birth_events,tau1,tau1p,rate)
        data[key]['n1']=data[key]['x1_means']
        data[key]['n2']=data[key]['x2_births']
        data[key]['n2p']=data[key]['x2p_births']
        data[key]['x_axis']=data[key]['n2']/data[key]['n1']
        data[key]['xp_axis']=data[key]['n2p']/data[key]['n1']
        data[key]['y_axis']=data[key]['x1_cvs']*data[key]['x1_means'] # fano factor
        
    paulsson_bound_func=lambda x:2/(1+np.sqrt(1+4*x))
    xx=np.logspace(-5,3,num=100)
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for key in fN_dic.keys():
        ax.plot(data[key]['x_axis'],data[key]['y_axis'],'o',markersize=10,color=clr_dic[key],label=key+' - tau1 without x2')
        ax.plot(data[key]['xp_axis'],data[key]['y_axis'],'s',markersize=10,color=clr_dic[key],label=key + ' - tau1 with x2')
    ax.plot(xx,paulsson_bound_func(xx),'-r',lw=2,label='Paulsson Bound')
    ax.set_xlabel('N2/N1')
    ax.set_ylabel('Fano for '+x1var)
    ax.set_yscale('log')
    ax.set_xscale('log')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1.,1.02),borderaxespad=0.0)
    plt.savefig(picN+'_x1_'+x1var+'_x2_'+x2var+'_Bound.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  
    pickle.dump(data,open(picN+'_x1_'+x1var+'_x2_'+x2var+'_Bound.p','bw'))


def cv_sim_par(arg):
    fN=arg['fN']
    if arg['model']=='deg':
        init_state,rate,input_var=init_deg_default(arg['param'])
        hs_ss=gillespie(init_state,rate,input_var,al_deg)
    elif arg['model']=='noDeg':
        init_state,rate,input_var=init_noDeg_default(arg['param'])
        hs_ss=gillespie(init_state,rate,input_var,al_noDeg)
    else: raise Exception('Invalid Model Specification')
    T=arg['T']
    st=time.time()
    sim(hs_ss,fN,T,dataFolder)
    ed=time.time()
    print(ed-st)
    print((ed-st)/T/60)

def cv_sim(fN_dic,param_dic,nCore,T,dataFolder):
    # param_list each element is a dictionary ((key,value),(key,value),...) corresponding to entries to be changed in rate dict.
    argsDeg=[{'dataFolder':dataFolder,'T':T,'param':x,'model':'deg','fN':fN+'_'+str(i)} for x,i,fN in zip(param_list,range(len(param_list)),fN_dic['deg'])]
    argsNoDeg=[{'dataFolder':dataFolder,'T':T,'param':x,'model':'noDeg','fN':fN+'_'+str(i)} for x,i,fN in zip(param_list,range(len(param_list)),fN_dic['noDeg'])]
    args=argsDeg+argsNoDeg
#    args=argsFull
    if nCore==1:
        for arg in args:
            cv_sim_par(arg)
    else:
        pool=Pool(nCore)
        pool.map(cv_sim_par,args)