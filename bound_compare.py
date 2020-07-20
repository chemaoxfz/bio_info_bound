#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:19:55 2017

@author: xfz
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig=plt.figure()
ax=fig.add_subplot(111)
xx=np.logspace(-3,3,num=100)
paulsson_bound_func=lambda x:2/(1+np.sqrt(1+4*x))
ax.plot(xx,paulsson_bound_func(xx),'-r',lw=2,label='Birth Control Bound')
yorie_bound_func=lambda x:1/(1+np.sqrt(1+2*x))
ax.plot(xx,yorie_bound_func(xx),'-b',lw=2,label='Degradation Control Bound')
ax.set_xlabel('Nc')
ax.set_ylabel('Var/Mean')
ax.set_yscale('log')
ax.set_xscale('log')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc=3, bbox_to_anchor=(0.,1.02,1.,1.02),borderaxespad=0.0)
plt.savefig('bound_compare.png', bbox_extra_artists=(lgd,), bbox_inches='tight')  