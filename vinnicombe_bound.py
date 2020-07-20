#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 12:52:57 2017

@author: xfz
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_bound():
    bound=lambda x: 1/(2*(1-np.log(x)))
    queue=lambda x: x*(1+x)/(1-x)**2
    mse=lambda x:x/(1-x)**2
    
    xx=np.linspace(0,0.3,1000)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(xx,bound(xx),'-r',lw=2)
    ax.plot(xx,queue(xx),'-b',lw=2)
    ax.plot(xx,mse(xx),'-k',lw=2)
    plt.show()
    
if __name__=='__main__':
    plot_bound()