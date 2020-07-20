#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:18:02 2017

@author: xfz
"""

import matplotlib.pyplot as plt
import numpy as np

delta = 0.025
xrange = np.arange(-5.0, 20.0, delta)
yrange = np.arange(-5.0, 20.0, delta)
X, Y = np.meshgrid(xrange,yrange)

# F is one side of the equation, G is the other
F = 

plt.contour(X, Y, F, [0])
plt.show()