import star
import numpy as np
from numpy import pi
from constants import *
from thermoCache import *
import matplotlib.pyplot as plt
from mpltools import style
from mpltools import layout
import os

style.use('ggplot')

# Compute stellar structures

x = 0.7
y = 0.27
thermcache = thermCache(x,y)
rhocache = rhoCache(thermcache)
convcache = convGradCache()
masses = [2.0,0.3]
rs = [2.0**0.9,0.43]
ls = lSun*np.array(masses)**3.5
delM = [0.05,0.05]
caution = 500
lext = [10*lSun,0.1*lSun]
stGrid = []
for i in range(2):
		st1 = star.star(x,y,mSun*masses[i],rSun*rs[i],ls[i],1.5,thermcache,rhocache,convcache,\
			delM=delM[i],lext=lext[i],caution=caution)
		st2 = star.star(x,y,mSun*masses[i],rSun*rs[i],ls[i],1.5,thermcache,rhocache,convcache,\
			delM=delM[i],lext=0,caution=caution)
		stGrid.append([st2,st1])

plt.figure(figsize=(10,8))
stGrid[0][0].plot('steady','sigma','t',True,True,'','',plt.subplot(111))
stGrid[0][1].plot('steady','sigma','t',True,True,'','',plt.subplot(111))
plt.savefig('Plots/apkerM=2T.eps')
plt.figure(figsize=(10,8))
stGrid[0][1].plot('steady','sigma','kappa',True,True,'','',plt.subplot(111))
stGrid[0][0].plot('steady','sigma','kappa',True,True,'','',plt.subplot(111))
plt.savefig('Plots/apkerM=2kappa.eps')
plt.figure(figsize=(10,8))
stGrid[1][0].plot('steady','sigma','t',True,True,'','',plt.subplot(111))
stGrid[1][1].plot('steady','sigma','t',True,True,'','',plt.subplot(111))
plt.savefig('Plots/apkerM=0.3T.eps')
