import star
import numpy as np
from numpy import pi
from constants import *
from thermoCache import *
import os.path
import pickle

q = 4.5 # Kramer's P,T opacity law
gA = 0.4 # Adiabatic gradient in ionized matter
x = 0.7
y = 0.27

if not os.path.exists('cachesLowRes'):
	thermcache = thermCache(x,y,resRho=100,resT=100)
	rhocache = rhoCache(thermcache)
	convcache = convGradCache()
	pickle.dump([thermcache,rhocache,convcache],open('cachesLowRes','w+'))
else:
	thermcache,rhocache,convcache = pickle.load(open('cachesLowRes','rb'))

m = 1.0
r = 1.0


st = star.star(x,y,mSun*m,r*rSun,0,1.5,thermcache,rhocache,convcache,delM=r**2/m/32702.6,lext=lSun,minRes=350,caution=50,quiet=True)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
st.plot('steady','sigma','t',True,True,'','',ax)
plt.show()
