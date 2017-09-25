import star
import numpy as np
from numpy import pi
from constants import *
from thermoCache import *

x = 0.7
y = 0.27
thermcache = thermCache(x,y)
rhocache = rhoCache(thermcache)
convcache = convGradCache()

st = star.star(x,y,0.01*mSun,0.1*rSun,3e-6*lSun,1.5,thermcache,rhocache,convcache,lext=0, delM=0.7, rot=1e-10)
r0 = st.state[:,2]
t0 = st.state[:,0]
print('Temp',t0[-1])
print('R',r0[-1])
st = star.star(x,y,0.01*mSun,0.1*rSun,3e-6*lSun,1.5,thermcache,rhocache,convcache,lext=0, delM=0.7, rot=1e-4)
r0 = st.state[:,2]
t0 = st.state[:,0]
print('Temp',t0[-1])
print('R',r0[-1])




