import os
os.chdir('../')
import sys
sys.path.append('./')
import numpy as np
from numpy import pi
import star
from constants import *
from thermoCache import *
import matplotlib.pyplot as plt
from mpltools import style
from mpltools import layout
#style.use('ggplot')

fig = plt.figure(figsize=(10,12))
lines = []
labels = []

x = 0.7
y = 0.27
thermcache = thermCache(x,y)
rhocache = rhoCache(thermcache)
convcache = convGradCache()

color=plt.cm.rainbow(np.linspace(0,1,101))

st = star.star(x,y,mSun,rSun,1*lSun,1.5,thermcache,rhocache,convcache,lext=1*lSun,caution=500)
eps = st.eps0
t0 = st.state[:,0]
for i in range(20):
	st.stepController(1e8,eps)
	t0 = np.copy(st.state[:,0])
for i in range(101):
	if i%10==0:
		plt.subplot(211)
		plt.title('$L_{in}=L_{sun}, \Delta t=1e8s$')
		plt.plot(np.log10(st.state[:,24]),(st.state[:,0]-t0)/t0,color = color[i])
		plt.ylabel('$\Delta T/T$')
		plt.subplot(212)
		line, = plt.plot(np.log10(st.state[:,24]),st.l[:-1]/lSun,color = color[i])
		lines.append(line)
		labels.append('t='+str(i/10)+'e7s')
		plt.ylabel('$L/L_{sun}$')
		plt.xlabel('Log $\Sigma$')
	st.stepController(1e6,(100-i)*eps/100)
	print 'Step',i
fig.legend(lines,labels,loc='right')
plt.savefig('Plots/1Lsun_short_subtractive.eps',dpi=200)